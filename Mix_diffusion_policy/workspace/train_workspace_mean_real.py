if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import numpy as np
import random
import wandb
import tqdm
import shutil
import time
from Mix_diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from Mix_diffusion_policy.workspace.base_workspace import BaseWorkspace
from Mix_diffusion_policy.policy.Mix_diffusion_policy_mean import MixPolicy
from Mix_diffusion_policy.dataset.base_dataset import BasePcdDataset
from Mix_diffusion_policy.env_runner.base_pcd_runner import BasePcdRunner
from Mix_diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from Mix_diffusion_policy.common.json_logger import JsonLogger
from Mix_diffusion_policy.model.diffusion.ema_model_hdp import EMAModel
from Mix_diffusion_policy.model.common.lr_scheduler import get_scheduler
from Mix_diffusion_policy.common.robot import sigmoid, compute_reward_nextSubgoal_from_subgoal

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainWorkspace(BaseWorkspace):
    include_keys = [
        'global_step_guider', 
        'global_step_critic', 
        'global_step_actor', 
        'epoch_guider', 
        'epoch_critic', 
        'epoch_actor']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # ************ configure model ************
        self.model:MixPolicy = hydra.utils.instantiate(cfg.policy)
        self.use_ema = cfg.training.use_ema
        self.ema_model: MixPolicy = None
        if cfg.training.use_ema:
            try:
                self.ema_model_guider = copy.deepcopy(self.model.guider_model)
                self.ema_model_actor = copy.deepcopy(self.model.actor_model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model_guider = hydra.utils.instantiate(cfg.policy)
                self.ema_model_actor = hydra.utils.instantiate(cfg.policy)

        # configure training state
        self.optimizer_guider = hydra.utils.instantiate(cfg.optimizer_guider, params=self.model.guider_model.parameters())
        self.optimizer_actor = hydra.utils.instantiate(cfg.optimizer_actor, params=self.model.actor_model.parameters())

        self.global_step_guider = 0
        self.global_step_actor = 0
        self.epoch_guider = 0
        self.epoch_actor = 0
        self.train_loss_guider=[]
        self.test_loss_guider=[]


    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        # ************ 加载模型权重 ************
        # if cfg.training.resume:
        #     lastest_ckpt_path = self.get_checkpoint_path()
        #     if lastest_ckpt_path.is_file():
        #         print(f"Resuming from checkpoint {lastest_ckpt_path}")
        #         self.load_checkpoint(path=lastest_ckpt_path)

        if cfg.guider_path is not None:
            guider_path = cfg.guider_path
            if not cfg.guider_path.endswith(".ckpt"):
                guider_path = pathlib.Path(guider_path).joinpath('checkpoints', 'guider_latest.ckpt')
            self.load_checkpoint_guider(pathlib.Path(guider_path))

        if cfg.actor_path is not None:
            actor_path = cfg.actor_path
            if not cfg.actor_path.endswith(".ckpt"):
                actor_path = pathlib.Path(actor_path).joinpath('checkpoints', 'actor_latest.ckpt')
            self.load_checkpoint_actor(pathlib.Path(actor_path))
        

        # configure dataset
        # ************ 数据集 ************
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        train_dataloader_noshuff = DataLoader(dataset, **cfg.dataloader_noshuff)
        normalizer = dataset.get_normalizer()   # 归一化
        self.model.set_normalizer(normalizer)

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # configure lr scheduler
        # ************ 设置 lr scheduler ************
        # pytorch assumes stepping LRScheduler every epoch
        # however huggingface diffusers steps it every batch
        lr_scheduler_guider = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer_guider,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            # num_training_steps=(len(train_dataloader) * cfg.training.num_epochs),
            num_training_steps=cfg.training.num_steps,
            last_epoch=self.global_step_guider-1
        )
        lr_scheduler_actor = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer_actor,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs),
            last_epoch=self.global_step_actor-1
        )
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema_guider = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model_guider)
            ema_actor = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model_actor)
        # configure env runner
        # ************ 设置任务的仿真环境 ************
        # if cfg.test_run:
        #     cfg.task.env_runner.n_envs = 1
        
        if cfg.test_run:
            env_runner: BasePcdRunner
            env_runner = hydra.utils.instantiate(
                cfg.env_runner)
            score = self.test_score(env_runner,test_nums=cfg.test_times)
            print(f"Test score: {score}")
            return 
        
        if cfg.test_guider:
            for batch_idx, batch in enumerate(train_dataloader_noshuff):
                #self.model.test_critic(batch)
                self.model.test_guider(batch)
            return 

        # configure logging
        # ************ 设置wandb ************
        cfg.logging.name += '_'+cfg.train_model
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update({"output_dir": self.output_dir,})

        # configure checkpoint
        # ************ 保存权重的路径生成器，动态保存topk个权重 ************
        # topk_manager = TopKCheckpointManager(
        #     save_dir=os.path.join(self.output_dir, 'checkpoints'),
        #     **cfg.checkpoint.topk
        # )

        # device transfer
        # ************ 迁移到指定设备 ************
        device = torch.device(cfg.training.device)
        self.model.to(device)
        optimizer_to(self.optimizer_guider, device)
        optimizer_to(self.optimizer_actor, device)
        if cfg.training.use_ema and self.ema_model_actor is not None and self.ema_model_guider is not None:
            ema_guider.set_device(device)  # 确保EMA模型参数在GPU上
            ema_actor.set_device(device)
        # ************ 在数据集中测试guider ************
        
        if cfg.test_guider:
            for batch_idx, batch in enumerate(train_dataloader_noshuff):
                #self.model.test_critic(batch)
                self.model.test_guider(batch)

        # save batch for sampling
        train_sampling_batch = None
        log_path = os.path.join(self.output_dir, 'logs.json.txt')

        if cfg.train_model == 'guider':
            # ************ 训练guider *************
            # training loop
            with JsonLogger(log_path) as json_logger:
                # ************ 遍历epoch开始训练 ************
                while True:
                    if self.global_step_guider >= cfg.training.num_steps:
                        break
                    step_log = dict()
                    # ************ train for this epoch ************
                    train_losses_subgoal = list()
                    with tqdm.tqdm(train_dataloader, desc=f"Training Guider - epoch {self.epoch_guider}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            # device transfer
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            if train_sampling_batch is None:
                                train_sampling_batch = batch

                            # optimize guider
                            raw_loss_subgoal = self.model.guider_compute_loss(batch)
                            self.optimizer_guider.zero_grad()
                            raw_loss_subgoal.backward()
                            self.optimizer_guider.step()
                            if self.use_ema:
                                ema_guider.step(self.model.guider_model)
                            lr_scheduler_guider.step()
                            
                            # logging
                            loss_subgoal = raw_loss_subgoal.item()
                            tepoch.set_postfix(loss=loss_subgoal, refresh=False)
                            train_losses_subgoal.append(loss_subgoal)
                            
                            step_log = {
                                'train_loss_subgoal': loss_subgoal,
                                'global_step_guider': self.global_step_guider,
                                'epoch_guider': self.epoch_guider,
                                'lr_guider': lr_scheduler_guider.get_last_lr()[0]
                            }

                            is_last_batch = (batch_idx == (len(train_dataloader)-1))
                            if not is_last_batch:
                                # log of last step is combined with validation and rollout
                                wandb_run.log(step_log, step=self.global_step_guider)
                                json_logger.log(step_log)
                                self.global_step_guider += 1

                            if (cfg.training.max_train_steps is not None) \
                                and batch_idx >= (cfg.training.max_train_steps-1):
                                break
                    
                    # at the end of each epoch, replace train_loss with epoch average
                    step_log['train_loss_subgoal'] = np.mean(train_losses_subgoal)

                    # ************ 在测试集中测试loss ************
                    # run validation
                    if (self.epoch_guider % cfg.training.val_every) == 0:
                        with torch.no_grad():
                            val_losses = list()
                            with tqdm.tqdm(val_dataloader, desc=f"Validation Guider - epoch {self.epoch_guider}", 
                                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                                for batch_idx, batch in enumerate(tepoch):
                                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                    loss = self.model.guider_compute_loss(batch)
                                    val_losses.append(loss)
                                    if (cfg.training.max_val_steps is not None) \
                                        and batch_idx >= (cfg.training.max_val_steps-1):
                                        break
                            if len(val_losses) > 0:
                                val_loss = torch.mean(torch.tensor(val_losses)).item()
                                # log epoch average validation loss
                                step_log['val_loss_subgoal'] = val_loss

                    # run diffusion sampling on a training batch
                    # ************ 在一个训练batch上测试完整逆扩散过程的损失 ************
                    if (self.epoch_guider % cfg.training.sample_every) == 0:
                        with torch.no_grad():
                            batch = train_sampling_batch    # Tensor, no norm
                            start_time=time.time()
                            pred_subgoal = self.model.guider_predict_target(batch)
                            end_time=time.time()
                            step_log['sampling_time'] = end_time-start_time
                            print('sampling time =', end_time-start_time)
                            target_subgoal = batch['subgoal'][:, :pred_subgoal.shape[1]]
                            mse_subgoal = torch.nn.functional.mse_loss(pred_subgoal, target_subgoal)
                            step_log['train_mse_error_subgoal'] = mse_subgoal.item()

                            # release RAM
                            del batch
                            del pred_subgoal, mse_subgoal
                    
                    # ************ checkpoint ************
                    if (self.epoch_guider % cfg.training.checkpoint_every) == 0 or self.epoch_actor==cfg.training.num_epochs-1:
                        # checkpointing
                        # 保存最新的权重
                        tag = 'guider_latest'
                        if cfg.checkpoint.save_last_ckpt:   # <--
                            self.save_checkpoint(tag=tag)
                        if cfg.checkpoint.save_last_snapshot:
                            self.save_snapshot(tag=tag)

                    # end of epoch
                    # log of last step is combined with validation and rollout
                    wandb_run.log(step_log, step=self.global_step_guider)
                    json_logger.log(step_log)
                    self.global_step_guider += 1
                    self.epoch_guider += 1

        if cfg.train_model == 'actor':
            # ********** 训练actor **********
            # training loop
            next_actions = None
            best_score = -float('inf')
            with JsonLogger(log_path) as json_logger:
                # ************ 遍历epoch开始训练 ************
                for local_epoch_idx in range(cfg.training.num_epochs):
                    if self.epoch_actor >= cfg.training.num_epochs:
                        break
                    step_log = dict()
                    # ************ train for this epoch ************
                    train_losses_actor = list()

                    with tqdm.tqdm(train_dataloader, desc=f"Training Actor - epoch {self.epoch_actor}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            # device transfer
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            if train_sampling_batch is None:
                                train_sampling_batch = batch

                            # optimize actor
                            raw_actor_loss = self.model.actor_compute_loss(batch)
                            raw_actor_loss.backward()

                            self.optimizer_actor.step()
                            self.optimizer_actor.zero_grad()
                            lr_scheduler_actor.step()
                            
                            if self.use_ema:
                                ema_actor.step(self.model.actor_model)
                            
                            actor_loss =\
                                raw_actor_loss.item()

                            # logging
                            tepoch.set_postfix(loss=actor_loss, refresh=False)
                            train_losses_actor.append(actor_loss)
                            
                            step_log = {
                                'train_loss_actor': actor_loss,
                                'global_step_actor': self.global_step_actor,
                                'epoch_actor': self.epoch_actor,
                                'lr_actor': lr_scheduler_actor.get_last_lr()[0]
                            }

                            is_last_batch = (batch_idx == (len(train_dataloader)-1))
                            if not is_last_batch:
                                # log of last step is combined with validation and rollout
                                wandb_run.log(step_log, step=self.global_step_actor)
                                json_logger.log(step_log)
                                self.global_step_actor += 1
                    
                    # at the end of each epoch, replace train_loss with epoch average
                    step_log['train_loss_actor'] = np.mean(train_losses_actor)

                    # ************ 更新next_action ************
                    # if cfg.policy.next_action_mode == 'pred_global':
                    #     if (self.epoch % cfg.training.updateAction_every) == 0:
                    #         with torch.no_grad():
                    #             pred_next_actions = list()
                    #             with tqdm.tqdm(train_dataloader_noshuff, desc=f"Update next actions", 
                    #                 leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch_noshuff:
                    #                 for batch_idx, batch in enumerate(tepoch_noshuff):
                    #                     batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    #                     pred_next_action = self.model.predict_next_action(batch)  # (B, T, A)
                    #                     pred_next_actions.append(pred_next_action)
                    #             next_actions = torch.concat(pred_next_actions, dim=0)

                    # run rollout
                    # ************ 在仿真环境中测试 ************
                    '''
                    if (self.epoch_actor % cfg.training.rollout_every) == 0:        
                        runner_log = env_runner.run(self.model, first=True)
                        runner_log = env_runner.run(self.model)
                        print('runner_log =', runner_log)
                        if runner_log['test/mean_score'] >= best_score:
                            tag = 'actor_best'
                            best_score = runner_log['test/mean_score']
                            if cfg.checkpoint.save_last_ckpt:   # <--
                                self.save_checkpoint(tag=tag)
                        wandb_run.log(runner_log, step=self.global_step_actor)'''
                    

                    # ************ 在测试集中测试loss ************
                    # run validation
                    if (self.epoch_actor % cfg.training.val_every) == 0:
                        with torch.no_grad():
                            val_losses = list()
                            val_bc_losses = list()
                            with tqdm.tqdm(val_dataloader, desc=f"Validation Actor - epoch {self.epoch_actor}", 
                                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                                for batch_idx, batch in enumerate(tepoch):
                                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                    loss= self.model.actor_compute_loss(batch)
                                    pred_action = self.model.actor_predict_action(batch)['action_pred']
                                    target_action = batch['action']
                                    mse_action = torch.nn.functional.mse_loss(pred_action,target_action)
                                    val_losses.append(loss)
                                    val_bc_losses.append(mse_action)
                            if len(val_losses) > 0:
                                val_loss = torch.mean(torch.tensor(val_losses)).item()
                                val_bc_loss = torch.mean(torch.tensor(val_bc_losses)).item()
                                # log epoch average validation loss
                                step_log['val_loss_action'] = val_loss
                                step_log['val_mse_error_action'] = val_bc_loss  

                    # run diffusion sampling on a training batch
                    # ************ 在一个训练batch上测试完整逆扩散过程的损失 ************
                    if (self.epoch_actor % cfg.training.sample_every) == 0:
                        with torch.no_grad():
                            batch = train_sampling_batch    # Tensor, no norm
                            start_time=time.time()
                            pred_action = self.model.actor_predict_action(batch)['action_pred']
                            target_action = batch['action']
                            end_time=time.time()
                            step_log['sampling_time'] = end_time-start_time
                            print('sampling time =', end_time-start_time)
                            mse_action = torch.nn.functional.mse_loss(pred_action, target_action)
                            step_log['train_mse_error_action'] = mse_action.item()
                            #print(f"pred_action:{pred_action}")
                            #print(f"target_action:{target_action}")
                            step_log['train_mse_error_action'] = mse_action.item()
                            print('train_mse_error_action =', mse_action.item())
                            
                            # release RAM
                            del batch
                            del pred_action, mse_action
                    
                    # ************ checkpoint ************
                    if (self.epoch_actor % cfg.training.checkpoint_every) == 0:
                        # checkpointing
                        # 保存最新的权重
                        tag = 'actor_latest'
                        if cfg.checkpoint.save_last_ckpt:   # <--
                            self.save_checkpoint(tag=tag)
                        if cfg.checkpoint.save_last_snapshot:
                            self.save_snapshot(tag=tag)
                    if cfg.training.num_epochs - self.epoch_actor < 500:
                        # 每50个epoch保存一次
                        if (self.epoch_actor % 50) == 0:
                            # 保存带epoch号的checkpoint
                            epoch_tag = f'actor_epoch_{self.epoch_actor}'
                            if cfg.checkpoint.save_last_ckpt:   # <--
                                self.save_checkpoint(tag=epoch_tag)
                            if cfg.checkpoint.save_last_snapshot:
                                self.save_checkpoint(tag=epoch_tag)
                    # end of epoch
                    # log of last step is combined with validation and rollout
                    wandb_run.log(step_log, step=self.global_step_actor)
                    json_logger.log(step_log)
                    self.global_step_actor += 1
                    self.epoch_actor += 1
    
        wandb.finish()
    

    def test_score(self, env_runner,test_nums=10):
        score = 0
        total_tests = test_nums
        
        for i in range(total_tests):
            # 每次开始前询问是否运行
            while True:
                start = input(f"第 {i+1} 次测试：是否开始运行？(y/n): ").strip().lower()
                if start == 'y':
                    break
                elif start == 'n':
                    print("已跳过本次测试，不计分。")
                    break
                else:
                    print("无效输入，请输入 y 或 n。")
            
            if start != 'y':
                break
            
            # 运行模型
            env_runner.run(self.model)
            
            # 询问是否成功
            while True:
                success = input("本次运行是否成功？(y/n): ").strip().lower()
                if success == 'y':
                    score += 1
                    break
                elif success == 'n':
                    break
                else:
                    print("无效输入，请输入 y 或 n。")
            
            print(f"当前累计得分：{score}\n")
        
        print(f"最终得分：{score}/{total_tests}")
        return score