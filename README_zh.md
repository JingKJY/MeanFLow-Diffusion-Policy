# Meanflow Diffusion Policy（MDP）

[English](README.md) | [简体中文](README_zh.md)

Meanflow Diffusion Policy（简称 **MDP**）是一种面向机器人操作任务的策略，将 MeanFlow 动作生成与 Diffusion Policy 的观测编码器、条件建模模块和训练框架相结合。项目同时支持点云观测和低维观测，可用于仿真与实物机器人实验。

MDP 提供 Push-T、Robomimic、非抓取操作和实物机器人任务的训练及评估流程。项目包含策略实现、实验配置、环境接口、仿真资源以及可复用的训练工具。

## 包含的组件

- 面向点云、Push-T、Critic 和实物机器人实验的 MeanFlow 策略。
- Robomimic、Push-T、非抓取倾斜操作和实物机器人任务的 Hydra 配置。
- 数据集适配器、仿真环境、环境 Runner 和实物机器人 ZMQ 客户端。
- MDP 所需的 Diffusion Policy 编码器、归一化模块、Guider/Critic、EMA 和公共工具。
- 仿真环境所需的 MuJoCo/Robosuite XML、网格和纹理资源。

Python 导入命名空间为 `Mix_diffusion_policy`。项目与 Python 发行包的名称为 **Meanflow Diffusion Policy**，简称 **MDP**。

## 安装

项目面向配备 NVIDIA GPU 的 Linux 系统，已测试环境为 Ubuntu 20.04 和 Python 3.9。

```bash
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
conda env create -f conda_environment.yaml
conda activate Mixdiff
```

参照 HDP 的安装方式，以下两个固定版本的第三方包需要单独安装。将它们与 `conda_environment.yaml` 分开，可以更方便地定位第三方源码构建或网络下载问题：

```bash
pip install "robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git@277ab9588ad7a4f4b55cf75508b44aa67ec171f0"
pip install "r3m @ git+https://github.com/facebookresearch/r3m.git@b2334e726887fa0206962d7984c69c5fb09cceab"
pip install -e .
```

环境名称区分大小写，请使用 `Mixdiff`。环境文件已经包含 MDP 的运行依赖，包括 Open3D、TorchCFM、PyZMQ 和 `ur-analytic-ik`。如果仅运行仿真实验，则不需要启动或安装机器人硬件服务与驱动。

准备数据集前，可以执行以下命令验证核心环境：

```bash
python -c "import torch, diffusers, open3d, robosuite, robomimic, r3m; print('MDP environment OK')"
python -c "import Mix_diffusion_policy.policy.Mix_diffusion_policy_mean; print('MDP policy import OK')"
```

## 数据目录

项目不附带数据集和模型权重。请将它们放在 `data/` 目录下，或者通过命令行覆盖对应的 Hydra 配置字段。

```text
data/
├── robomimic/datasets/<task>/<dataset>/low_dim_abs_pcd.hdf5
├── nonprehensile/tilt_fast.h5
├── pusht/pusht_cchi_v7_replay.zarr
├── real/<experiment>/gello_dataset.h5
└── checkpoints/<experiment>/*.ckpt
```

实验配置位于 `Mix_diffusion_policy/config/Mean/`。配置文件中的路径均为示例，可以直接通过命令行覆盖，无需修改配置文件。

## 仿真实验

统一训练入口支持 `Mean` 配置目录中的所有实验配置：

```bash
python train_mdp.py \
  --config-dir=Mix_diffusion_policy/config/Mean \
  --config-name=Mix_pusht.yaml \
  training.seed=42 \
  training.device=cuda:0
```

Robomimic 实验示例：

```bash
python train_mdp.py \
  --config-dir=Mix_diffusion_policy/config/Mean \
  --config-name=Mix_square_ph.yaml \
  task.dataset.dataset_path=data/robomimic/datasets/square/ph/low_dim_abs_pcd.hdf5 \
  training.device=cuda:0
```

可以添加 `training.debug=true` 进行短时冒烟测试。实验输出位置由各配置文件中的 `hydra.run.dir` 字段控制。

## 实物机器人实验

文件名中包含 `_real` 的配置用于实物机器人实验。连接机器人之前，至少需要设置数据集路径、模型权重路径和机器人服务地址：

```bash
python train_mdp.py \
  --config-dir=Mix_diffusion_policy/config/Mean \
  --config-name=Mix_cylinder_real_2.yaml \
  task.dataset.dataset_path=data/real/cylinder_place/gello_dataset.h5 \
  env_runner.env.env_client.host=192.168.1.10 \
  training.device=cuda:0
```

仓库配置中的默认主机地址为安全占位值 `127.0.0.1`。启用机器人运动前，请根据实际平台检查工作空间限制、急停功能、坐标系、观测与动作维度以及网络端点。实物机器人执行与具体硬件平台相关，项目中的实现主要作为研究参考。

## 配置分类

| 分类 | 配置文件 |
| --- | --- |
| Push-T | `Mix_pusht.yaml` |
| Robomimic | Can、Lift、Square 和 Tool Hang 相关配置 |
| 实物机器人 | 方块堆叠、圆柱插入、放置和套环相关配置 |

## 数据和生成文件

项目不包含大型实验文件。请勿提交数据集、模型权重、W&B 运行目录、生成视频、本地 Hydra 输出或机器人凭据。发布时的默认忽略规则请参阅 `.gitignore`。

## 致谢

MDP 使用了 MeanFlow、Diffusion Policy、Hierarchical Diffusion Policy（HDP）、Robomimic 和 Robosuite 的相关思想与基础设施。使用本项目开展研究时，请引用这些项目；MDP 论文的引用信息公布后，也请一并引用。

## 许可证

项目采用 MIT 许可证发布。第三方软件包和资源仍遵循各自的许可证。
