import Mix_diffusion_policy.common.transformation as tf
import numpy as np
from Mix_diffusion_policy.common.visual import segmentation_to_rgb, getFingersPos, show_rgb_seg_dep, show_pcd,show_pcd_tf,show_two_pcds,show_pcd_finger
import robomimic.utils.file_utils as FileUtils
from scipy.spatial.transform import Rotation as R
import math
import shapely
import copy
import torch
from Mix_diffusion_policy.common.visual import create_pusht_pts


def get_subgoals_stage_moveT(
        state: dict,
        object_pcd: np.ndarray,
        fin_rad: float,
        sim_thresh: list,
        max_reward=10,
        Tr=1,
        reward_mode='tanh'
        ):
    """
    计算手指位置子目标, 不考虑平滑

    args:
        - state: np.ndarray (20,) {eef_pos, eef_quat, fingers_position，object_pos, object_quat}
        - object_pcd: object pointcloud, np.ndarray, shape=(N, 3)
        - fin_rad: 手指半径
        - sim_thresh: list(平移误差, 弧度误差) 计算物体位姿子目标是否达到的阈值，各任务单独设置
        - max_reward
        - reward_mode (str): 'only_success' or 'tanh'

    return:
        - subgoal: (N-1, 8) 对应每个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - next_subgoal: (N-1, 8) 对应下一个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - reward: (N-1,)
    """

    """
    (1) 初选：当手指与物体接触时记录手指位置, 当接触状态变化时记录物体位姿
    (2) 过滤：当记录的两次相邻物体位姿相似时(阈值很小)，删除前一个物体位姿，并将对应的手指位置设为全空
    (3) 配置：遍历状态，当物体到达记录的位姿时，设置对应的手指位置为子目标(对应指物体位姿索引或相同的索引)，
            如果手指位置为全空，则设置后面最近的手指位置为子目标
    """

    # ********** 初选子目标 **********
    is_last_fl_contact = False
    is_last_fr_contact = False
    obj_subgoals = list()   # 物体位姿子目标
    obj_subgoals_id = list()   # 物体位姿子目标索引
    fin_subgoals_obj_init = list()  # 手指位置子目标(物体坐标系下)

    contact_thresh = fin_rad + 0.01
    
    # (1) 初选：当手指与物体接触时记录手指位置, 当接触状态变化时记录物体位姿
    sequence_length = state.shape[0]
    for step in range(sequence_length):
        # fingers position
        fl_pos = state[step, 7:10]
        fr_pos = state[step, 10:13]
        # 计算手指与物体是否接触, 如果手指与点云的最小距离小于阈值，认为接触
        obj_pos = state[step, 13:16]
        obj_qua = state[step, 16:20]
        # 将手指位置转到物体坐标系
        T_W_O = tf.PosQua_to_TransMat(obj_pos, obj_qua)
        T_O_W = np.linalg.inv(T_W_O)
        fl_pos_obj = tf.transPt(fl_pos, T_f2_f1=T_O_W)
        fr_pos_obj = tf.transPt(fr_pos, T_f2_f1=T_O_W)
        # 计算点云到手指的距离
        fl_dists = object_pcd - fl_pos_obj
        fr_dists = object_pcd - fr_pos_obj
        fl_dist = np.min(np.sqrt(np.sum(np.square(fl_dists), axis=1)))
        fr_dist = np.min(np.sqrt(np.sum(np.square(fr_dists), axis=1)))

        is_fl_contact = fl_dist < contact_thresh
        is_fr_contact = fr_dist < contact_thresh

        if step > sequence_length-Tr-1:
            is_fl_contact = True
            is_fr_contact = True

        # 记录手指接触位置
        fin_subgoals_obj_init.append(
            np.concatenate((fl_pos_obj, fr_pos_obj, [is_fl_contact,], [is_fr_contact,])))
        # 记录物体位姿
        if (is_fl_contact != is_last_fl_contact) or (is_fr_contact != is_last_fr_contact):
            obj_subgoals.append(np.concatenate((obj_pos, obj_qua)))
            obj_subgoals_id.append(step)
        
        is_last_fl_contact = is_fl_contact
        is_last_fr_contact = is_fr_contact

    # ********** 子目标精简 **********
    # (2) 过滤：当记录的两次相邻物体位姿相似时，删除前一个物体位姿，并将对应的手指位置设为None
    i = 0
    while i < len(obj_subgoals)-1:
        obj_sim = check_poses_similarity(
            obj_subgoals[i], obj_subgoals[i+1], 
            pos_th=sim_thresh[0], euler_th=sim_thresh[1])   

        if obj_sim:
            # 直到下一个物体子目标之前的手指接触都设为0
            for s in np.arange(obj_subgoals_id[i], obj_subgoals_id[i+1]):
                fin_subgoals_obj_init[s] = np.zeros((8,))
            # 删除前一个物体子目标
            obj_subgoals.pop(i)
            obj_subgoals_id.pop(i)
        else:
            # 继续对比下一个
            i+=1
    # 子目标前移一位
    fin_subgoals_obj_init.pop(0)
    fin_subgoals_obj_init.append(np.zeros((8,)))


    # ********** 子目标配置 **********
    # (3) 配置：遍历状态，当手指到达子目标，且物体到达记录的位姿，时，设置对应的手指位置为子目标(对应指物体位姿索引或相同的索引)，
    #        如果手指位置为全空，则设置后面最近的手指位置为子目标
    fin_sgs = list()
    next_fin_sgs = list()
    reward = list()
    goal_thresh = fin_rad/2
    obj_sg_id = 0   # 已达到的物体位姿子目标
    last_done = 'obj'   # obj/fin
    r = 0

    for step in range(sequence_length-Tr):
        # 手指位置
        fl_pos = state[step, 7:10]
        fr_pos = state[step, 10:13]
        # 物体位姿
        obj_pos = state[step, 13:16]
        obj_qua = state[step, 16:20]
        obj_pose = np.concatenate((obj_pos, obj_qua))
        if last_done == 'obj':
            # 检测手指是否到达子目标
            if r == max_reward:
                last_done = 'fin'
        
        if last_done == 'fin' and obj_sg_id < len(obj_subgoals_id)-1:
            # 检测物体是否到达子目标
            obj_sim = check_poses_similarity(
                obj_pose, obj_subgoals[obj_sg_id+1], 
                pos_th=sim_thresh[0], euler_th=sim_thresh[1])
            if obj_sim: 
                obj_sg_id += 1
                last_done = 'obj'

        # 设置子目标
        try:
            fin_sg_id = max(obj_subgoals_id[obj_sg_id], step)
        except:
            fin_sg_id = step
        fin_sg = fin_subgoals_obj_init[fin_sg_id]

        # 记录世界坐标下的当前时刻的子目标
        fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_sg[6]
        fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_sg[7]
        fin_sgs.append(np.concatenate((fl_sg, fr_sg, [fin_sg[6],], [fin_sg[7],])))        
        # 记录下一时刻的子目标
        next_obj_pos = state[step+Tr, 13:16]
        next_obj_qua = state[step+Tr, 16:20]
        next_fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=next_obj_pos, q_f2_f1=next_obj_qua) * fin_sg[6]
        next_fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=next_obj_pos, q_f2_f1=next_obj_qua) * fin_sg[7]
        next_fin_sgs.append(np.concatenate((next_fl_sg, next_fr_sg, [fin_sg[6],], [fin_sg[7],])))

        # 计算reward
        #* 未来n步内，有一步到达goal，就设r=max
        for n in range(1, Tr+1):
            # 目标位置
            _next_obj_pos = state[step+n, 13:16]
            _next_obj_qua = state[step+n, 16:20]
            _next_fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=_next_obj_pos, q_f2_f1=_next_obj_qua) * fin_sg[6]
            _next_fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=_next_obj_pos, q_f2_f1=_next_obj_qua) * fin_sg[7]
            # 手指位置(世界坐标系下)
            _next_fl_pos = state[step+n, 7:10]
            _next_fr_pos = state[step+n, 10:13]
            # 手指距离 - 欧式距离
            _next_fl_dp = np.linalg.norm(_next_fl_pos - _next_fl_sg) * fin_sg[6]
            _next_fr_dp = np.linalg.norm(_next_fr_pos - _next_fr_sg) * fin_sg[7]
            if max(_next_fl_dp, _next_fr_dp) < goal_thresh:
                r = max_reward
                break
            else:
                if reward_mode == 'only_success':
                    r = 0
                elif reward_mode == 'tanh':
                    reward_weights = 3
                    r_fl = -np.tanh(_next_fl_dp * reward_weights)
                    r_fr = -np.tanh(_next_fr_dp * reward_weights)
                    r = (r_fl+r_fr) / 3 * 2 + 1
                else:
                    raise ValueError('reward_mode must be `only_success` or `tanh`')
        
        reward.append(r)

    return {
        'subgoal': np.array(fin_sgs),
        'next_subgoal': np.array(next_fin_sgs),
        'reward': np.array(reward)
    }


def get_subgoals_stage_nonprehensile(
        raw_obs: dict,
        object_pcd: np.ndarray,
        fin_rad: float,
        sim_thresh: list,
        max_reward=10,
        Tr=1,
        reward_mode='tanh'
        ):
    """
    计算手指位置子目标, 不考虑平滑

    args:
        - raw_obs: h5py dict {object_pos, object_quat, eef_pos, eef_quat, fingers_position}
        - object_pcd: object pointcloud, np.ndarray, shape=(N, 3)
        - fin_rad: 手指半径
        - sim_thresh: list(平移误差, 弧度误差) 计算物体位姿子目标是否达到的阈值，各任务单独设置
        - max_reward
        - reward_mode (str): 'only_success' or 'tanh'

    return:
        - subgoal: (N-1, 8) 对应每个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - next_subgoal: (N-1, 8) 对应下一个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - reward: (N-1,)
    """
    

    """
    (1) 初选：当手指与物体接触时记录手指位置, 当接触状态变化时记录物体位姿
    (2) 过滤：当记录的两次相邻物体位姿相似时(阈值很小)，删除前一个物体位姿，并将对应的手指位置设为全空
    (3) 配置：遍历状态，当物体到达记录的位姿时，设置对应的手指位置为子目标(对应指物体位姿索引或相同的索引)，
            如果手指位置为全空，则设置后面最近的手指位置为子目标
    """

    # ********** 初选子目标 **********
    is_last_fl_contact = False
    is_last_fr_contact = False
    obj_subgoals = list()   # 物体位姿子目标
    obj_subgoals_id = list()   # 物体位姿子目标索引
    fin_subgoals_obj_init = list()  # 手指位置子目标(物体坐标系下)

    contact_thresh = fin_rad + 0.01 #!!!
    
    # (1) 初选：当手指与物体接触时记录手指位置, 当接触状态变化时记录物体位姿
    sequence_length = raw_obs['object_pos'].shape[0]
    for step in range(sequence_length):
        # fingers position
        fl_pos = raw_obs['fingers_position'][step, :3]
        fr_pos = raw_obs['fingers_position'][step, 3:]
        # 计算手指与物体是否接触, 如果手指与点云的最小距离小于阈值，认为接触
        obj_pos = raw_obs['object_pos'][step]
        obj_qua = raw_obs['object_quat'][step]
        # 将手指位置转到物体坐标系
        T_W_O = tf.PosQua_to_TransMat(obj_pos, obj_qua)
        T_O_W = np.linalg.inv(T_W_O)
        fl_pos_obj = tf.transPt(fl_pos, T_f2_f1=T_O_W)
        fr_pos_obj = tf.transPt(fr_pos, T_f2_f1=T_O_W)
        # 计算点云到手指的距离
        fl_dists = object_pcd - fl_pos_obj
        fr_dists = object_pcd - fr_pos_obj
        fl_dist = np.min(np.sqrt(np.sum(np.square(fl_dists), axis=1)))
        fr_dist = np.min(np.sqrt(np.sum(np.square(fr_dists), axis=1)))

        is_fl_contact = fl_dist < contact_thresh
        is_fr_contact = fr_dist < contact_thresh
        # 记录手指接触位置
        fin_subgoals_obj_init.append(
            np.concatenate((fl_pos_obj, fr_pos_obj, [is_fl_contact,], [is_fr_contact,])))
        # 记录物体位姿
        if (is_fl_contact != is_last_fl_contact) or (is_fr_contact != is_last_fr_contact):
            obj_subgoals.append(np.concatenate((obj_pos, obj_qua)))
            obj_subgoals_id.append(step)
        
        is_last_fl_contact = is_fl_contact
        is_last_fr_contact = is_fr_contact


    # ********** 子目标精简 **********
    # (2) 过滤：当记录的两次相邻物体位姿相似时，删除前一个物体位姿，并将对应的手指位置设为None
    i = 0
    while i < len(obj_subgoals)-1:
        obj_sim = check_poses_similarity(
            obj_subgoals[i], obj_subgoals[i+1], 
            pos_th=sim_thresh[0], euler_th=sim_thresh[1])   

        if obj_sim:
            # 直到下一个物体子目标之前的手指接触都设为0
            for s in np.arange(obj_subgoals_id[i], obj_subgoals_id[i+1]):
                fin_subgoals_obj_init[s] = np.zeros((8,))
            # 删除前一个物体子目标
            obj_subgoals.pop(i)
            obj_subgoals_id.pop(i)
        else:
            # 继续对比下一个
            i+=1
    # 子目标前移一位
    fin_subgoals_obj_init.pop(0)
    fin_subgoals_obj_init.append(np.zeros((8,)))


    # ********** 子目标配置 **********
    # (3) 配置：遍历状态，当手指到达子目标，且物体到达记录的位姿，时，设置对应的手指位置为子目标(对应指物体位姿索引或相同的索引)，
    #        如果手指位置为全空，则设置后面最近的手指位置为子目标
    fin_sgs = list()
    next_fin_sgs = list()
    reward = list()
    goal_thresh = fin_rad/2
    obj_sg_id = 0   # 已达到的物体位姿子目标
    last_done = 'obj'   # obj/fin
    r = 0
    for step in range(sequence_length-Tr):
        # 手指位置
        fl_pos = raw_obs['fingers_position'][step, :3]
        fr_pos = raw_obs['fingers_position'][step, 3:]
        # 物体位姿
        obj_pos = raw_obs['object_pos'][step]
        obj_qua = raw_obs['object_quat'][step]
        obj_pose = np.concatenate((obj_pos, obj_qua))
        if last_done == 'obj':
            # 检测手指是否到达子目标
            if r == max_reward: #!
                last_done = 'fin'
        
        if last_done == 'fin' and obj_sg_id < len(obj_subgoals_id)-1:
            # 检测物体是否到达子目标
            obj_sim = check_poses_similarity(
                obj_pose, obj_subgoals[obj_sg_id+1], 
                pos_th=sim_thresh[0], euler_th=sim_thresh[1])
            if obj_sim: 
                obj_sg_id += 1
                last_done = 'obj'

        # 设置子目标
        fin_sg_id = max(obj_subgoals_id[obj_sg_id], step)
        fin_sg = fin_subgoals_obj_init[fin_sg_id]

        # 记录世界坐标下的当前时刻的子目标
        fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_sg[6]
        fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_sg[7]
        fin_sgs.append(np.concatenate((fl_sg, fr_sg, [fin_sg[6],], [fin_sg[7],])))        
        # 记录下一时刻的子目标
        next_obj_pos = raw_obs['object_pos'][step+Tr]
        next_obj_qua = raw_obs['object_quat'][step+Tr]
        next_fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=next_obj_pos, q_f2_f1=next_obj_qua) * fin_sg[6]
        next_fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=next_obj_pos, q_f2_f1=next_obj_qua) * fin_sg[7]
        next_fin_sgs.append(np.concatenate((next_fl_sg, next_fr_sg, [fin_sg[6],], [fin_sg[7],])))

        # 计算reward
        #* 未来n步内，有一步到达goal，就设r=max
        for n in range(1, Tr+1):
            # 目标位置
            _next_obj_pos = raw_obs['object_pos'][step+n]
            _next_obj_qua = raw_obs['object_quat'][step+n]
            _next_fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=_next_obj_pos, q_f2_f1=_next_obj_qua) * fin_sg[6]
            _next_fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=_next_obj_pos, q_f2_f1=_next_obj_qua) * fin_sg[7]
            # 手指位置(世界坐标系下)
            _next_fl_pos = raw_obs['fingers_position'][step+n, :3]
            _next_fr_pos = raw_obs['fingers_position'][step+n, 3:]
            # 手指距离 - 欧式距离
            _next_fl_dp = np.linalg.norm(_next_fl_pos - _next_fl_sg) * fin_sg[6]
            _next_fr_dp = np.linalg.norm(_next_fr_pos - _next_fr_sg) * fin_sg[7]
            if max(_next_fl_dp, _next_fr_dp) < goal_thresh:
                r = max_reward
                break
            else:
                if reward_mode == 'only_success':
                    r = 0
                elif reward_mode == 'tanh':
                    reward_weights = 3
                    r_fl = -np.tanh(_next_fl_dp * reward_weights)
                    r_fr = -np.tanh(_next_fr_dp * reward_weights)
                    r = (r_fl+r_fr) / 3 * 2 + 1
                else:
                    raise ValueError('reward_mode must be `only_success` or `tanh`')
        
        reward.append(r)

    return {
        'subgoal': np.array(fin_sgs),
        'next_subgoal': np.array(next_fin_sgs),
        'reward': np.array(reward)
    }


def get_subgoals_realtime_nonprehensile(
        raw_obs: dict,
        object_pcd: np.ndarray,
        fin_rad: float,
        max_reward=10,
        Tr=1,
        reward_mode='tanh'
        ):
    """
    计算手指位置子目标, 以后面时刻中第一个接触物体的接触点为子目标

    args:
        - raw_obs: h5py dict {object_pos, object_quat, eef_pos, eef_quat, fingers_position}
        - object_pcd: object pointcloud, np.ndarray, shape=(N, 3)
        - fin_rad: 手指半径
        - max_reward
        - reward_mode (str): 'only_success' or 'tanh'

    return:
        - subgoal: (N-1, 8) 对应每个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - next_subgoal: (N-1, 8) 对应下一个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - reward: (N-1,)
    """

    # ********** 记录接触位置 **********
    fin_subgoals_obj = list()  # 手指位置子目标(物体坐标系下)
    contact_thresh = fin_rad + 0.01
    sequence_length = raw_obs['object_pos'].shape[0]
    for step in range(sequence_length):
        # fingers position
        fl_pos = raw_obs['fingers_position'][step, :3]
        fr_pos = raw_obs['fingers_position'][step, 3:]
        # 计算手指与物体是否接触, 如果手指与点云的最小距离小于阈值，认为接触
        obj_pos = raw_obs['object_pos'][step]
        obj_qua = raw_obs['object_quat'][step]
        # 将手指位置转到物体坐标系
        T_W_O = tf.PosQua_to_TransMat(obj_pos, obj_qua)
        T_O_W = np.linalg.inv(T_W_O)
        fl_pos_obj = tf.transPt(fl_pos, T_f2_f1=T_O_W)
        fr_pos_obj = tf.transPt(fr_pos, T_f2_f1=T_O_W)
        # 计算点云到手指的距离
        fl_dists = object_pcd - fl_pos_obj
        fr_dists = object_pcd - fr_pos_obj
        fl_dist = np.min(np.sqrt(np.sum(np.square(fl_dists), axis=1)))
        fr_dist = np.min(np.sqrt(np.sum(np.square(fr_dists), axis=1)))

        is_fl_contact = fl_dist < contact_thresh
        is_fr_contact = fr_dist < contact_thresh
        if is_fl_contact or is_fr_contact:
            # 记录手指接触位置
            fin_subgoals_obj.append(
                np.concatenate((fl_pos_obj, fr_pos_obj, [is_fl_contact,], [is_fr_contact,])))
        else:
            fin_subgoals_obj.append(None)


    # ********** 子目标补全 **********
    for step in range(sequence_length)[:-1][::-1]:
        if fin_subgoals_obj[step] is None and fin_subgoals_obj[step+1] is not None:
            fin_subgoals_obj[step] = fin_subgoals_obj[step+1]
    fin_subgoals_obj.pop(0) # 子目标前移一位


    # ********** 子目标配置 **********
    # (3) 配置：遍历状态，当手指到达子目标，且物体到达记录的位姿，时，设置对应的手指位置为子目标(对应指物体位姿索引或相同的索引)，
    #        如果手指位置为全空，则设置后面最近的手指位置为子目标
    fin_sgs = list()
    next_fin_sgs = list()
    reward = list()
    goal_thresh = fin_rad/2
    for step in range(sequence_length-Tr):
        # 物体位姿
        obj_pos = raw_obs['object_pos'][step]
        obj_qua = raw_obs['object_quat'][step]
        # 设置子目标
        fin_sg = fin_subgoals_obj[step]

        # 记录世界坐标下的当前时刻的子目标
        fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_sg[6]
        fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_sg[7]
        fin_sgs.append(np.concatenate((fl_sg, fr_sg, [fin_sg[6],], [fin_sg[7],])))        
        # 记录下一时刻的子目标
        next_obj_pos = raw_obs['object_pos'][step+Tr]
        next_obj_qua = raw_obs['object_quat'][step+Tr]
        next_fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=next_obj_pos, q_f2_f1=next_obj_qua) * fin_sg[6]
        next_fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=next_obj_pos, q_f2_f1=next_obj_qua) * fin_sg[7]
        next_fin_sgs.append(np.concatenate((next_fl_sg, next_fr_sg, [fin_sg[6],], [fin_sg[7],])))

        # 计算reward
        #* 未来n步内，有一步到达goal，就设r=max
        for n in range(1, Tr+1):
            # 目标位置
            _next_obj_pos = raw_obs['object_pos'][step+n]
            _next_obj_qua = raw_obs['object_quat'][step+n]
            _next_fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=_next_obj_pos, q_f2_f1=_next_obj_qua) * fin_sg[6]
            _next_fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=_next_obj_pos, q_f2_f1=_next_obj_qua) * fin_sg[7]
            # 手指位置(世界坐标系下)
            _next_fl_pos = raw_obs['fingers_position'][step+n, :3]
            _next_fr_pos = raw_obs['fingers_position'][step+n, 3:]
            # 手指距离 - 欧式距离
            next_fl_dp = np.linalg.norm(_next_fl_pos - _next_fl_sg) * fin_sg[6]
            next_fr_dp = np.linalg.norm(_next_fr_pos - _next_fr_sg) * fin_sg[7]
            if max(next_fl_dp, next_fr_dp) < goal_thresh:
                r = max_reward
                break
            else:
                if reward_mode == 'only_success':
                    r = 0
                elif reward_mode == 'tanh':
                    reward_weights = 3
                    r_fl = -np.tanh(next_fl_dp * reward_weights)
                    r_fr = -np.tanh(next_fr_dp * reward_weights)
                    r = (r_fl+r_fr) / 3 * 2 + 1
                else:
                    raise ValueError('reward_mode must be `only_success` or `tanh`')
        
        reward.append(r)

    return {
        'subgoal': np.array(fin_sgs),
        'next_subgoal': np.array(next_fin_sgs),
        'reward': np.array(reward)
    }


def get_subgoals_stage_robomimic(
        raw_obs: dict,
        object_pcd: np.ndarray,
        fin_rad: float,
        sim_thresh: list,
        max_reward=10,
        reward_mode='tanh',
        Tr=1
        ):
    """
    计算手指位置子目标, 不考虑平滑

    args:
        - raw_obs: h5py dict {object_pos, object_quat, eef_pos, eef_quat, fingers_position}
        - object_pcd: object pointcloud, np.ndarray, shape=(N, 3)
        - fin_rad: 手指半径
        - sim_thresh: list(平移误差, 弧度误差) 计算物体位姿子目标是否达到的阈值，各任务单独设置
        - max_reward
        - reward_mode (str): 'only_success' or 'tanh'

    return:
        - subgoal: (N-1, 8) 对应每个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - next_subgoal: (N-1, 8) 对应下一个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - reward: (N-1,)
    """
    

    """
    (1) 初选：当手指与物体接触时记录手指位置, 当接触状态变化时记录物体位姿
    (2) 过滤：当记录的两次相邻物体位姿相似时(阈值很小)，删除前一个物体位姿，并将对应的手指位置设为全空
    (3) 配置：遍历状态，当物体到达记录的位姿时，设置对应的手指位置为子目标(对应指物体位姿索引或相同的索引)，
            如果手指位置为全空，则设置后面最近的手指位置为子目标
    """

    # ********** 初选子目标 **********
    is_last_fl_contact = False
    is_last_fr_contact = False
    obj_subgoals = list()   # 物体位姿子目标
    obj_subgoals_id = list()   # 物体位姿子目标索引
    fin_subgoals_obj_init = list()  # 手指位置子目标(物体坐标系下)

    contact_thresh = fin_rad + 0.01
    
    # (1) 初选：当手指与物体接触时记录手指位置, 当接触状态变化时记录物体位姿
    sequence_length = raw_obs['object'].shape[0]
    for step in range(sequence_length):
        # fingers position
        fl_pos, fr_pos = getFingersPos(
            raw_obs['robot0_eef_pos'][step], 
            raw_obs['robot0_eef_quat'][step], 
            raw_obs['robot0_gripper_qpos'][step, 0]+0.0145/2,
            raw_obs['robot0_gripper_qpos'][step, 1]-0.0145/2
            )
        # 计算手指与物体是否接触, 如果手指与点云的最小距离小于阈值，认为接触
        obj_pos = raw_obs['object'][step, :3]
        obj_qua = raw_obs['object'][step, 3:7]
        # 将手指位置转到物体坐标系
        T_W_O = tf.PosQua_to_TransMat(obj_pos, obj_qua)
        T_O_W = np.linalg.inv(T_W_O)
        fl_pos_obj = tf.transPt(fl_pos, T_f2_f1=T_O_W)
        fr_pos_obj = tf.transPt(fr_pos, T_f2_f1=T_O_W)
        # 计算点云到手指的距离
        fl_dists = object_pcd - fl_pos_obj
        fr_dists = object_pcd - fr_pos_obj
        fl_dist = np.min(np.sqrt(np.sum(np.square(fl_dists), axis=1)))
        fr_dist = np.min(np.sqrt(np.sum(np.square(fr_dists), axis=1)))

        is_fl_contact = fl_dist < contact_thresh
        is_fr_contact = fr_dist < contact_thresh
        # 记录手指接触位置
        fin_subgoals_obj_init.append(
            np.concatenate((fl_pos_obj, fr_pos_obj, [is_fl_contact,], [is_fr_contact,])))
        # 记录物体位姿
        if (is_fl_contact != is_last_fl_contact) or (is_fr_contact != is_last_fr_contact):
            obj_subgoals.append(np.concatenate((obj_pos, obj_qua)))
            obj_subgoals_id.append(step)
        
        is_last_fl_contact = is_fl_contact
        is_last_fr_contact = is_fr_contact


    # ********** 子目标精简 **********
    # (2) 过滤：当记录的两次相邻物体位姿相似时，删除前一个物体位姿，并将对应的手指位置设为None
    i = 0
    while i < len(obj_subgoals)-1:
        obj_sim = check_poses_similarity(
            obj_subgoals[i], obj_subgoals[i+1], 
            pos_th=sim_thresh[0], euler_th=sim_thresh[1])   

        if obj_sim:
            # 直到下一个物体子目标之前的手指接触都设为0
            for s in np.arange(obj_subgoals_id[i], obj_subgoals_id[i+1]):
                fin_subgoals_obj_init[s] = np.zeros((8,))
            # 删除前一个物体子目标
            obj_subgoals.pop(i)
            obj_subgoals_id.pop(i)
        else:
            # 继续对比下一个
            i+=1
    # 子目标前移一位
    fin_subgoals_obj_init.pop(0)
    fin_subgoals_obj_init.append(np.zeros((8,)))


    # ********** 子目标配置 **********
    # (3) 配置：遍历状态，当手指到达子目标，且物体到达记录的位姿，时，设置对应的手指位置为子目标(对应指物体位姿索引或相同的索引)，
    #        如果手指位置为全空，则设置后面最近的手指位置为子目标
    fin_sgs = list()
    next_fin_sgs = list()
    reward = list()
    goal_thresh = fin_rad/2
    obj_sg_id = 0   # 已达到的物体位姿子目标
    last_done = 'obj'   # obj/fin
    r = 0
    if len(obj_subgoals_id) == 0:
        # 没有物体子目标，返回全零数据
        zero_subgoal = np.zeros((sequence_length-Tr, 8))
        zero_reward = np.zeros((sequence_length-Tr,))
        return {
            'subgoal': zero_subgoal,
            'next_subgoal': zero_subgoal,
            'reward': zero_reward
        }
    for step in range(sequence_length-Tr):
        # 手指位置
        fl_pos, fr_pos = getFingersPos(
            raw_obs['robot0_eef_pos'][step], 
            raw_obs['robot0_eef_quat'][step], 
            raw_obs['robot0_gripper_qpos'][step, 0]+0.0145/2,
            raw_obs['robot0_gripper_qpos'][step, 1]-0.0145/2
            )
        # 物体位姿
        obj_pos = raw_obs['object'][step, :3]
        obj_qua = raw_obs['object'][step, 3:7]
        obj_pose = np.concatenate((obj_pos, obj_qua))
        if last_done == 'obj':
            # 检测手指是否到达子目标
            if r == max_reward: #!
                last_done = 'fin'
        
        if last_done == 'fin' and obj_sg_id < len(obj_subgoals_id)-1:
            # 检测物体是否到达子目标
            obj_sim = check_poses_similarity(
                obj_pose, obj_subgoals[obj_sg_id+1], 
                pos_th=sim_thresh[0], euler_th=sim_thresh[1])
            if obj_sim: 
                obj_sg_id += 1
                last_done = 'obj'

        # 设置子目标
        fin_sg_id = max(obj_subgoals_id[obj_sg_id], step)
        fin_sg = fin_subgoals_obj_init[fin_sg_id]

        # 记录世界坐标下的当前时刻的子目标
        fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_sg[6]
        fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_sg[7]
        fin_sgs.append(np.concatenate((fl_sg, fr_sg, [fin_sg[6],], [fin_sg[7],])))        
        # 记录下一时刻的子目标
        next_obj_pos = raw_obs['object'][step+Tr, :3]
        next_obj_qua = raw_obs['object'][step+Tr, 3:7]
        next_fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=next_obj_pos, q_f2_f1=next_obj_qua) * fin_sg[6]
        next_fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=next_obj_pos, q_f2_f1=next_obj_qua) * fin_sg[7]
        next_fin_sgs.append(np.concatenate((next_fl_sg, next_fr_sg, [fin_sg[6],], [fin_sg[7],])))

        # 计算reward
        #* 未来n步内，有一步到达goal，就设r=max
        for n in range(1, Tr+1):
            # 目标位置
            _next_obj_pos = raw_obs['object'][step+n, :3]
            _next_obj_qua = raw_obs['object'][step+n, 3:7]
            _next_fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=_next_obj_pos, q_f2_f1=_next_obj_qua) * fin_sg[6]
            _next_fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=_next_obj_pos, q_f2_f1=_next_obj_qua) * fin_sg[7]
            # 手指位置(世界坐标系下)
            _next_fl_pos, _next_fr_pos = getFingersPos(
                raw_obs['robot0_eef_pos'][step+n], 
                raw_obs['robot0_eef_quat'][step+n], 
                raw_obs['robot0_gripper_qpos'][step+n, 0]+0.0145/2,
                raw_obs['robot0_gripper_qpos'][step+n, 1]-0.0145/2
                )
            
            # 手指距离 - 欧式距离
            _next_fl_dp = np.linalg.norm(_next_fl_pos - _next_fl_sg) * fin_sg[6]
            _next_fr_dp = np.linalg.norm(_next_fr_pos - _next_fr_sg) * fin_sg[7]
            if max(_next_fl_dp, _next_fr_dp) < goal_thresh:
                r = max_reward
                break
            else:
                if reward_mode == 'only_success':
                    r = 0
                elif reward_mode == 'tanh':
                    reward_weights = 3
                    r_fl = -np.tanh(_next_fl_dp * reward_weights)
                    r_fr = -np.tanh(_next_fr_dp * reward_weights)
                    r = (r_fl+r_fr) / 3 * 2 + 1
                else:
                    raise ValueError('reward_mode must be `only_success` or `tanh`')
        
        reward.append(r)

    return {
        'subgoal': np.array(fin_sgs),
        'next_subgoal': np.array(next_fin_sgs),
        'reward': np.array(reward)
    }


def get_subgoals_stage_robomimic_v61(
        raw_obs: dict,
        object_pcd: np.ndarray,
        fin_rad: float,
        sim_thresh: list,
        max_reward=10,
        reward_mode='tanh',
        horizon=16
        ):
    """
    计算手指位置子目标, 不考虑平滑

    计算state之后horizon个状态的next_subgoal和reward

    args:
        - raw_obs: h5py dict {object_pos, object_quat, eef_pos, eef_quat, fingers_position}
        - object_pcd: object pointcloud, np.ndarray, shape=(N, 3)
        - fin_rad: 手指半径
        - sim_thresh: list(平移误差, 弧度误差) 计算物体位姿子目标是否达到的阈值，各任务单独设置
        - max_reward
        - reward_mode (str): 'only_success' or 'tanh'

    return:
        - subgoal: (N, 8) 对应每个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - next_subgoal: (N, horizon, 8) 对应下一个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - reward: (N, horizon)
    """

    """
    (1) 初选：当手指与物体接触时记录手指位置, 当接触状态变化时记录物体位姿
    (2) 过滤：当记录的两次相邻物体位姿相似时(阈值很小)，删除前一个物体位姿，并将对应的手指位置设为全空
    (3) 配置：遍历状态，当物体到达记录的位姿时，设置对应的手指位置为子目标(对应指物体位姿索引或相同的索引)，
            如果手指位置为全空，则设置后面最近的手指位置为子目标
    """

    # ********** 初选子目标 **********
    is_last_fl_contact = False
    is_last_fr_contact = False
    obj_subgoals = list()   # 物体位姿子目标
    obj_subgoals_id = list()   # 物体位姿子目标索引
    fin_subgoals_obj_init = list()  # 手指位置子目标(物体坐标系下)

    contact_thresh = fin_rad + 0.01
    
    # (1) 初选：当手指与物体接触时记录手指位置, 当接触状态变化时记录物体位姿
    sequence_length = raw_obs['object'].shape[0]
    for step in range(sequence_length):
        # fingers position
        fl_pos, fr_pos = getFingersPos(
            raw_obs['robot0_eef_pos'][step], 
            raw_obs['robot0_eef_quat'][step], 
            raw_obs['robot0_gripper_qpos'][step, 0]+0.0145/2,
            raw_obs['robot0_gripper_qpos'][step, 1]-0.0145/2
            )
        # 计算手指与物体是否接触, 如果手指与点云的最小距离小于阈值，认为接触
        obj_pos = raw_obs['object'][step, :3]
        obj_qua = raw_obs['object'][step, 3:7]
        # 将手指位置转到物体坐标系
        T_W_O = tf.PosQua_to_TransMat(obj_pos, obj_qua)
        T_O_W = np.linalg.inv(T_W_O)
        fl_pos_obj = tf.transPt(fl_pos, T_f2_f1=T_O_W)
        fr_pos_obj = tf.transPt(fr_pos, T_f2_f1=T_O_W)
        # 计算点云到手指的距离
        fl_dists = object_pcd - fl_pos_obj
        fr_dists = object_pcd - fr_pos_obj
        fl_dist = np.min(np.sqrt(np.sum(np.square(fl_dists), axis=1)))
        fr_dist = np.min(np.sqrt(np.sum(np.square(fr_dists), axis=1)))

        is_fl_contact = fl_dist < contact_thresh
        is_fr_contact = fr_dist < contact_thresh
        # 记录手指接触位置
        fin_subgoals_obj_init.append(
            np.concatenate((fl_pos_obj, fr_pos_obj, [is_fl_contact,], [is_fr_contact,])))
        # 记录物体位姿
        if (is_fl_contact != is_last_fl_contact) or (is_fr_contact != is_last_fr_contact):
            obj_subgoals.append(np.concatenate((obj_pos, obj_qua)))
            obj_subgoals_id.append(step)
        
        is_last_fl_contact = is_fl_contact
        is_last_fr_contact = is_fr_contact


    # ********** 子目标精简 **********
    # (2) 过滤：当记录的两次相邻物体位姿相似时，删除前一个物体位姿，并将对应的手指位置设为None
    i = 0
    while i < len(obj_subgoals)-1:
        obj_sim = check_poses_similarity(
            obj_subgoals[i], obj_subgoals[i+1], 
            pos_th=sim_thresh[0], euler_th=sim_thresh[1])   

        if obj_sim:
            # 直到下一个物体子目标之前的手指接触都设为0
            for s in np.arange(obj_subgoals_id[i], obj_subgoals_id[i+1]):
                fin_subgoals_obj_init[s] = np.zeros((8,))
            # 删除前一个物体子目标
            obj_subgoals.pop(i)
            obj_subgoals_id.pop(i)
        else:
            # 继续对比下一个
            i+=1
    # 子目标前移一位
    fin_subgoals_obj_init.pop(0)
    fin_subgoals_obj_init.append(np.zeros((8,)))


    # ********** 子目标配置 **********
    # (3) 配置：遍历状态，当手指到达子目标，且物体到达记录的位姿，时，设置对应的手指位置为子目标(对应指物体位姿索引或相同的索引)，
    #        如果手指位置为全空，则设置后面最近的手指位置为子目标
    fin_sgs = list()
    next_fin_sgss = list()
    rewards = list()
    goal_thresh = fin_rad     #!!! 增加 原来是fin_rad/2
    # goal_thresh = 0.03     #!!! 增加 原来是fin_rad/2
    obj_sg_id = 0   # 已达到的物体位姿子目标
    last_done = 'obj'   # obj/fin
    r = 0
    for step in range(sequence_length):
        # 手指位置
        fl_pos, fr_pos = getFingersPos(
            raw_obs['robot0_eef_pos'][step], 
            raw_obs['robot0_eef_quat'][step], 
            raw_obs['robot0_gripper_qpos'][step, 0]+0.0145/2,
            raw_obs['robot0_gripper_qpos'][step, 1]-0.0145/2
            )
        # 物体位姿
        obj_pos = raw_obs['object'][step, :3]
        obj_qua = raw_obs['object'][step, 3:7]
        obj_pose = np.concatenate((obj_pos, obj_qua))
        if last_done == 'obj':
            # 检测手指是否到达子目标
            if r == max_reward: #!
                last_done = 'fin'
        
        if last_done == 'fin' and obj_sg_id < len(obj_subgoals_id)-1:
            # 检测物体是否到达子目标
            obj_sim = check_poses_similarity(
                obj_pose, obj_subgoals[obj_sg_id+1], 
                pos_th=sim_thresh[0], euler_th=sim_thresh[1])
            if obj_sim: 
                obj_sg_id += 1
                last_done = 'obj'

        # 设置子目标
        fin_sg_id = max(obj_subgoals_id[obj_sg_id], step)
        fin_sg = fin_subgoals_obj_init[fin_sg_id]

        # 记录世界坐标下的当前时刻的子目标
        fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_sg[6]
        fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_sg[7]
        fin_sgs.append(np.concatenate((fl_sg, fr_sg, [fin_sg[6],], [fin_sg[7],]))) 

        # 记录下一时刻的子目标
        next_fin_sgs = list()
        reward = list()
        for h in range(horizon+1)[1:]:
            next_step = step+h
            if next_step >= sequence_length:
                next_fin_sgs.append(fin_sgs[-1])
            else:
                next_obj_pos = raw_obs['object'][next_step, :3]
                next_obj_qua = raw_obs['object'][next_step, 3:7]
                next_fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=next_obj_pos, q_f2_f1=next_obj_qua) * fin_sg[6]
                next_fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=next_obj_pos, q_f2_f1=next_obj_qua) * fin_sg[7]
                next_fin_sgs.append(np.concatenate((next_fl_sg, next_fr_sg, [fin_sg[6],], [fin_sg[7],])))

            # 计算reward
            # 手指位置(世界坐标系下)
            if next_step >= sequence_length:
                r = max_reward
            else:
                next_fl_pos, next_fr_pos = getFingersPos(
                    raw_obs['robot0_eef_pos'][next_step], 
                    raw_obs['robot0_eef_quat'][next_step], 
                    raw_obs['robot0_gripper_qpos'][next_step, 0]+0.0145/2,
                    raw_obs['robot0_gripper_qpos'][next_step, 1]-0.0145/2)
                
                # 手指距离 - 欧式距离
                next_fl_dp = np.linalg.norm(next_fl_pos - next_fin_sgs[-1][:3]) * next_fin_sgs[-1][6]
                next_fr_dp = np.linalg.norm(next_fr_pos - next_fin_sgs[-1][3:6]) * next_fin_sgs[-1][7]
                if max(next_fl_dp, next_fr_dp) < goal_thresh:
                    r = max_reward
                else:
                    if reward_mode == 'only_success':
                        r = 0
                    elif reward_mode == 'tanh':
                        reward_weights = 3
                        r_fl = -np.tanh(next_fl_dp * reward_weights)
                        r_fr = -np.tanh(next_fr_dp * reward_weights)
                        r = (r_fl+r_fr) / 3 * 2 + 1
                    else:
                        raise ValueError('reward_mode must be `only_success` or `tanh`')
            
            reward.append(r)
        
        next_fin_sgss.append(np.array(next_fin_sgs))
        rewards.append(np.array(reward))

    return {
        'subgoal': np.array(fin_sgs),
        'next_subgoal': np.array(next_fin_sgss),
        'reward': np.array(rewards)
    }


def get_subgoals_realtime_robomimic(
        raw_obs: dict,
        object_pcd: np.ndarray,
        fin_rad: float,
        max_reward=10,
        reward_mode='tanh',
        Tr=1
        ):
    """
    计算手指位置子目标, 不考虑平滑

    args:
        - raw_obs: h5py dict {object_pos, object_quat, eef_pos, eef_quat, fingers_position}
        - object_pcd: object pointcloud, np.ndarray, shape=(N, 3)
        - fin_rad: 手指半径
        - max_reward
        - reward_mode (str): 'only_success' or 'tanh'

    return:
        - subgoal: (N-1, 8) 对应每个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - next_subgoal: (N-1, 8) 对应下一个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - reward: (N-1,)
    """


    # ********** 初选子目标 **********
    fin_subgoals_obj = list()  # 手指位置子目标(物体坐标系下)
    contact_thresh = fin_rad + 0.01
    sequence_length = raw_obs['object'].shape[0]
    for step in range(sequence_length):
        # fingers position
        fl_pos, fr_pos = getFingersPos(
            raw_obs['robot0_eef_pos'][step], 
            raw_obs['robot0_eef_quat'][step], 
            raw_obs['robot0_gripper_qpos'][step, 0]+0.0145/2,
            raw_obs['robot0_gripper_qpos'][step, 1]-0.0145/2
            )
        # 计算手指与物体是否接触, 如果手指与点云的最小距离小于阈值，认为接触
        obj_pos = raw_obs['object'][step, :3]
        obj_qua = raw_obs['object'][step, 3:7]
        # 将手指位置转到物体坐标系
        T_W_O = tf.PosQua_to_TransMat(obj_pos, obj_qua)
        T_O_W = np.linalg.inv(T_W_O)
        fl_pos_obj = tf.transPt(fl_pos, T_f2_f1=T_O_W)
        fr_pos_obj = tf.transPt(fr_pos, T_f2_f1=T_O_W)
        # 计算点云到手指的距离
        fl_dists = object_pcd - fl_pos_obj
        fr_dists = object_pcd - fr_pos_obj
        fl_dist = np.min(np.sqrt(np.sum(np.square(fl_dists), axis=1)))
        fr_dist = np.min(np.sqrt(np.sum(np.square(fr_dists), axis=1)))

        is_fl_contact = fl_dist < contact_thresh
        is_fr_contact = fr_dist < contact_thresh
        fin_subgoals_obj.append(
            np.concatenate((fl_pos_obj, fr_pos_obj, [is_fl_contact,], [is_fr_contact,])))

    fin_subgoals_obj.pop(0) # 子目标前移一位


    # ********** 子目标配置 **********
    # (3) 配置：遍历状态，当手指到达子目标，且物体到达记录的位姿，时，设置对应的手指位置为子目标(对应指物体位姿索引或相同的索引)，
    #        如果手指位置为全空，则设置后面最近的手指位置为子目标
    fin_sgs = list()
    next_fin_sgs = list()
    reward = list()
    goal_thresh = fin_rad/2
    for step in range(sequence_length-Tr):
        # 物体位姿
        obj_pos = raw_obs['object'][step, :3]
        obj_qua = raw_obs['object'][step, 3:7]
        obj_pose = np.concatenate((obj_pos, obj_qua))
        # 设置子目标
        fin_sg = fin_subgoals_obj[step]

        # 记录世界坐标下的当前时刻的子目标
        fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_sg[6]
        fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_sg[7]
        fin_sgs.append(np.concatenate((fl_sg, fr_sg, [fin_sg[6],], [fin_sg[7],])))        
        # 记录下一时刻的子目标
        next_obj_pos = raw_obs['object'][step+Tr, :3]
        next_obj_qua = raw_obs['object'][step+Tr, 3:7]
        next_fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=next_obj_pos, q_f2_f1=next_obj_qua) * fin_sg[6]
        next_fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=next_obj_pos, q_f2_f1=next_obj_qua) * fin_sg[7]
        next_fin_sgs.append(np.concatenate((next_fl_sg, next_fr_sg, [fin_sg[6],], [fin_sg[7],])))

        # 计算reward
        #* 未来n步内，有一步到达goal，就设r=max
        for n in range(1, Tr+1):
            # 目标位置
            _next_obj_pos = raw_obs['object'][step+n, :3]
            _next_obj_qua = raw_obs['object'][step+n, 3:7]
            _next_fl_sg = tf.transPt(fin_sg[:3], t_f2_f1=_next_obj_pos, q_f2_f1=_next_obj_qua) * fin_sg[6]
            _next_fr_sg = tf.transPt(fin_sg[3:6], t_f2_f1=_next_obj_pos, q_f2_f1=_next_obj_qua) * fin_sg[7]
            # 手指位置(世界坐标系下)
            next_fl_pos, next_fr_pos = getFingersPos(
                raw_obs['robot0_eef_pos'][step+n], 
                raw_obs['robot0_eef_quat'][step+n], 
                raw_obs['robot0_gripper_qpos'][step+1, 0]+0.0145/2,
                raw_obs['robot0_gripper_qpos'][step+1, 1]-0.0145/2
                )
            
            # 手指距离 - 欧式距离
            next_fl_dp = np.linalg.norm(next_fl_pos - _next_fl_sg) * fin_sg[6]
            next_fr_dp = np.linalg.norm(next_fr_pos - _next_fr_sg) * fin_sg[7]
            if max(next_fl_dp, next_fr_dp) < goal_thresh:
                r = max_reward
                break
            else:
                if reward_mode == 'only_success':
                    r = 0
                elif reward_mode == 'tanh':
                    reward_weights = 3
                    r_fl = -np.tanh(next_fl_dp * reward_weights)
                    r_fr = -np.tanh(next_fr_dp * reward_weights)
                    r = (r_fl+r_fr) / 3 * 2 + 1
                else:
                    raise ValueError('reward_mode must be `only_success` or `tanh`')
        
        reward.append(r)

    return {
        'subgoal': np.array(fin_sgs),
        'next_subgoal': np.array(next_fin_sgs),
        'reward': np.array(reward)
    }

def get_subgoals_pusht(
        raw_obs: np.ndarray,
        episode_ends: np.ndarray,
        fin_rad: float,
        sim_thresh: list,
        max_reward=10,
        Tr=1,
        reward_mode='tanh'
        ):
    """
    计算手指位置子目标, 不考虑平滑
    输入包含多个轨迹, 根据物体位姿突变划分轨迹

    args:
        - raw_obs: (N, 5) 手指位置2/物体位置2/旋转角1 （旋转角为0时，T的竖线朝上，物体逆时针旋转时，角度增加）
        - object_pcd: object pointcloud, np.ndarray, shape=(N, 3)
        - fin_rad: 手指半径
        - sim_thresh: list(平移误差, 弧度误差) 计算物体位姿子目标是否达到的阈值，各任务单独设置

    return:
        - subgoal: (N-1, 8) 对应每个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - next_subgoal: (N-1, 8) 对应下一个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - reward: (N-1,)
    """
    # 构建物体点云
    object_pcd = create_pusht_pts(pts_num=1024*5) # (n, 2)

    fin_sgs = list()
    next_fin_sgs = list()
    reward = list()
    episode_ends = np.insert(episode_ends, 0, 0)
    for j in range(episode_ends.shape[0])[1:]:
        start = episode_ends[j-1]
        end = episode_ends[j]

        # ********** 初选子目标 **********
        is_last_fin_contact = False
        obj_subgoals = list()   # 物体位姿子目标
        obj_subgoals_id = list()   # 物体位姿子目标索引
        fin_subgoals_obj_init = list()  # 手指位置子目标(物体坐标系下)
        contact_thresh = fin_rad + 3
        
        # (1) 初选：当手指与物体接触时记录手指位置, 当接触状态变化时记录物体位姿
        for step in np.arange(start, end):
            # fingers position
            fin_pos = raw_obs[step, :2]
            obj_pos = raw_obs[step, 2:4]
            obj_rad = raw_obs[step, 4]
            # 将手指位置转到物体坐标系
            T_W_O = tf.PosRad_to_Tmat(obj_pos, obj_rad)
            T_O_W = np.linalg.inv(T_W_O)
            fin_pos_obj = tf.transPt2D(fin_pos, T_f2_f1=T_O_W)
            # 计算点云到手指的距离
            fin_dists = object_pcd - fin_pos_obj
            fin_dist = np.min(np.sqrt(np.sum(np.square(fin_dists), axis=1)))
            is_fin_contact = fin_dist < contact_thresh
            # 记录手指接触位置
            fin_subgoals_obj_init.append(np.append(fin_pos_obj, float(is_fin_contact)))
            # 记录物体位姿
            if is_fin_contact != is_last_fin_contact:
                obj_subgoals.append(raw_obs[step, 2:])
                obj_subgoals_id.append(step-start)
            
            is_last_fin_contact = is_fin_contact

        # ********** 子目标精简 **********
        # (2) 过滤：当记录的两次相邻物体位姿相似时，删除前一个物体位姿，并将对应的手指位置设为None
        i = 0
        while i < len(obj_subgoals)-1:
            obj_sim = check_poses_similarity_2d(
                obj_subgoals[i], obj_subgoals[i+1], 
                pos_th=sim_thresh[0], euler_th=sim_thresh[1])   

            if obj_sim:
                # 直到下一个物体子目标之前的手指接触都设为0
                for s in np.arange(obj_subgoals_id[i], obj_subgoals_id[i+1]):
                    fin_subgoals_obj_init[s] = np.zeros((3,))
                # 删除前一个物体子目标
                obj_subgoals.pop(i)
                obj_subgoals_id.pop(i)
            else:
                # 继续对比下一个
                i+=1
        # 子目标前移一位
        fin_subgoals_obj_init.pop(0)
        fin_subgoals_obj_init.append(np.zeros((3,)))

        # ********** 子目标配置 **********
        # (3) 配置：遍历状态，当手指到达子目标，且物体到达记录的位姿，时，设置对应的手指位置为子目标(对应指物体位姿索引或相同的索引)，
        #        如果手指位置为全空，则设置后面最近的手指位置为子目标
        goal_thresh = fin_rad/2
        obj_sg_id = 0   # 已达到的物体位姿子目标
        last_done = 'obj'   # obj/fin
        r = 0
        for step in np.arange(start, end):
            if step >= end-Tr:
                fin_sgs.append(np.zeros((3,)))
                next_fin_sgs.append(np.zeros((3,)))
                reward.append(max_reward)
                if step == end-1: break
                else: continue

            # 物体位姿
            obj_pose = raw_obs[step, 2:]
            if last_done == 'obj':
                # 检测手指是否到达子目标
                if r == max_reward:
                    last_done = 'fin'
            
            if last_done == 'fin' and obj_sg_id < len(obj_subgoals_id)-1:
                # 检测物体是否到达子目标
                obj_sim = check_poses_similarity_2d(
                    obj_pose, obj_subgoals[obj_sg_id+1], 
                    pos_th=sim_thresh[0], euler_th=sim_thresh[1])
                if obj_sim: 
                    obj_sg_id += 1
                    last_done = 'obj'

            # 设置子目标
            fin_sg_id = max(obj_subgoals_id[obj_sg_id], step-start)
            fin_sg = fin_subgoals_obj_init[fin_sg_id]

            # 记录世界坐标下的当前时刻的子目标
            T_W_O = tf.PosRad_to_Tmat(raw_obs[step, 2:4], raw_obs[step, 4])
            fin_sg_pos = tf.transPt2D(fin_sg[:2], T_W_O) * fin_sg[2]
            fin_sgs.append(np.append(fin_sg_pos, fin_sg[2]))
            # 记录下一时刻的子目标
            T_W_On = tf.PosRad_to_Tmat(raw_obs[step+Tr, 2:4], raw_obs[step+Tr, 4])
            next_fin_sg_pos = tf.transPt2D(fin_sg[:2], T_W_On) * fin_sg[2]
            next_fin_sgs.append(np.append(next_fin_sg_pos, fin_sg[2]))

            # 计算reward
            #* 未来n步内，有一步到达goal，就设r=max
            for n in range(1, Tr+1):
                # 手指位置(世界坐标系下)
                _next_fin_pos = raw_obs[step+n, :2]
                # 目标位置
                _T_W_On = tf.PosRad_to_Tmat(raw_obs[step+n, 2:4], raw_obs[step+n, 4])
                _next_fin_sg_pos = tf.transPt2D(fin_sg[:2], _T_W_On) * fin_sg[2]
                # 手指距离 - 欧式距离
                _next_fin_dp = np.linalg.norm(_next_fin_pos - _next_fin_sg_pos) * fin_sg[2]
                if _next_fin_dp < goal_thresh:
                    r = max_reward
                    break
                else:
                    if reward_mode == 'only_success':
                        r = 0
                    elif reward_mode == 'tanh':
                        reward_weights = 3
                        r_fl = -np.tanh(_next_fin_dp/15*0.008 * reward_weights)
                        r = r_fl / 3 * 2 + 1
                    else:
                        raise ValueError('reward_mode must be `only_success` or `tanh`')
                
            reward.append(r)

    return {
        'subgoal': np.array(fin_sgs),
        'next_subgoal': np.array(next_fin_sgs),
        'reward': np.array(reward)
    }


def get_subgoals_realtime_pusht(
        raw_obs: np.ndarray,
        episode_ends: np.ndarray,
        fin_rad: float,
        max_reward=10,
        Tr=1,
        reward_mode='tanh'
        ):
    """
    计算手指位置子目标, 不考虑平滑
    输入包含多个轨迹, 根据物体位姿突变划分轨迹

    args:
        - raw_obs: (N, 5) 手指位置2/物体位置2/旋转角1 （旋转角为0时，T的竖线朝上，物体逆时针旋转时，角度增加）
        - object_pcd: object pointcloud, np.ndarray, shape=(N, 3)
        - fin_rad: 手指半径
        - sim_thresh: list(平移误差, 弧度误差) 计算物体位姿子目标是否达到的阈值，各任务单独设置

    return:
        - subgoal: (N-1, 8) 对应每个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - next_subgoal: (N-1, 8) 对应下一个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - reward: (N-1,)
    """
    # 构建物体点云
    object_pcd = create_pusht_pts(pts_num=1024*5) # (n, 2)

    fin_sgs = list()
    next_fin_sgs = list()
    reward = list()
    episode_ends = np.insert(episode_ends, 0, 0)
    for j in range(episode_ends.shape[0])[1:]:
        start = episode_ends[j-1]
        end = episode_ends[j]

        # ********** 初选子目标 **********
        fin_subgoals_obj_init = list()  # 手指位置子目标(物体坐标系下)
        contact_thresh = fin_rad + 3
        
        # (1) 初选：当手指与物体接触时记录手指位置
        for step in np.arange(start, end):
            # fingers position
            fin_pos = raw_obs[step, :2]
            obj_pos = raw_obs[step, 2:4]
            obj_rad = raw_obs[step, 4]
            # 将手指位置转到物体坐标系
            T_W_O = tf.PosRad_to_Tmat(obj_pos, obj_rad)
            T_O_W = np.linalg.inv(T_W_O)
            fin_pos_obj = tf.transPt2D(fin_pos, T_f2_f1=T_O_W)
            # 计算点云到手指的距离
            fin_dists = object_pcd - fin_pos_obj
            fin_dist = np.min(np.sqrt(np.sum(np.square(fin_dists), axis=1)))
            is_fin_contact = fin_dist < contact_thresh
            # print('is_fin_contact =', is_fin_contact)
            # 记录手指接触位置
            fin_subgoals_obj_init.append(np.append(fin_pos_obj, float(is_fin_contact)))

        # 子目标前移一位
        fin_subgoals_obj_init.pop(0)

        # ********** 子目标配置 **********
        goal_thresh = fin_rad/2
        r = 0
        for step in np.arange(start, end):
            if step >= end-Tr:
                fin_sgs.append(np.zeros((3,)))
                next_fin_sgs.append(np.zeros((3,)))
                reward.append(max_reward)
                if step == end-1: break
                else: continue

            # 设置子目标
            fin_sg = fin_subgoals_obj_init[step - start]

            # 记录世界坐标下的当前时刻的子目标
            T_W_O = tf.PosRad_to_Tmat(raw_obs[step, 2:4], raw_obs[step, 4])
            fin_sg_pos = tf.transPt2D(fin_sg[:2], T_W_O) * fin_sg[2]
            fin_sgs.append(np.append(fin_sg_pos, fin_sg[2]))
            # 记录下一时刻的子目标
            T_W_On = tf.PosRad_to_Tmat(raw_obs[step+Tr, 2:4], raw_obs[step+Tr, 4])
            next_fin_sg_pos = tf.transPt2D(fin_sg[:2], T_W_On) * fin_sg[2]
            next_fin_sgs.append(np.append(next_fin_sg_pos, fin_sg[2]))

            # 计算reward
            #* 未来n步内，有一步到达goal，就设r=max
            for n in range(1, Tr+1):
                # 手指位置(世界坐标系下)
                _next_fin_pos = raw_obs[step+n, :2]
                # 目标位置
                _T_W_On = tf.PosRad_to_Tmat(raw_obs[step+n, 2:4], raw_obs[step+n, 4])
                _next_fin_sg_pos = tf.transPt2D(fin_sg[:2], _T_W_On) * fin_sg[2]
                # 手指距离 - 欧式距离
                _next_fin_dp = np.linalg.norm(_next_fin_pos - _next_fin_sg_pos) * fin_sg[2]
                if _next_fin_dp < goal_thresh:
                    r = max_reward
                    break
                else:
                    if reward_mode == 'only_success':
                        r = 0
                    elif reward_mode == 'tanh':
                        reward_weights = 3
                        r_fl = -np.tanh(_next_fin_dp/15*0.008 * reward_weights)
                        r = r_fl / 3 * 2 + 1
                    else:
                        raise ValueError('reward_mode must be `only_success` or `tanh`')
                
            reward.append(r)

    return {
        'subgoal': np.array(fin_sgs),
        'next_subgoal': np.array(next_fin_sgs),
        'reward': np.array(reward)
    }


def get_subgoals_stage_real(
        state: np.ndarray,
        fin_rad,
        contact_state: np.ndarray,
        max_reward=10,
        reward_mode='tanh',
        Tr=1
        ):
    """
    计算真实任务中的手指位置子目标, 不考虑平滑

    args:
        - state (np.ndarray): (N, 13) eef_pos, eef_qua, fl_pos, fr_pos
        - contact_state (np.ndarray): (N) 手指与物体的接触状态, 0-无接触, 1-有接触
        - max_reward
        - reward_mode (str): 'only_success' or 'tanh'
        - Tr: 选择下个状态的间隔

    return:
        - subgoal: (N-1, 8) 对应每个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - next_subgoal: (N-1, 8) 对应下一个state的子目标 手指位置(世界坐标系)/是否接触, 不接触的手指子目标位置全为0
        - reward: (N-1,)
    """
    # ********** set subgoal **********
    finger_sgs = list()  # 手指位置子目标    
    sequence_length = state.shape[0]
    for step in range(sequence_length):
        is_contact = contact_state[step]
        # fingers position
        fl_pos = state[step, 7:10]*is_contact
        fr_pos = state[step, 10:13]*is_contact
        # 记录手指接触位置
        finger_sgs.append(
            np.concatenate((fl_pos, fr_pos, [is_contact,], [is_contact,])))
    
    # 补全接触状态
    for step in range(sequence_length)[:-1][::-1]:
        if np.sum(finger_sgs[step][-2:]) == 0 and np.sum(finger_sgs[step+1][-2:]) != 0:
            finger_sgs[step] = finger_sgs[step+1]


    # ********** set next_subgoal and reward **********
    finger_sgs = np.array(finger_sgs)
    fin_sgs = finger_sgs[:-Tr]
    next_fin_sgs = finger_sgs[Tr:]
    reward = list()
    goal_thresh = fin_rad/2
    
    for step in range(sequence_length-Tr):
        # 未来Tr步内，有一步到达目标就设r=max
        for n in range(1, Tr+1):
            # 手指位置
            next_fl_pos = state[step+n, 7:10]
            next_fr_pos = state[step+n, 10:13]
            # 目标
            next_pos_sg = finger_sgs[step+n]
            # 手指距离 - 欧式距离
            next_fl_dp = np.linalg.norm(next_fl_pos - next_pos_sg[:3]) * next_pos_sg[6]
            next_fr_dp = np.linalg.norm(next_fr_pos - next_pos_sg[3:6]) * next_pos_sg[7]
            if max(next_fl_dp, next_fr_dp) < goal_thresh:
                r = max_reward
                break
            else:
                if reward_mode == 'only_success':
                    r = 0
                elif reward_mode == 'tanh':
                    reward_weights = 3
                    r_fl = -np.tanh(next_fl_dp * reward_weights)
                    r_fr = -np.tanh(next_fr_dp * reward_weights)
                    r = (r_fl+r_fr) / 3 * 2 + 1
                else:
                    raise ValueError('reward_mode must be `only_success` or `tanh`')
        
        reward.append(r)

    return {
        'subgoal': np.array(fin_sgs),
        'next_subgoal': np.array(next_fin_sgs),
        'reward': np.array(reward)
    }

def compute_reward_nextSubgoal_from_subgoal(
        subgoal: np.ndarray, 
        obj_pose: np.ndarray, 
        next_obj_pose: np.ndarray, 
        next_fin_pos: np.ndarray,
        fin_rad,
        max_reward=10,
        ) -> torch.Tensor:
    """使用新生成的subgoal计算reward
    将subgoal转到物体坐标系下，再转到下一时刻的世界坐标系下，计算subgoal与手指位置的差异

    args(torch.Tensor): 
        - subgoal: (B, 8) 当前时刻的子目标(world)
        - obj_pose: (B, 7) 当前时刻的物体位姿
        - next_obj_pose: (B, 7) 下一时刻的物体位姿
        - next_fin_pos: (B, 6) 下一时刻的手指位置
        - fin_rad: 手指半径
    
    return:
        - reward: (B,) done=1，其余为0
        - next_subgoal: (B, 8)
    """
    reward = list()
    next_subgoal = list()
    for i in range(subgoal.shape[0]):
        sg = subgoal[i]
        op = obj_pose[i]
        nop = next_obj_pose[i]
        nfp = next_fin_pos[i]
        # 子目标转到下一时刻的世界坐标系
        # (1) 转到物体坐标系: P_O_sg = T_O_W * P_W_sg
        T_W_O = tf.PosQua_to_TransMat(op[:3], op[3:])
        T_O_W = np.linalg.inv(T_W_O)
        P_O_sgl = tf.transPt(P_f1_pt=sg[:3], T_f2_f1=T_O_W)
        P_O_sgr = tf.transPt(P_f1_pt=sg[3:6], T_f2_f1=T_O_W)
        # (2) 转到下一时刻的世界坐标系: P_W_sg = T_W_O * P_O_sg
        T_W_O_ = tf.PosQua_to_TransMat(nop[:3], nop[3:])
        P_W_sgl = tf.transPt(P_f1_pt=P_O_sgl, T_f2_f1=T_W_O_) * sg[6]
        P_W_sgr = tf.transPt(P_f1_pt=P_O_sgr, T_f2_f1=T_W_O_) * sg[7]
        # 计算reward
        fl_dist = np.linalg.norm(nfp[:3] - P_W_sgl) * sg[6]
        fr_dist = np.linalg.norm(nfp[3:6] - P_W_sgr) * sg[7]
        if max(fl_dist, fl_dist) < fin_rad/2:
            r = max_reward
        else:
            reward_weights = 3
            r_fl = -np.tanh(fl_dist * reward_weights)
            r_fr = -np.tanh(fr_dist * reward_weights)
            r = (r_fl+r_fr) / 3 * 2 + 1

        reward.append(r)
        next_subgoal.append(np.concatenate((P_W_sgl, P_W_sgr, sg[6:])))

    return torch.tensor(reward), torch.tensor(np.array(next_subgoal))


def distTwoPtWithCube(rect, pt1, pt2, z_th):
    """
    计算空间中两个点的路径距离, 两点可能被矩形分割

    计算流程:
        (1) 将矩形和两个点的x维度删除
        (2) 判断两点形成的线段是否可能rect分割为两个多边形
        (3) 如果分割结果为1个多边形(即无法分割), 则直接返回pt1和pt2的L2范数
        (4) 如果分割结果为2个多边形:
        (5) 计算不含 z<(z_th+0.01) 的多边形的边长,边长不包含分割线
        (6) 计算pt1和pt2到两个分割点的距离的较小值的和, 加上第5步多边形的边长, 和x距离计算L2范数, 返回

    args:
        rect (np.array): 空间中的矩形, shape=(4,3)
        pt1 (np.array): 三维点1, shape=(3,)
        pt2 (np.array): 三维点2, shape=(3,)
        z_th (float): z坐标阈值, 路径不能在z_th下面
    """
    # (1) 将矩形和两个点的x维度删除
    rect_yz = rect[:, 1:]
    pt1_yz = pt1[1:]
    pt2_yz = pt2[1:]
    # (2) 判断两点形成的线段是否可能rect分割为两个多边形
    line = shapely.geometry.LineString([list(pt1_yz), list(pt2_yz)])
    polygon = shapely.geometry.Polygon([list(rect_yz[0]), list(rect_yz[1]), list(rect_yz[2]), list(rect_yz[3])])
    polygons = shapely.ops.split(polygon, line)
    # (3) 如果分割结果为1个多边形(即无法分割), 则直接返回pt1和pt2的L2范数
    if len(polygons.geoms) == 1:
        return np.linalg.norm(pt1 - pt2)
    # (5) 计算不含 z<(z_th+0.01) 的多边形的边长,边长不包含分割线
    id = 0
    for z in polygons.geoms[id].exterior.coords.xy[1]:
        if z < (z_th+0.01):
            id = 1
    polygon_path = polygons.geoms[id]
    ys = polygon_path.exterior.coords.xy[0]
    zs = polygon_path.exterior.coords.xy[1]
    # 计算边长, 不含分割线
    l = 0
    for i in range(len(ys)-2):
        l += math.sqrt((ys[i] - ys[i+1])**2 + (zs[i] - zs[i+1])**2)
    # print('l =', l)
    # (6) 计算pt1和pt2到两个分割点的距离的较小值的和, 加上第5步多边形的边长, 和x距离计算L2范数, 返回
    seg_pt1 = np.array([ys[-1], zs[-1]])
    seg_pt2 = np.array([ys[-2], zs[-2]])
    l1 = min( np.linalg.norm(pt1_yz - seg_pt1), np.linalg.norm(pt1_yz - seg_pt2) )
    l2 = min( np.linalg.norm(pt2_yz - seg_pt1), np.linalg.norm(pt2_yz - seg_pt2) )
    l += l1 + l2 + abs(pt1[0] - pt2[0])
    # print('l1 =', l1)
    # print('l2 =', l2)
    # print('abs(pt1[0] - pt2[0]) =', abs(pt1[0] - pt2[0]))
    return l


def getCubeXPosWorld(cube_half_size, cube_pos, cube_quat):
    """
    获取物体x轴正方向上四个角点在世界坐标系下的位置
    """
    l1, l2, l3 = cube_half_size
    pts = np.array([
            [l1, -l2, l3],
            [l1, l2, l3],
            [l1, l2, -l3],
            [l1, -l2, -l3]
        ])
    # 转到世界坐标系下
    # P_o_p
    one = np.ones((1, pts.shape[0]))
    P_O_p = np.concatenate((pts.T, one), axis=0)   # (4,4)
    # T_w_o
    cube_rotMat = tf.quaternion_to_rotation_matrix(cube_quat)
    T_W_O = tf.PosRmat_to_TransMat(cube_pos, cube_rotMat)
    # P_w_p = T_w_o * P_o_p
    P_w_p = np.matmul(T_W_O, P_O_p)
    return P_w_p.T[:, :3]


def angle_diff(angle_1, angle_2):
    """
    输入弧度，输出差值 0-pi
    """
    angle_1 = angle_1 % (2*np.pi)
    angle_2 = angle_2 % (2*np.pi)
    
    angle_min = min(angle_1, angle_2)
    angle_max = max(angle_1, angle_2)

    error = angle_max - angle_min
    if error <= np.pi:
        return error
    else:
        return angle_min + 2*np.pi - angle_max


def check_poses_similarity_moveT(pose1, pose2, pos_th=0.005, euler_th=5./180.*np.pi):
    """
    计算两位姿的相似性(旋转只计算z轴旋转)
    pose: 平移+四元数xyzw
    return: 是否相似
    """
    # 当前时刻的平移和旋转差
    obj_dp = np.linalg.norm(pose1[:3] - pose2[:3])
    # 方案1：计算欧拉角的夹角，有时候会计算错误
    # obj_euler_1 = R.from_quat(pose1[3:]).as_euler('xyz', degrees=False)
    # obj_euler_2 = R.from_quat(pose2[3:]).as_euler('xyz', degrees=False)
    # obj_dr = ([tf.angle_diff(obj_euler_1[i], obj_euler_2[i]) for i in range(3)])
    # if obj_dp > pos_th or max(obj_dr) > euler_th:
    #     return False
    
    # 方案2：计算四元数的夹角
    rz1 = tf.Qua_to_Euler(pose1[3:])[2]
    rz2 = tf.Qua_to_Euler(pose2[3:])[2]
    obj_dr = angle_diff(rz1, rz2)
    if obj_dp > pos_th or obj_dr > euler_th:
        return False

    return True

def check_poses_similarity(pose1, pose2, pos_th=0.005, euler_th=5./180.*np.pi):
    """
    计算两位姿的相似性
    pose: 平移+四元数xyzw
    return: 是否相似
    """
    # 当前时刻的平移和旋转差
    obj_dp = np.linalg.norm(pose1[:3] - pose2[:3])
    # 方案1：计算欧拉角的夹角，有时候会计算错误
    # obj_euler_1 = R.from_quat(pose1[3:]).as_euler('xyz', degrees=False)
    # obj_euler_2 = R.from_quat(pose2[3:]).as_euler('xyz', degrees=False)
    # obj_dr = ([tf.angle_diff(obj_euler_1[i], obj_euler_2[i]) for i in range(3)])
    # if obj_dp > pos_th or max(obj_dr) > euler_th:
    #     return False
    
    # 方案2：计算四元数的夹角
    obj_dr = tf.qua_diff(pose1[3:], pose2[3:])
    if obj_dp > pos_th or obj_dr > euler_th:
        return False

    return True


def check_poses_similarity_2d(pose1, pose2, pos_th=0.005, euler_th=5./180.*np.pi):
    """
    计算两位姿的相似性
    pose: 坐标+旋转角xyr
    return: 是否相似
    """
    obj_dp = np.linalg.norm(pose1[:2] - pose2[:2])
    obj_dr = angle_diff(pose1[2], pose2[2])
    if obj_dp > pos_th or obj_dr > euler_th:
        return False
    return True


def check_pos_similarity(pos1, pos2, pos_th=0.005):
    """
    计算两位姿的相似性
    pose: 平移+四元数xyzw
    return: 是否相似
    """
    # 当前时刻的平移和旋转差
    obj_dp = np.linalg.norm(pos1 - pos2)
    if obj_dp > pos_th:
        return False
    return True


def get_scene_object_pcd_goal(dataset_path, visual=False, dtype=np.float32):
    """
    get scene_pcd / object_pcd / object_goal_pose(not use in HDP)
    dataset_path: robomimic数据集路径
    visual: 是否可视化图像/pcd
    return:
        - scene_pcd: (N, 3)
        - object_pcd: (N, 3)
        - object_goal_pose: (7,) pos+quat
    """

    import robosuite as suite
    from robosuite.controllers import load_controller_config
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_real_depth_map
    import time
    import Mix_diffusion_policy.common.transformation as tf

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

    options = {}

    options["env_name"] = env_meta['env_name']
    options["robots"] = env_meta['env_kwargs']["robots"]
    controller_name = "OSC_POSE"
    camera = "agentview"
    segmentation_level = 'instance'  # Options are {instance, class, element}

    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=20,
        camera_names=camera,
        camera_depths=True,
        camera_heights=512,
        camera_widths=512,
    )
    env.reset()
    
    # **************** object goal pose ****************
    object_goal_pose = env.object_goal_pose()

    # **************** image ****************
    env.remove_all_objects()    # 移除所有物体
    for i in range(50):
        action = [-1, 0, 0, 0, 0, 0, 0]
        obs, reward, done, _ = env.step(action)

    # segmentation
    img_seg = obs[f"{camera}_segmentation_{segmentation_level}"].squeeze(-1)[::-1]
    img_seg[np.where(img_seg > 0)] = 1
    # rgb
    img_rgb = obs[f"{camera}_image"][::-1]
    # depth
    img_dep = obs[f"{camera}_depth"].squeeze(-1)[::-1]
    img_dep = get_real_depth_map(env.sim, img_dep)
    if visual:
        # show
        show_rgb_seg_dep(camera, img_rgb, img_seg, img_dep)

    # **************** scene_pcd ****************
    cameraInMatrix = get_camera_intrinsic_matrix(env.sim, camera, 512, 512)
    cameraPoseMatrix = get_camera_extrinsic_matrix(env.sim, camera)
    mask = np.zeros(img_seg.shape[:2], dtype=np.bool)
    mask[np.where(img_seg == 0)] = 1
    scene_pcd = tf.create_point_cloud(img_rgb, img_dep, cameraInMatrix, mask)
    # 转到世界坐标系下
    scene_pcd = tf.transPts_T(scene_pcd, T_f2_f1=cameraPoseMatrix)
    # 去除工作范围外的点云
    scene_pcd_norm = scene_pcd - np.array([0, 0, 0.8])
    scene_pcd = np.delete(scene_pcd, np.where(np.abs(scene_pcd_norm) > 0.6)[0], axis=0)
    # FPS
    scene_pcd = tf.farthest_point_sample(scene_pcd, npoint=1024)
    # 删除离群点
    # scene_pcd = tf.removeOutLier_pcl(scene_pcd, nb_points=20, radius=0.1)
    # 补全
    # short_points_num = max(1024-scene_pcd.shape[0], 0)
    # if short_points_num > 0:
    #     extra_points = np.expand_dims(scene_pcd[0], axis=0).repeat(short_points_num, axis=0)
    #     scene_pcd = np.concatenate((scene_pcd, extra_points), axis=0)

    if visual:
        # show
        show_pcd(scene_pcd)
    
    # **************** object_pcd ****************
    object_pcd = env.get_object_pcd(num=1024)
    if visual:
        # show
        show_pcd(object_pcd)
        # show
        object_pcd_in_scene = tf.transPts_tq(object_pcd, object_goal_pose[:3], object_goal_pose[3:])
        pcd = np.concatenate((scene_pcd, object_pcd_in_scene), axis=0)
        show_pcd(pcd)
    
    if dtype is not None:
        scene_pcd = scene_pcd.astype(dtype)
        object_pcd = object_pcd.astype(dtype)
        object_goal_pose = object_goal_pose.astype(dtype)

    return scene_pcd, object_pcd, object_goal_pose


def get_scene_object_pcd_goal_toolhang(dataset_path, visual=False, n=1024, dtype=np.float32):
    """
    获取toolhang任务的scene_pcd / object_pcd / object_goal_pose
    物体点云由深度图得到，得到两个物体点云
    dataset_path: robomimic(toolhang)数据集路径
    visual: 是否可视化图像/pcd
    return:
        - scene_pcd: (N, 3)
        - frame_pcd: (N, 3)
        - tool_pcd: (N, 3)
        - object_goal_pose: (7,) pos+quat
    """
    import robosuite as suite
    from robosuite.controllers import load_controller_config
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_real_depth_map
    import time
    import Mix_diffusion_policy.common.transformation as tf

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

    options = {}

    assert env_meta['env_name'] == 'ToolHang'
    options["env_name"] = env_meta['env_name']
    options["robots"] = env_meta['env_kwargs']["robots"]
    controller_name = "OSC_POSE"
    # Load the desired controller
    options["controller_configs"] = load_controller_config(default_controller=controller_name)
    
    # ****************  ****************
    # initialize the task
    camera = ["agentview", "sideview"]
    segmentation_level = ['instance', 'instance']  # Options are {instance, class, element}
    camera_heights = 512
    camera_widths = 512
    env = suite.make(
        **options,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=20,
        camera_names=camera,
        camera_depths=True,
        camera_heights=camera_heights,
        camera_widths=camera_widths,
    )
    env._camera_segmentations = {camera_name: "instance" for camera_name in [camera[0], camera[1]]}
    env.reset()
    
    # **************** object goal pose ****************
    object_goal_pose = env.object_goal_pose()

    # **************** bird image ****************
    for i in range(20):
        action = [-1, 0, 0, 0, 0, 0, 0]
        obs_bird, reward, done, _ = env.step(action)
    for key in obs_bird.keys():
        print(key, obs_bird[key].shape)
    # segmentation
    # stand:1, frame:2, tool:3
    img_seg_bird = obs_bird[f"{camera[1]}_segmentation"][::-1]
    img_seg_bird_objs = np.zeros_like(img_seg_bird, dtype=int)  
    img_seg_bird_objs[np.where(img_seg_bird == 2)] = 1  # frame
    img_seg_bird_objs[np.where(img_seg_bird == 3)] = 2  # tool
    # rgb
    img_rgb_bird = obs_bird[f"{camera[1]}_image"][::-1]
    # depth
    img_dep_bird = obs_bird[f"{camera[1]}_depth"].squeeze(-1)[::-1]
    img_dep_bird = get_real_depth_map(env.sim, img_dep_bird)

    # **************** front image ****************
    env.remove_all_objects()    # 移除所有物体
    action = [-1, 0, 0, 0, 0, 0, 0]
    obs_agent, reward, done, _ = env.step(action)
    # segmentation
    img_seg_agent = obs_agent[f"{camera[0]}_segmentation_{segmentation_level[0]}"].squeeze(-1)[::-1]
    img_seg_agent[np.where(img_seg_agent < 2)] = 0
    img_seg_agent[np.where(img_seg_agent > 0)] = 1
    # rgb
    img_rgb_agent = obs_agent[f"{camera[0]}_image"][::-1]
    # depth
    img_dep_agent = obs_agent[f"{camera[0]}_depth"].squeeze(-1)[::-1]
    img_dep_agent = get_real_depth_map(env.sim, img_dep_agent)

    if visual:
        show_rgb_seg_dep(camera[1], img_rgb_bird, img_seg_bird_objs, img_dep_bird)
        show_rgb_seg_dep(camera[0], img_rgb_agent, img_seg_agent, img_dep_agent)

    # **************** camera info ****************
    cameraInMatrix_agent = get_camera_intrinsic_matrix(env.sim, camera[0], camera_heights, camera_widths)
    cameraPoseMatrix_agent = get_camera_extrinsic_matrix(env.sim, camera[0])
    cameraInMatrix_bird = get_camera_intrinsic_matrix(env.sim, camera[1], camera_heights, camera_widths)
    cameraPoseMatrix_bird = get_camera_extrinsic_matrix(env.sim, camera[1])

    # **************** scene_pcd ****************
    mask = np.zeros(img_seg_agent.shape[:2], dtype=np.bool)
    mask[np.where(img_seg_agent == 0)] = 1
    scene_pcd = tf.create_point_cloud(img_rgb_agent, img_dep_agent, cameraInMatrix_agent, mask)
    # 转到世界坐标系下
    scene_pcd = tf.transPts_T(scene_pcd, T_f2_f1=cameraPoseMatrix_agent)
    # 去除工作范围外的点云
    scene_pcd_norm = scene_pcd - np.array([0, 0, 0.8])
    scene_pcd = np.delete(scene_pcd, np.where(np.abs(scene_pcd_norm) > 0.6)[0], axis=0)
    # FPS
    scene_pcd = tf.farthest_point_sample(scene_pcd, npoint=n)
    # 删除离群点
    # scene_pcd = tf.removeOutLier_pcl(scene_pcd, nb_points=20, radius=0.1)
    # 补全
    # short_points_num = max(1024-scene_pcd.shape[0], 0)
    # if short_points_num > 0:
    #     extra_points = np.expand_dims(scene_pcd[0], axis=0).repeat(short_points_num, axis=0)
    #     scene_pcd = np.concatenate((scene_pcd, extra_points), axis=0)

    if visual:
        show_pcd(scene_pcd)
    
    # **************** object_pcd (frame) ****************
    mask_frame = np.zeros(img_seg_bird_objs.shape[:2], dtype=np.bool)
    mask_frame[np.where(img_seg_bird_objs == 1)] = 1
    frame_pcd_camera = tf.create_point_cloud(img_rgb_bird, img_dep_bird, cameraInMatrix_bird, mask_frame)
    # 转到世界坐标系下
    frame_pcd_W = tf.transPts_T(frame_pcd_camera, T_f2_f1=cameraPoseMatrix_bird)
    # 删除离群点
    frame_pcd_W = tf.removeOutLier_pcl(frame_pcd_W, nb_points=30, radius=0.01)
    # 转到物体坐标系下
    T_W_frame = tf.PosQua_to_TransMat(obs_bird['frame_pos'], obs_bird['frame_quat'])
    frame_pcd = tf.transPts_T(frame_pcd_W, T_f2_f1=np.linalg.inv(T_W_frame))
    if frame_pcd.shape[0] > n:
        # FPS
        frame_pcd = tf.farthest_point_sample(frame_pcd, npoint=n)
    elif frame_pcd.shape[0] < n:
        # 补全
        short_points_num = n-frame_pcd.shape[0]
        extra_points = np.expand_dims(frame_pcd[0], axis=0).repeat(short_points_num, axis=0)
        frame_pcd = np.concatenate((frame_pcd, extra_points), axis=0)

    # **************** object_pcd (tool) ****************
    mask_tool = np.zeros(img_seg_bird_objs.shape[:2], dtype=np.bool)
    mask_tool[np.where(img_seg_bird_objs == 2)] = 1
    tool_pcd_camera = tf.create_point_cloud(img_rgb_bird, img_dep_bird, cameraInMatrix_bird, mask_tool)
    # 转到世界坐标系下
    tool_pcd_W = tf.transPts_T(tool_pcd_camera, T_f2_f1=cameraPoseMatrix_bird)
    # 删除离群点
    # tool_pcd_W = tf.removeOutLier_pcl(tool_pcd_W, nb_points=30, radius=0.01)
    # 转到物体坐标系下
    T_W_tool = tf.PosQua_to_TransMat(obs_bird['tool_pos'], obs_bird['tool_quat'])
    tool_pcd = tf.transPts_T(tool_pcd_W, T_f2_f1=np.linalg.inv(T_W_tool))
    if tool_pcd.shape[0] > n:
        # FPS
        tool_pcd = tf.farthest_point_sample(tool_pcd, npoint=n)
    elif tool_pcd.shape[0] < n:
        # 补全
        short_points_num = n-tool_pcd.shape[0]
        extra_points = np.expand_dims(tool_pcd[0], axis=0).repeat(short_points_num, axis=0)
        tool_pcd = np.concatenate((tool_pcd, extra_points), axis=0)

    if visual:
        # show objects pcd  in obj frames
        show_pcd(frame_pcd)
        show_pcd(tool_pcd_W)
    
    if dtype is not None:
        scene_pcd = scene_pcd.astype(dtype)
        frame_pcd = frame_pcd.astype(dtype)
        tool_pcd = tool_pcd.astype(dtype)
        object_goal_pose = object_goal_pose.astype(dtype)

    return scene_pcd, frame_pcd, tool_pcd, object_goal_pose



def get_scene_object_pcd_goal_toolhang_nosperate(dataset_path, visual=False, n=1024, dtype=np.float32):
    import robosuite as suite
    from robosuite.controllers import load_controller_config
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_real_depth_map
    import numpy as np
    import Mix_diffusion_policy.common.transformation as tf

    camera_front = "agentview"
    camera_bird = "sideview"
    camera_heights = 512
    camera_widths = 512

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    options = {}
    assert env_meta['env_name'] == 'ToolHang'
    options["env_name"] = env_meta['env_name']
    options["robots"] = env_meta['env_kwargs']["robots"]
    options["controller_configs"] = load_controller_config(default_controller="OSC_POSE")

    env = suite.make(
        **options,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=20,
        camera_names=[camera_front, camera_bird],
        camera_depths=True,
        camera_heights=camera_heights,
        camera_widths=camera_widths,
    )
    env.reset()

    object_goal_pose = env.object_goal_pose()
    obs = env._get_observations()

    # ----- 物体局部点云（物体坐标系）-----
    pc_dict = env.get_part_point_clouds_local_from_sim(num_points=2048)
    tool_pc = pc_dict["tool"]
    frame_pc = pc_dict["frame"]
    stand_pc = pc_dict["stand"]

    def sample_points(pcd, n):
        if pcd.shape[0] > n:
            return tf.farthest_point_sample(pcd, npoint=n)
        elif pcd.shape[0] < n:
            short = n - pcd.shape[0]
            extra = np.expand_dims(pcd[0], axis=0).repeat(short, axis=0)
            return np.concatenate([pcd, extra], axis=0)
        return pcd

    tool_pc = sample_points(tool_pc, int(n/4))
    frame_pc = sample_points(frame_pc, int(n/4))
    stand_pc = sample_points(stand_pc, int(n/2))

    # ----- 获取底座在世界坐标系中的位姿（用于可视化 stand 点云）-----
    base_pos = obs['base_pos']      # 底座位姿
    base_quat = obs['base_quat']
    T_world_stand = tf.PosQua_to_TransMat(base_pos, base_quat)
    stand_pc_world = tf.transPts_T(stand_pc, T_f2_f1=T_world_stand)

    # ----- 场景点云（世界坐标系）-----
    img_depth = obs[f"{camera_front}_depth"].squeeze(-1)[::-1]
    img_depth = get_real_depth_map(env.sim, img_depth)
    img_rgb = obs[f"{camera_front}_image"][::-1]
    cam_int = get_camera_intrinsic_matrix(env.sim, camera_front, camera_heights, camera_widths)
    cam_ext = get_camera_extrinsic_matrix(env.sim, camera_front)

    img_depth_bird = obs[f"{camera_bird}_depth"].squeeze(-1)[::-1]
    img_depth_bird = get_real_depth_map(env.sim, img_depth_bird)
    img_rgb_bird = obs[f"{camera_bird}_image"][::-1]
    cam_int_bird = get_camera_intrinsic_matrix(env.sim, camera_bird, camera_heights, camera_widths)
    cam_ext_bird = get_camera_extrinsic_matrix(env.sim, camera_bird)

    full_pcd_cam = tf.create_point_cloud(img_rgb, img_depth, cam_int, workspace_mask=None)
    full_pcd_world = tf.transPts_T(full_pcd_cam, T_f2_f1=cam_ext)

    full_pcd_cam_bird = tf.create_point_cloud(img_rgb_bird, img_depth_bird, cam_int_bird, workspace_mask=None)
    full_pcd_world_bird = tf.transPts_T(full_pcd_cam_bird, T_f2_f1=cam_ext_bird)

    full_pcd_world = np.concatenate([full_pcd_world, full_pcd_world_bird])

    # 裁剪工作空间
    scene_pcd_norm = full_pcd_world - np.array([0, 0, 0.8])
    full_pcd_world = np.delete(full_pcd_world, np.where(np.abs(scene_pcd_norm) > 0.6)[0], axis=0)

    # 剔除物体点云：需要世界坐标系下的物体点云用于剔除
    frame_pos = obs['frame_pos']
    frame_quat = obs['frame_quat']
    tool_pos = obs['tool_pos']
    tool_quat = obs['tool_quat']

    T_world_frame = tf.PosQua_to_TransMat(frame_pos, frame_quat)
    frame_pcd_world = tf.transPts_T(frame_pc, T_f2_f1=T_world_frame)
    T_world_tool = tf.PosQua_to_TransMat(tool_pos, tool_quat)
    tool_pcd_world = tf.transPts_T(tool_pc, T_f2_f1=T_world_tool)

    from scipy.spatial import cKDTree
    threshold = 0.01
    tree_frame = cKDTree(frame_pcd_world)
    tree_tool = cKDTree(tool_pcd_world)
    tree_stand = cKDTree(stand_pc_world)

    dist_frame, _ = tree_frame.query(full_pcd_world, k=1)
    dist_tool, _ = tree_tool.query(full_pcd_world, k=1)
    dist_stand, _ = tree_stand.query(full_pcd_world, k=1)
    
    mask_scene = (dist_frame > threshold) & (dist_tool > threshold) & (dist_stand > threshold)
    scene_pcd_world = full_pcd_world#[mask_scene]

    scene_pcd_world = sample_points(scene_pcd_world, n)

    if visual:
        # 可视化世界坐标系下的点云
        total_pcd = np.concatenate([scene_pcd_world, frame_pcd_world, tool_pcd_world, stand_pc_world], axis=0)
        show_two_pcds(sample_points(full_pcd_world,1000),total_pcd)
        show_pcd(frame_pc)          # 局部坐标系下的框架点云
        show_pcd(tool_pc)           # 局部坐标系下的工具点云
        show_pcd(stand_pc)          # 局部坐标系下的底座点云（原始）
        def quat_xyzw_to_matrix(position, quat_xyzw):
            """
            将位置和四元数（x, y, z, w 顺序）转换为 4x4 变换矩阵
            参数:
                position: (3,) 平移向量
                quat_xyzw: (4,) 四元数 [x, y, z, w]
            返回:
                4x4 numpy 矩阵，表示从局部坐标系到世界坐标系的变换
            """
            rot = R.from_quat(quat_xyzw).as_matrix()  # scipy 要求四元数顺序为 (x, y, z, w)
            T = np.eye(4)
            T[:3, :3] = rot
            T[:3, 3] = position
            return T        
        object1_pose = quat_xyzw_to_matrix(frame_pos, frame_quat)
        object2_pose = quat_xyzw_to_matrix(tool_pos, tool_quat)
        object3_pose = quat_xyzw_to_matrix(base_pos, base_quat)
        show_pcd_tf(scene_pcd_world, world_frame=True, camera_pose=cam_ext, frame_scale=0.2,obj_pcd=total_pcd,
                    obj_pose1=object1_pose, obj_pose2=object2_pose, obj_pose3=object3_pose)
        show_pcd(np.concatenate([frame_pcd_world, tool_pcd_world, stand_pc_world], axis=0))

    if dtype is not None:
        scene_pcd_world = scene_pcd_world.astype(dtype)
        tool_pc = tool_pc.astype(dtype)
        frame_pc = frame_pc.astype(dtype)
        stand_pc = stand_pc.astype(dtype)
        object_goal_pose = object_goal_pose.astype(dtype)

    return scene_pcd_world, tool_pc, frame_pc, stand_pc, object_goal_pose

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

import numpy as np
from scipy.spatial.transform import Rotation as R

# ======================== 辅助函数 ========================
def quat_to_rot_matrix(q):
    """四元数 (w, x, y, z) -> 3x3 旋转矩阵"""
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

def pos_quat_to_transform(pos, quat):
    """位置 + 四元数 -> 4x4 齐次变换矩阵 (世界→物体)"""
    T = np.eye(4)
    T[:3, :3] = quat_to_rot_matrix(quat)
    T[:3, 3] = pos
    return T

def transform_point(point, transform):
    """应用齐次变换: point (3,) -> new_point (3,)"""
    p_h = np.append(point, 1.0)
    return (transform @ p_h)[:3]

def check_poses_similarity(pose1, pose2, pos_th=0.01, euler_th=0.1):
    """判断两个位姿是否相似"""
    pos1, quat1 = pose1[:3], pose1[3:]
    pos2, quat2 = pose2[:3], pose2[3:]
    if np.linalg.norm(pos1 - pos2) > pos_th:
        return False
    r1 = R.from_quat([quat1[1], quat1[2], quat1[3], quat1[0]])
    r2 = R.from_quat([quat2[1], quat2[2], quat2[3], quat2[0]])
    angle = r1.inv() * r2
    angle_magnitude = np.linalg.norm(angle.as_rotvec())
    return angle_magnitude < euler_th

def min_distance_to_pointcloud(point, pcd_local, obj_pos, obj_quat):
    """计算世界点到物体局部点云的最小距离"""
    T_WO = pos_quat_to_transform(obj_pos, obj_quat)
    T_OW = np.linalg.inv(T_WO)
    point_local = transform_point(point, T_OW)
    dists = np.linalg.norm(pcd_local - point_local, axis=1)
    return np.min(dists)

def get_subgoals_assembly_contact(
        raw_obs: dict,
        tool_pcd: np.ndarray,
        frame_pcd: np.ndarray,
        fin_rad: float,
        sim_thresh: list,
        max_reward=10,
        reward_mode='tanh',
        Tr=1
):
    """
    基于手指与工具/框架的接触生成装配子目标。
    支持单指接触检测、阶段过滤（frame优先）、相似性过滤、向后优先填充。
    """

    sequence_length = raw_obs['object'].shape[0]
    contact_thresh_frame = fin_rad + 0.03
    contact_thresh_tool = fin_rad + 0.03

    frame_done_arr = raw_obs['object'][:, 42] > 0.5
    tool_done_arr = raw_obs['object'][:, 43] > 0.5

    # 后备：最后一帧指尖世界坐标（使用 getFingersPos 并加偏移）
    last_fl, last_fr = getFingersPos(
        raw_obs['robot0_eef_pos'][-1],
        raw_obs['robot0_eef_quat'][-1],
        raw_obs['robot0_gripper_qpos'][-1, 0] + 0.0145/2,
        raw_obs['robot0_gripper_qpos'][-1, 1] - 0.0145/2
    )

    # 存储每步信息：物体局部系下的手指位置和接触标志
    fin_local_per_step = []  # 每个元素: {'frame': (8,), 'tool': (8,)}

    last_fl_contact_frame = last_fr_contact_frame = False
    last_fl_contact_tool = last_fr_contact_tool = False

    frame_obj_subgoals = []   # {'step': int, 'obj_pose': (7,), 'fin_local': (8,)}
    tool_obj_subgoals = []

    for step in range(sequence_length):
        frame_pos = raw_obs['object'][step,21:24]
        frame_quat = raw_obs['object'][step,17:21]
        tool_pos = raw_obs['object'][step,28:31]
        tool_quat = raw_obs['object'][step,31:35]
        eef_pos = raw_obs['robot0_eef_pos'][step]
        eef_quat = raw_obs['robot0_eef_quat'][step]
        gripper = raw_obs['robot0_gripper_qpos'][step]
        # 替换为 getFingersPos 并加偏移
        fl, fr = getFingersPos(eef_pos, eef_quat, gripper[0] + 0.0145/2, gripper[1] - 0.0145/2)
        T_W_Frame = pos_quat_to_transform(frame_pos, frame_quat)
        T_W_Tool = pos_quat_to_transform(tool_pos, tool_quat)
        frame_pcd_world = tf.transPts_T(frame_pcd, T_W_Frame)
        tool_pcd_world = tf.transPts_T(tool_pcd, T_W_Tool)
        total_pcd_world = np.concatenate([frame_pcd_world, tool_pcd_world], axis=0)
        show_pcd_finger(total_pcd_world,fl,fr)
        # 接触检测（单指）
        fl_dist_frame = min_distance_to_pointcloud(fl, frame_pcd, frame_pos, frame_quat)
        fr_dist_frame = min_distance_to_pointcloud(fr, frame_pcd, frame_pos, frame_quat)
        is_fl_contact_frame = fl_dist_frame < contact_thresh_frame
        is_fr_contact_frame = fr_dist_frame < contact_thresh_frame

        is_contact_frame = is_fr_contact_frame and is_fl_contact_frame
        is_fr_contact_frame = is_contact_frame
        is_fl_contact_frame = is_contact_frame
        
        fl_dist_tool = min_distance_to_pointcloud(fl, tool_pcd, tool_pos, tool_quat)
        fr_dist_tool = min_distance_to_pointcloud(fr, tool_pcd, tool_pos, tool_quat)
        is_fl_contact_tool = fl_dist_tool < contact_thresh_tool
        is_fr_contact_tool = fr_dist_tool < contact_thresh_tool
        
        is_contact_tool = is_fr_contact_tool and is_fl_contact_tool
        is_fr_contact_tool = is_contact_tool
        is_fl_contact_tool = is_contact_tool    

        # 局部坐标变换（世界→物体局部）
        T_W_Frame = pos_quat_to_transform(frame_pos, frame_quat)
        T_Frame_W = np.linalg.inv(T_W_Frame)
        fl_frame = transform_point(fl, T_Frame_W)
        fr_frame = transform_point(fr, T_Frame_W)
        fin_local_frame = np.array([*fl_frame, *fr_frame, is_fl_contact_frame, is_fr_contact_frame])

        T_W_Tool = pos_quat_to_transform(tool_pos, tool_quat)
        T_Tool_W = np.linalg.inv(T_W_Tool)
        fl_tool = transform_point(fl, T_Tool_W)
        fr_tool = transform_point(fr, T_Tool_W)
        fin_local_tool = np.array([*fl_tool, *fr_tool, is_fl_contact_tool, is_fr_contact_tool])

        fin_local_per_step.append({'frame': fin_local_frame, 'tool': fin_local_tool})

        # 记录物体子目标（接触状态变化时）
        if not frame_done_arr[step]:
            change_frame = (is_fl_contact_frame != last_fl_contact_frame) or (is_fr_contact_frame != last_fr_contact_frame)
            if change_frame:
                frame_obj_subgoals.append({
                    'step': step,
                    'obj_pose': np.concatenate([frame_pos, frame_quat]),
                    'fin_local': fin_local_frame.copy()
                })
        if frame_done_arr[step] and not tool_done_arr[step]:
            change_tool = (is_fl_contact_tool != last_fl_contact_tool) or (is_fr_contact_tool != last_fr_contact_tool)
            if change_tool:
                tool_obj_subgoals.append({
                    'step': step,
                    'obj_pose': np.concatenate([tool_pos, tool_quat]),
                    'fin_local': fin_local_tool.copy()
                })

        last_fl_contact_frame, last_fr_contact_frame = is_fl_contact_frame, is_fr_contact_frame
        last_fl_contact_tool, last_fr_contact_tool = is_fl_contact_tool, is_fr_contact_tool

    # ---------- 相似性过滤（整个子目标清零） ----------
    i = 0
    while i < len(frame_obj_subgoals) - 1:
        pose_i = frame_obj_subgoals[i]['obj_pose']
        pose_ip1 = frame_obj_subgoals[i+1]['obj_pose']
        if check_poses_similarity(pose_i, pose_ip1, sim_thresh[0], sim_thresh[1]):
            start = frame_obj_subgoals[i]['step']
            end = frame_obj_subgoals[i+1]['step']
            for s in range(start, end):
                if s < len(fin_local_per_step):
                    fin_local_per_step[s]['frame'] = np.zeros(8)
            frame_obj_subgoals.pop(i)
        else:
            i += 1

    i = 0
    while i < len(tool_obj_subgoals) - 1:
        pose_i = tool_obj_subgoals[i]['obj_pose']
        pose_ip1 = tool_obj_subgoals[i+1]['obj_pose']
        if check_poses_similarity(pose_i, pose_ip1, sim_thresh[0], sim_thresh[1]):
            start = tool_obj_subgoals[i]['step']
            end = tool_obj_subgoals[i+1]['step']
            for s in range(start, end):
                if s < len(fin_local_per_step):
                    fin_local_per_step[s]['tool'] = np.zeros(8)
            tool_obj_subgoals.pop(i)
        else:
            i += 1

    # ---------- 向后填充局部子目标 ----------
    frame_fin_filled = [np.zeros(8) for _ in range(sequence_length)]
    last_valid = None
    for step in range(sequence_length-1, -1, -1):
        fin = fin_local_per_step[step]['frame']
        if fin[6] > 0.5 or fin[7] > 0.5:
            last_valid = fin.copy()
        if last_valid is not None:
            frame_fin_filled[step] = last_valid.copy()

    tool_fin_filled = [np.zeros(8) for _ in range(sequence_length)]
    last_valid = None
    for step in range(sequence_length-1, -1, -1):
        fin = fin_local_per_step[step]['tool']
        if fin[6] > 0.5 or fin[7] > 0.5:
            last_valid = fin.copy()
        if last_valid is not None:
            tool_fin_filled[step] = last_valid.copy()

    # ---------- 根据完成阶段选择最终局部子目标和对应的物体位姿 ----------
    final_fin_local = []      # (type, data(8))
    final_obj_pose = []       # (7,) or None
    for step in range(sequence_length):
        if not frame_done_arr[step]:   # frame 阶段
            fin = frame_fin_filled[step]
            if np.all(fin[6:] == 0):   # 无有效接触
                final_fin_local.append(('world', np.concatenate([last_fl, last_fr, [1.0, 1.0]])))
                final_obj_pose.append(None)
            else:
                final_fin_local.append(('frame', fin))
                final_obj_pose.append(np.concatenate([raw_obs['object'][step, 14:17], raw_obs['object'][step, 17:21]]))
        else:                          # tool 阶段
            fin = tool_fin_filled[step]
            if np.all(fin[6:] == 0):
                final_fin_local.append(('world', np.concatenate([last_fl, last_fr, [1.0, 1.0]])))
                final_obj_pose.append(None)
            else:
                final_fin_local.append(('tool', fin))
                final_obj_pose.append(np.concatenate([raw_obs['object'][step, 28:31], raw_obs['object'][step, 31:35]]))

    # 前移一位（与输出对齐）
    final_fin_local.pop(0)
    final_obj_pose.pop(0)
    final_fin_local.append(('none', np.zeros(8)))
    final_obj_pose.append(None)

    # ---------- 生成世界坐标系下的子目标 ----------
    fin_sgs = []
    next_fin_sgs = []
    goal_thresh = fin_rad * 1.2

    for step in range(sequence_length - Tr):
        entry_type, fin_local = final_fin_local[step]
        if entry_type == 'world':
            fl_sg = fin_local[:3]
            fr_sg = fin_local[3:6]
            cf, cr = fin_local[6] > 0.5, fin_local[7] > 0.5
        else:
            obj_pose = final_obj_pose[step]
            if obj_pose is None:
                fl_sg, fr_sg, cf, cr = np.zeros(3), np.zeros(3), False, False
            else:
                # 使用记录时的物体位姿进行变换（关键修复）
                T_WO = pos_quat_to_transform(obj_pose[:3], obj_pose[3:7])
                fl_sg = transform_point(fin_local[:3], T_WO)
                fr_sg = transform_point(fin_local[3:6], T_WO)
                cf, cr = fin_local[6] > 0.5, fin_local[7] > 0.5
        fin_sgs.append(np.concatenate([fl_sg, fr_sg, [float(cf)], [float(cr)]]))

        # 下一时刻子目标（Tr步后）
        if step + Tr < sequence_length:
            n_type, n_fin = final_fin_local[step+Tr]
            if n_type == 'world':
                n_fl = n_fin[:3]
                n_fr = n_fin[3:6]
                n_cf, n_cr = n_fin[6] > 0.5, n_fin[7] > 0.5
            else:
                n_obj = final_obj_pose[step+Tr]
                if n_obj is None:
                    n_fl, n_fr, n_cf, n_cr = np.zeros(3), np.zeros(3), False, False
                else:
                    T_WN = pos_quat_to_transform(n_obj[:3], n_obj[3:7])
                    n_fl = transform_point(n_fin[:3], T_WN)
                    n_fr = transform_point(n_fin[3:6], T_WN)
                    n_cf, n_cr = n_fin[6] > 0.5, n_fin[7] > 0.5

            next_fin_sgs.append(np.concatenate([n_fl, n_fr, [float(n_cf)], [float(n_cr)]]))
        else:
            next_fin_sgs.append(np.zeros(8))

    # ---------- 奖励计算（修复越界） ----------
    reward = []
    for step in range(sequence_length - Tr):
        r = 0
        fl_sg = fin_sgs[step][:3]
        fr_sg = fin_sgs[step][3:6]
        cf = fin_sgs[step][6] > 0.5
        cr = fin_sgs[step][7] > 0.5
        if not (cf and cr):
            r = 0.1 if reward_mode == 'tanh' else 0.0
        else:
            eef_pos = raw_obs['robot0_eef_pos'][step]
            eef_quat = raw_obs['robot0_eef_quat'][step]
            gripper = raw_obs['robot0_gripper_qpos'][step]
            # 替换为 getFingersPos 并加偏移
            fl, fr = getFingersPos(eef_pos, eef_quat, gripper[0] + 0.0145/2, gripper[1] - 0.0145/2)
            fl_dist = np.linalg.norm(fl - fl_sg)
            fr_dist = np.linalg.norm(fr - fr_sg)
            if max(fl_dist, fr_dist) < goal_thresh:
                r = max_reward
            else:
                reached = False
                for n in range(1, Tr+1):
                    if step + n >= len(fin_sgs):
                        break
                    _eef = raw_obs['robot0_eef_pos'][step+n]
                    _eef_q = raw_obs['robot0_eef_quat'][step+n]
                    _g = raw_obs['robot0_gripper_qpos'][step+n]
                    # 替换为 getFingersPos 并加偏移
                    _fl, _fr = getFingersPos(_eef, _eef_q, _g[0] + 0.0145/2, _g[1] - 0.0145/2)
                    _fl_sg = fin_sgs[step+n][:3]
                    _fr_sg = fin_sgs[step+n][3:6]
                    _fl_d = np.linalg.norm(_fl - _fl_sg)
                    _fr_d = np.linalg.norm(_fr - _fr_sg)
                    if max(_fl_d, _fr_d) < goal_thresh:
                        r = max_reward
                        reached = True
                        break
                if not reached:
                    if reward_mode == 'only_success':
                        r = 0
                    elif reward_mode == 'tanh':
                        w = 3
                        r_fl = -np.tanh(fl_dist * w)
                        r_fr = -np.tanh(fr_dist * w)
                        r = (r_fl + r_fr) / 3 * 2 + 1
        reward.append(r)

    return {
        'subgoal': np.array(fin_sgs),
        'next_subgoal': np.array(next_fin_sgs),
        'reward': np.array(reward)
    }
def get_subgoals_assembly_contact_single(
        raw_obs: dict,
        tool_pcd: np.ndarray,
        frame_pcd: np.ndarray,
        fin_rad: float,
        sim_thresh: list,
        max_reward=10,
        reward_mode='tanh',
        Tr=1
):
    """
    基于手指与工具/框架的接触生成装配子目标。
    支持单指接触检测、阶段过滤（frame优先）、相似性过滤、向后优先填充。
    接触判断方式参考 get_subgoals_stage_robomimic：物体局部坐标系下点云最近距离判断，每个手指独立。
    修改：不再使用 last_fl / last_fr 后备，无有效接触时使用零向量且接触标志为 False。
    """

    sequence_length = raw_obs['object'].shape[0]
    contact_thresh = fin_rad + 0.1

    frame_done_arr = raw_obs['object'][:, 42] > 0.5
    tool_done_arr = raw_obs['object'][:, 43] > 0.5

    # 存储每步信息：物体局部系下的手指位置和接触标志
    fin_local_per_step = []  # 每个元素: {'frame': (8,), 'tool': (8,)}

    last_fl_contact_frame = last_fr_contact_frame = False
    last_fl_contact_tool = last_fr_contact_tool = False

    frame_obj_subgoals = []   # {'step': int, 'obj_pose': (7,), 'fin_local': (8,)}
    tool_obj_subgoals = []

    for step in range(sequence_length):
        # 当前帧物体位姿
        frame_pos = raw_obs['object'][step,14:17]
        frame_quat = raw_obs['object'][step, 17:21]
        tool_pos = raw_obs['object'][step, 28:31]
        tool_quat = raw_obs['object'][step, 31:35]
        eef_pos = raw_obs['robot0_eef_pos'][step]
        eef_quat = raw_obs['robot0_eef_quat'][step]
        gripper = raw_obs['robot0_gripper_qpos'][step]
        # 手指世界坐标
        fl, fr = getFingersPos(eef_pos, eef_quat, gripper[0] + 0.0145/2, gripper[1] - 0.0145/2)

        # ---------- 接触检测（局部坐标系下点云最近距离，独立手指）----------
        # Frame 部分
        T_W_Frame = pos_quat_to_transform(frame_pos, frame_quat)
        T_Frame_W = np.linalg.inv(T_W_Frame)
        fl_frame = transform_point(fl, T_Frame_W)
        fr_frame = transform_point(fr, T_Frame_W)
        #show_pcd_finger(frame_pcd,fl_pos=fl_frame,fr_pos=fr_frame)
        
        # 计算到 frame_pcd（局部点云）的最小距离
        dist_fl_frame = np.min(np.linalg.norm(frame_pcd - fl_frame, axis=1)) if len(frame_pcd) > 0 else np.inf
        dist_fr_frame = np.min(np.linalg.norm(frame_pcd - fr_frame, axis=1)) if len(frame_pcd) > 0 else np.inf
        is_fl_contact_frame = dist_fl_frame < contact_thresh
        is_fr_contact_frame = dist_fr_frame < contact_thresh
        print(f"Frame: {is_fl_contact_frame}, {is_fr_contact_frame}")
        # Tool 部分
        T_W_Tool = pos_quat_to_transform(tool_pos, tool_quat)
        T_Tool_W = np.linalg.inv(T_W_Tool)
        fl_tool = transform_point(fl, T_Tool_W)
        fr_tool = transform_point(fr, T_Tool_W)
        #show_pcd_finger(tool_pcd,fl_pos=fl_tool,fr_pos=fr_tool)
        dist_fl_tool = np.min(np.linalg.norm(tool_pcd - fl_tool, axis=1)) if len(tool_pcd) > 0 else np.inf
        dist_fr_tool = np.min(np.linalg.norm(tool_pcd - fr_tool, axis=1)) if len(tool_pcd) > 0 else np.inf
        is_fl_contact_tool = dist_fl_tool < contact_thresh
        is_fr_contact_tool = dist_fr_tool < contact_thresh
        print(f"Tool: {is_fl_contact_tool}, {is_fr_contact_tool}")
        # 记录局部手指位置和接触标志（8维：左指尖xyz，右指尖xyz，左接触标志，右接触标志）
        fin_local_frame = np.array([*fl_frame, *fr_frame, is_fl_contact_frame, is_fr_contact_frame])
        fin_local_tool = np.array([*fl_tool, *fr_tool, is_fl_contact_tool, is_fr_contact_tool])
        fin_local_per_step.append({'frame': fin_local_frame, 'tool': fin_local_tool})

        # 记录物体子目标（接触状态变化时）
        if not frame_done_arr[step]:
            change_frame = (is_fl_contact_frame != last_fl_contact_frame) or (is_fr_contact_frame != last_fr_contact_frame)
            if change_frame:
                frame_obj_subgoals.append({
                    'step': step,
                    'obj_pose': np.concatenate([frame_pos, frame_quat]),
                    'fin_local': fin_local_frame.copy()
                })
        if frame_done_arr[step] and not tool_done_arr[step]:
            change_tool = (is_fl_contact_tool != last_fl_contact_tool) or (is_fr_contact_tool != last_fr_contact_tool)
            if change_tool:
                tool_obj_subgoals.append({
                    'step': step,
                    'obj_pose': np.concatenate([tool_pos, tool_quat]),
                    'fin_local': fin_local_tool.copy()
                })

        last_fl_contact_frame, last_fr_contact_frame = is_fl_contact_frame, is_fr_contact_frame
        last_fl_contact_tool, last_fr_contact_tool = is_fl_contact_tool, is_fr_contact_tool

    # ---------- 相似性过滤（整个子目标清零） ----------
    i = 0
    while i < len(frame_obj_subgoals) - 1:
        pose_i = frame_obj_subgoals[i]['obj_pose']
        pose_ip1 = frame_obj_subgoals[i+1]['obj_pose']
        if check_poses_similarity(pose_i, pose_ip1, sim_thresh[0], sim_thresh[1]):
            start = frame_obj_subgoals[i]['step']
            end = frame_obj_subgoals[i+1]['step']
            for s in range(start, end):
                if s < len(fin_local_per_step):
                    fin_local_per_step[s]['frame'] = np.zeros(8)
            frame_obj_subgoals.pop(i)
        else:
            i += 1

    i = 0
    while i < len(tool_obj_subgoals) - 1:
        pose_i = tool_obj_subgoals[i]['obj_pose']
        pose_ip1 = tool_obj_subgoals[i+1]['obj_pose']
        if check_poses_similarity(pose_i, pose_ip1, sim_thresh[0], sim_thresh[1]):
            start = tool_obj_subgoals[i]['step']
            end = tool_obj_subgoals[i+1]['step']
            for s in range(start, end):
                if s < len(fin_local_per_step):
                    fin_local_per_step[s]['tool'] = np.zeros(8)
            tool_obj_subgoals.pop(i)
        else:
            i += 1

    # ---------- 向后填充局部子目标 ----------
    frame_fin_filled = [np.zeros(8) for _ in range(sequence_length)]
    last_valid = None
    for step in range(sequence_length-1, -1, -1):
        fin = fin_local_per_step[step]['frame']
        if fin[6] > 0.5 or fin[7] > 0.5:
            last_valid = fin.copy()
        if last_valid is not None:
            frame_fin_filled[step] = last_valid.copy()

    tool_fin_filled = [np.zeros(8) for _ in range(sequence_length)]
    last_valid = None
    for step in range(sequence_length-1, -1, -1):
        fin = fin_local_per_step[step]['tool']
        if fin[6] > 0.5 or fin[7] > 0.5:
            last_valid = fin.copy()
        if last_valid is not None:
            tool_fin_filled[step] = last_valid.copy()

    # ---------- 根据完成阶段选择最终局部子目标和对应的物体位姿 ----------
    final_fin_local = []      # (type, data(8))
    final_obj_pose = []       # (7,) or None
    for step in range(sequence_length):
        if not frame_done_arr[step]:   # frame 阶段
            fin = frame_fin_filled[step]
            if np.all(fin[6:] == 0):   # 无有效接触
                # 修改：不再使用 last_fl/last_fr，改用 'none' 类型和全零数据
                final_fin_local.append(('none', np.zeros(8)))
                final_obj_pose.append(None)
            else:
                final_fin_local.append(('frame', fin))
                final_obj_pose.append(np.concatenate([raw_obs['object'][step, 21:24], raw_obs['object'][step, 24:28]]))
        else:                          # tool 阶段
            fin = tool_fin_filled[step]
            if np.all(fin[6:] == 0):
                final_fin_local.append(('none', np.zeros(8)))
                final_obj_pose.append(None)
            else:
                final_fin_local.append(('tool', fin))
                final_obj_pose.append(np.concatenate([raw_obs['object'][step, 35:38], raw_obs['object'][step, 38:42]]))

    # 前移一位（与输出对齐）
    final_fin_local.pop(0)
    final_obj_pose.pop(0)
    final_fin_local.append(('none', np.zeros(8)))
    final_obj_pose.append(None)

    # ---------- 生成世界坐标系下的子目标 ----------
    fin_sgs = []
    next_fin_sgs = []
    goal_thresh = fin_rad * 1.2

    for step in range(sequence_length - Tr):
        entry_type, fin_local = final_fin_local[step]
        if entry_type == 'world':
            # 此分支理论上不再出现，保留以兼容可能存在的旧数据
            fl_sg = fin_local[:3]
            fr_sg = fin_local[3:6]
            cf, cr = fin_local[6] > 0.5, fin_local[7] > 0.5
        elif entry_type == 'none':
            fl_sg, fr_sg, cf, cr = np.zeros(3), np.zeros(3), False, False
        else:  # 'frame' or 'tool'
            obj_pose = final_obj_pose[step]
            if obj_pose is None:
                fl_sg, fr_sg, cf, cr = np.zeros(3), np.zeros(3), False, False
            else:
                T_WO = pos_quat_to_transform(obj_pose[:3], obj_pose[3:7])
                fl_sg = transform_point(fin_local[:3], T_WO)
                fr_sg = transform_point(fin_local[3:6], T_WO)
                cf, cr = fin_local[6] > 0.5, fin_local[7] > 0.5
        fin_sgs.append(np.concatenate([fl_sg, fr_sg, [float(cf)], [float(cr)]]))

        # 下一时刻子目标（Tr步后）
        if step + Tr < sequence_length:
            n_type, n_fin = final_fin_local[step+Tr]
            if n_type == 'world':
                n_fl = n_fin[:3]
                n_fr = n_fin[3:6]
                n_cf, n_cr = n_fin[6] > 0.5, n_fin[7] > 0.5
            elif n_type == 'none':
                n_fl, n_fr, n_cf, n_cr = np.zeros(3), np.zeros(3), False, False
            else:
                n_obj = final_obj_pose[step+Tr]
                if n_obj is None:
                    n_fl, n_fr, n_cf, n_cr = np.zeros(3), np.zeros(3), False, False
                else:
                    T_WN = pos_quat_to_transform(n_obj[:3], n_obj[3:7])
                    n_fl = transform_point(n_fin[:3], T_WN)
                    n_fr = transform_point(n_fin[3:6], T_WN)
                    n_cf, n_cr = n_fin[6] > 0.5, n_fin[7] > 0.5
            next_fin_sgs.append(np.concatenate([n_fl, n_fr, [float(n_cf)], [float(n_cr)]]))
        else:
            next_fin_sgs.append(np.zeros(8))

    # ---------- 奖励计算 ----------
    reward = []
    for step in range(sequence_length - Tr):
        r = 0
        fl_sg = fin_sgs[step][:3]
        fr_sg = fin_sgs[step][3:6]
        cf = fin_sgs[step][6] > 0.5
        cr = fin_sgs[step][7] > 0.5
        if not (cf and cr):
            r = 0.1 if reward_mode == 'tanh' else 0.0
        else:
            eef_pos = raw_obs['robot0_eef_pos'][step]
            eef_quat = raw_obs['robot0_eef_quat'][step]
            gripper = raw_obs['robot0_gripper_qpos'][step]
            fl, fr = getFingersPos(eef_pos, eef_quat, gripper[0] + 0.0145/2, gripper[1] - 0.0145/2)
            fl_dist = np.linalg.norm(fl - fl_sg)
            fr_dist = np.linalg.norm(fr - fr_sg)
            if max(fl_dist, fr_dist) < goal_thresh:
                r = max_reward
            else:
                reached = False
                for n in range(1, Tr+1):
                    if step + n >= len(fin_sgs):
                        break
                    _eef = raw_obs['robot0_eef_pos'][step+n]
                    _eef_q = raw_obs['robot0_eef_quat'][step+n]
                    _g = raw_obs['robot0_gripper_qpos'][step+n]
                    _fl, _fr = getFingersPos(_eef, _eef_q, _g[0] + 0.0145/2, _g[1] - 0.0145/2)
                    _fl_sg = fin_sgs[step+n][:3]
                    _fr_sg = fin_sgs[step+n][3:6]
                    _fl_d = np.linalg.norm(_fl - _fl_sg)
                    _fr_d = np.linalg.norm(_fr - _fr_sg)
                    if max(_fl_d, _fr_d) < goal_thresh:
                        r = max_reward
                        reached = True
                        break
                if not reached:
                    if reward_mode == 'only_success':
                        r = 0
                    elif reward_mode == 'tanh':
                        w = 3
                        r_fl = -np.tanh(fl_dist * w)
                        r_fr = -np.tanh(fr_dist * w)
                        r = (r_fl + r_fr) / 3 * 2 + 1
        reward.append(r)

    return {
        'subgoal': np.array(fin_sgs),
        'next_subgoal': np.array(next_fin_sgs),
        'reward': np.array(reward)
    }

import numpy as np
from typing import Dict, List, Tuple

# 假设存在以下辅助函数（需根据实际环境实现或导入）
# def getFingersPos(eef_pos, eef_quat, left_q, right_q) -> Tuple[np.ndarray, np.ndarray]:
#     """返回左右手指在世界坐标系中的位置 (3,)"""
#     pass
#
# tf 模块提供坐标变换函数：
# def PosQua_to_TransMat(pos, qua) -> np.ndarray: ...
# def transPt(pt, T_f2_f1=None, t_f2_f1=None, q_f2_f1=None) -> np.ndarray: ...
#
# def check_poses_similarity(pose1, pose2, pos_th, euler_th) -> bool:
#     """判断两个位姿（位置+四元数）是否相似"""
#     pass


def get_subgoals_assembly_contact_new(
        raw_obs: dict,
        tool_pcd: np.ndarray,
        frame_pcd: np.ndarray,
        fin_rad: float,
        sim_thresh: list,
        max_reward=10,
        reward_mode='tanh',
        Tr=1
):
    """
    计算装配任务（先frame后tool）的手指位置子目标

    args:
        - raw_obs: 包含以下字段的字典
            'object': (N, 43+) 其中存储 frame 位姿(17:21四元数,21:24位置)、tool位姿(28:31位置,31:35四元数)、
                      frame_done(42), tool_done(43)
            'robot0_eef_pos': (N,3)
            'robot0_eef_quat': (N,4)
            'robot0_gripper_qpos': (N,2)
        - tool_pcd: 工具点云 (M1,3)
        - frame_pcd: 框架点云 (M2,3)
        - fin_rad: 手指半径
        - sim_thresh: [位置阈值, 角度阈值] 判断物体位姿相似
        - max_reward: 最大奖励值
        - reward_mode: 'only_success' 或 'tanh'
        - Tr: 前瞻步数
    return:
        - dict: {'subgoal': (N-Tr,8), 'next_subgoal': (N-Tr,8), 'reward': (N-Tr,)}
    """
    N = raw_obs['object'].shape[0]
    contact_thresh = fin_rad + 0.03
    goal_thresh = fin_rad / 2

    # 提取任务完成标志
    frame_done_arr = raw_obs['object'][:, 42] > 0.5
    tool_done_arr = raw_obs['object'][:, 43] > 0.5

    # ---------- 1. 初选：记录每个时间步的手指位置（物体坐标系）和物体子目标 ----------
    fin_subgoals_obj = []          # 长度 N，每个元素 [fl_x, fl_y, fl_z, fr_x, fr_y, fr_z, is_fl_contact, is_fr_contact]
    obj_subgoals = []              # 物体位姿子目标列表 [pos+quat]
    obj_subgoals_id = []           # 对应的时间步索引

    # 用于跟踪接触状态变化（按物体类型分开）
    last_object_type = None     # 'frame' or 'tool'
    last_fl_contact = False
    last_fr_contact = False

    for step in range(N):
        # 当前阶段的目标物体
        pcd = None
        if not frame_done_arr[step]:
            curr_type = 'frame'
            obj_pos = raw_obs['object'][step, 14:17]
            obj_qua = raw_obs['object'][step, 17:21]
            pcd = frame_pcd
        else:
            curr_type = 'tool'
            obj_pos = raw_obs['object'][step, 28:31]
            obj_qua = raw_obs['object'][step, 31:35]
            pcd = tool_pcd

        # 获取手指世界坐标
        eef_pos = raw_obs['robot0_eef_pos'][step]
        eef_quat = raw_obs['robot0_eef_quat'][step]
        gripper_q = raw_obs['robot0_gripper_qpos'][step]
        fl_world, fr_world = getFingersPos(eef_pos, eef_quat,
                                           gripper_q[0] + 0.0145/2,
                                           gripper_q[1] - 0.0145/2)

        # 转换到当前物体坐标系
        T_W_O = tf.PosQua_to_TransMat(obj_pos, obj_qua)
        T_O_W = np.linalg.inv(T_W_O)
        fl_obj = tf.transPt(fl_world, T_f2_f1=T_O_W)
        fr_obj = tf.transPt(fr_world, T_f2_f1=T_O_W)
        #show_pcd_finger(pcd, fl_obj, fr_obj)
        # 计算是否接触（距离点云最近点）
        fl_dists = pcd - fl_obj
        fr_dists = pcd - fr_obj
        fl_dist = np.min(np.linalg.norm(fl_dists, axis=1))
        fr_dist = np.min(np.linalg.norm(fr_dists, axis=1))
        is_fl_contact = fl_dist < contact_thresh
        is_fr_contact = fr_dist < contact_thresh
        is_contact = is_fl_contact and is_fr_contact
        is_fl_contact = is_contact
        is_fr_contact = is_contact

        # 保存手指位置（物体坐标系）
        fin_subgoals_obj.append(np.concatenate((fl_obj, fr_obj, [is_fl_contact], [is_fr_contact])))

        # 处理物体切换：重置接触状态，并记录新物体的初始位姿作为子目标
        if curr_type != last_object_type:
            if last_object_type is not None:
                # 可选：记录前一物体的最终位姿？为了连续性，记录当前新物体的初始位姿
                obj_subgoals.append(np.concatenate((obj_pos, obj_qua)))
                obj_subgoals_id.append(step)
            last_fl_contact = False
            last_fr_contact = False
            last_object_type = curr_type

        # 检测接触状态变化（同一物体内）
        if (is_fl_contact != last_fl_contact) and (is_fr_contact != last_fr_contact):
            obj_subgoals.append(np.concatenate((obj_pos, obj_qua)))
            obj_subgoals_id.append(step)

        last_fl_contact = is_fl_contact
        last_fr_contact = is_fr_contact

    # 确保 fin_subgoals_obj 长度为 N
    fin_subgoals_obj = np.array(fin_subgoals_obj)  # (N,8)

    # ---------- 2. 过滤：删除相邻的相似物体位姿，并清零对应时间区间的手指子目标 ----------
    i = 0
    while i < len(obj_subgoals) - 1:
        similar = check_poses_similarity(obj_subgoals[i], obj_subgoals[i+1],
                                          pos_th=sim_thresh[0], euler_th=sim_thresh[1])
        if similar:
            # 将两个子目标之间的手指位置清零（不包含后一个子目标的索引）
            start = obj_subgoals_id[i]
            end = obj_subgoals_id[i+1]
            fin_subgoals_obj[start:end] = 0.0
            # 删除前一个物体子目标
            obj_subgoals.pop(i)
            obj_subgoals_id.pop(i)
        else:
            i += 1

    # 无有效物体子目标时返回全零
    if len(obj_subgoals_id) == 0:
        zero_shape = (N - Tr, 8)
        return {
            'subgoal': np.zeros(zero_shape),
            'next_subgoal': np.zeros(zero_shape),
            'reward': np.zeros(N - Tr)
        }

    # ---------- 3. 配置：为每个时间步生成子目标（世界坐标系）和奖励 ----------
    fin_sgs = []      # 当前时刻子目标
    next_fin_sgs = [] # 下一时刻子目标
    rewards = []

    obj_sg_id = 0            # 当前已达到的物体子目标索引
    last_done = 'obj'        # 上次完成的是 'obj'（等待手指）还是 'fin'（等待物体）
    r = 0

    for step in range(N - Tr):
        # 获取当前 step 的物体位姿（用于转换手指子目标到世界坐标系）
        if not frame_done_arr[step]:
            obj_pos = raw_obs['object'][step, 14:17]
            obj_qua = raw_obs['object'][step, 17:21]
        else:
            obj_pos = raw_obs['object'][step, 28:31]
            obj_qua = raw_obs['object'][step, 31:35]

        # ------- 状态机：先完成手指子目标，再完成物体子目标 -------
        if last_done == 'obj':
            # 当奖励达到最大值（即手指已到达子目标）时，标记手指阶段完成
            if r == max_reward:
                last_done = 'fin'

        if last_done == 'fin' and obj_sg_id < len(obj_subgoals_id) - 1:
            # 检查物体是否到达下一个子目标（注意：比较的是下一个子目标）
            obj_pose = np.concatenate((obj_pos, obj_qua))
            next_obj_pose = obj_subgoals[obj_sg_id + 1]
            if check_poses_similarity(obj_pose, next_obj_pose,
                                      pos_th=sim_thresh[0], euler_th=sim_thresh[1]):
                obj_sg_id += 1
                last_done = 'obj'

        # ------- 确定当前应使用的手指子目标（物体坐标系下） -------
        # 取 max(物体子目标对应的起始步, 当前步) 确保不会使用未来的手指位置
        fin_sg_idx = max(obj_subgoals_id[obj_sg_id], step)
        fin_sg_obj = fin_subgoals_obj[fin_sg_idx]  # (8,)

        # 转换到世界坐标系（只对接触标志为真的手指进行变换）
        fl_sg_world = tf.transPt(fin_sg_obj[:3], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_sg_obj[6]
        fr_sg_world = tf.transPt(fin_sg_obj[3:6], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_sg_obj[7]
        cur_subgoal = np.concatenate((fl_sg_world, fr_sg_world, [fin_sg_obj[6]], [fin_sg_obj[7]]))
        fin_sgs.append(cur_subgoal)

        # 下一时刻子目标（使用 step+Tr 时的物体位姿）
        next_step = step + Tr
        if not frame_done_arr[next_step]:
            next_obj_pos = raw_obs['object'][next_step, 14:17]
            next_obj_qua = raw_obs['object'][next_step, 17:21]
        else:
            next_obj_pos = raw_obs['object'][next_step, 28:31]
            next_obj_qua = raw_obs['object'][next_step, 31:35]
        next_fl_sg = tf.transPt(fin_sg_obj[:3], t_f2_f1=next_obj_pos, q_f2_f1=next_obj_qua) * fin_sg_obj[6]
        next_fr_sg = tf.transPt(fin_sg_obj[3:6], t_f2_f1=next_obj_pos, q_f2_f1=next_obj_qua) * fin_sg_obj[7]
        next_subgoal = np.concatenate((next_fl_sg, next_fr_sg, [fin_sg_obj[6]], [fin_sg_obj[7]]))
        next_fin_sgs.append(next_subgoal)

        # ------- 计算奖励：Tr 步内若有一步手指到达子目标则给满奖励 -------
        r = 0
        for n in range(1, Tr + 1):
            future_step = step + n
            # 未来时刻的物体位姿
            if not frame_done_arr[future_step]:
                fut_obj_pos = raw_obs['object'][future_step, 14:17]
                fut_obj_qua = raw_obs['object'][future_step, 17:21]
            else:
                fut_obj_pos = raw_obs['object'][future_step, 28:31]
                fut_obj_qua = raw_obs['object'][future_step, 31:35]

            # 未来时刻手指世界坐标
            fut_eef_pos = raw_obs['robot0_eef_pos'][future_step]
            fut_eef_quat = raw_obs['robot0_eef_quat'][future_step]
            fut_gripper = raw_obs['robot0_gripper_qpos'][future_step]
            fut_fl_world, fut_fr_world = getFingersPos(
                fut_eef_pos, fut_eef_quat,
                fut_gripper[0] + 0.0145/2,
                fut_gripper[1] - 0.0145/2
            )

            # 子目标转换到世界坐标系（基于未来物体位姿）
            fut_fl_sg = tf.transPt(fin_sg_obj[:3], t_f2_f1=fut_obj_pos, q_f2_f1=fut_obj_qua) * fin_sg_obj[6]
            fut_fr_sg = tf.transPt(fin_sg_obj[3:6], t_f2_f1=fut_obj_pos, q_f2_f1=fut_obj_qua) * fin_sg_obj[7]

            dist_fl = np.linalg.norm(fut_fl_world - fut_fl_sg) * fin_sg_obj[6]
            dist_fr = np.linalg.norm(fut_fr_world - fut_fr_sg) * fin_sg_obj[7]
            if max(dist_fl, dist_fr) < goal_thresh:
                r = max_reward
                break
        else:
            # 未达到目标，根据模式计算奖励
            if reward_mode == 'only_success':
                r = 0
            elif reward_mode == 'tanh':
                # 使用当前步（step）的手指距离（也可使用未来第一步，与原函数保持一致）
                # 这里简化：使用 step 时刻的手指与子目标的距离（也可使用未来平均，原函数用 step+n 但 n=1）
                # 为与原函数类似，我们使用 step 时刻的手指位置计算距离
                fl_world_cur, fr_world_cur = getFingersPos(
                    raw_obs['robot0_eef_pos'][step],
                    raw_obs['robot0_eef_quat'][step],
                    raw_obs['robot0_gripper_qpos'][step, 0] + 0.0145/2,
                    raw_obs['robot0_gripper_qpos'][step, 1] - 0.0145/2
                )
                dist_fl_cur = np.linalg.norm(fl_world_cur - fl_sg_world) * fin_sg_obj[6]
                dist_fr_cur = np.linalg.norm(fr_world_cur - fr_sg_world) * fin_sg_obj[7]
                reward_weights = 3.0
                r_fl = -np.tanh(dist_fl_cur * reward_weights)
                r_fr = -np.tanh(dist_fr_cur * reward_weights)
                r = (r_fl + r_fr) / 3.0 * 2.0 + 1.0
            else:
                raise ValueError("reward_mode must be 'only_success' or 'tanh'")

        rewards.append(r)

    return {
        'subgoal': np.array(fin_sgs),          # (N-Tr, 8)
        'next_subgoal': np.array(next_fin_sgs),# (N-Tr, 8)
        'reward': np.array(rewards)            # (N-Tr,)
    }


def get_subgoals_assembly_contact_keyframe(
        raw_obs: dict,
        tool_pcd: np.ndarray,
        frame_pcd: np.ndarray,
        fin_rad: float,
        sim_thresh: list,
        max_reward=10,
        reward_mode='tanh',
        Tr=1
):
    """
    ToolHang keyframe subgoal extractor.

    接口和 get_subgoals_assembly_contact_new 完全一致，但 subgoal 语义不同：
    1. 先检测手指是否真实接触当前阶段目标物体。
    2. 只保留“接触期间物体位姿相对上一关键点发生明显变化”的关键帧。
    3. 关键帧手指点从物体局部坐标转换成世界坐标后固定。
    4. 两个关键帧之间保持同一个世界坐标 subgoal，避免 subgoal 退化为连续轨迹。

    return:
        {'subgoal': (N-Tr,8), 'next_subgoal': (N-Tr,8), 'reward': (N-Tr,)}
    """
    N = raw_obs['object'].shape[0]
    if N <= Tr:
        zero_shape = (max(N - Tr, 0), 8)
        return {
            'subgoal': np.zeros(zero_shape),
            'next_subgoal': np.zeros(zero_shape),
            'reward': np.zeros(zero_shape[0])
        }

    frame_done_arr = raw_obs['object'][:, 42] > 0.5

    # ToolHang 的 frame 插入 stand 更精细，原 fin_rad + 0.03 太松。
    frame_contact_thresh = fin_rad + 0.012
    tool_contact_thresh = fin_rad + 0.015
    goal_thresh = max(fin_rad + 0.005, 0.012)

    def _target_type(step):
        return 'frame' if not frame_done_arr[step] else 'tool'

    def _target_pose(step, target_type):
        # robomimic ToolHang: frame world pose 是 14:17 / 17:21，不是 21:24。
        if target_type == 'frame':
            return raw_obs['object'][step, 14:17], raw_obs['object'][step, 17:21]
        return raw_obs['object'][step, 28:31], raw_obs['object'][step, 31:35]

    def _target_pcd(target_type):
        return frame_pcd if target_type == 'frame' else tool_pcd

    def _contact_thresh(target_type):
        return frame_contact_thresh if target_type == 'frame' else tool_contact_thresh

    def _finger_world(step):
        eef_pos = raw_obs['robot0_eef_pos'][step]
        eef_quat = raw_obs['robot0_eef_quat'][step]
        gripper_q = raw_obs['robot0_gripper_qpos'][step]
        return getFingersPos(
            eef_pos,
            eef_quat,
            gripper_q[0] + 0.0145 / 2,
            gripper_q[1] - 0.0145 / 2
        )

    def _pose_vec(step, target_type):
        obj_pos, obj_qua = _target_pose(step, target_type)
        return np.concatenate((obj_pos, obj_qua))

    # ---------- 1. 每帧检测接触，并记录手指在目标物体局部坐标系下的位置 ----------
    fin_subgoals_obj = np.zeros((N, 8), dtype=np.float32)
    contact_any = np.zeros(N, dtype=bool)
    target_types = []

    for step in range(N):
        target_type = _target_type(step)
        target_types.append(target_type)
        obj_pos, obj_qua = _target_pose(step, target_type)
        pcd = _target_pcd(target_type)

        fl_world, fr_world = _finger_world(step)
        T_W_O = tf.PosQua_to_TransMat(obj_pos, obj_qua)
        T_O_W = np.linalg.inv(T_W_O)
        fl_obj = tf.transPt(fl_world, T_f2_f1=T_O_W)
        fr_obj = tf.transPt(fr_world, T_f2_f1=T_O_W)

        fl_dist = np.min(np.linalg.norm(pcd - fl_obj, axis=1))
        fr_dist = np.min(np.linalg.norm(pcd - fr_obj, axis=1))
        thresh = _contact_thresh(target_type)

        # 保留左右手指独立接触标志；不强制双指同时接触。
        is_fl_contact = fl_dist < thresh
        is_fr_contact = fr_dist < thresh
        fin_subgoals_obj[step] = np.concatenate(
            (fl_obj, fr_obj, [float(is_fl_contact)], [float(is_fr_contact)])
        )
        contact_any[step] = is_fl_contact or is_fr_contact

    # ---------- 2. 提取导致目标物体位姿明显变化的关键接触帧 ----------
    keyframes = []
    last_key_pose = {'frame': None, 'tool': None}
    last_key_step = {'frame': -10**9, 'tool': -10**9}

    min_key_gap = 8

    for step in range(N):
        if not contact_any[step]:
            continue
        target_type = target_types[step]
        pose = _pose_vec(step, target_type)

        is_new_phase = last_key_pose[target_type] is None
        is_far_enough = step - last_key_step[target_type] >= min_key_gap
        pose_changed = (
            is_new_phase or
            not check_poses_similarity(
                last_key_pose[target_type],
                pose,
                pos_th=sim_thresh[0],
                euler_th=sim_thresh[1]
            )
        )

        if is_new_phase or (is_far_enough and pose_changed):
            obj_pos, obj_qua = _target_pose(step, target_type)
            fin_obj = fin_subgoals_obj[step]
            fl_world = tf.transPt(fin_obj[:3], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_obj[6]
            fr_world = tf.transPt(fin_obj[3:6], t_f2_f1=obj_pos, q_f2_f1=obj_qua) * fin_obj[7]
            keyframes.append({
                'step': step,
                'type': target_type,
                'pose': pose,
                'subgoal_world': np.concatenate((fl_world, fr_world, [fin_obj[6]], [fin_obj[7]])).astype(np.float32)
            })
            last_key_pose[target_type] = pose
            last_key_step[target_type] = step

    if len(keyframes) == 0:
        zero_shape = (N - Tr, 8)
        return {
            'subgoal': np.zeros(zero_shape),
            'next_subgoal': np.zeros(zero_shape),
            'reward': np.zeros(N - Tr)
        }

    # ---------- 3. 为每个时间步分配最近未来关键帧；没有未来关键帧则使用最近过去关键帧 ----------
    key_steps = np.array([k['step'] for k in keyframes], dtype=np.int64)
    fin_sgs = []
    next_fin_sgs = []
    rewards = []

    for step in range(N - Tr):
        future_ids = np.where(key_steps >= step)[0]
        if len(future_ids) > 0:
            key_id = future_ids[0]
        else:
            key_id = len(keyframes) - 1

        subgoal = keyframes[key_id]['subgoal_world']
        fin_sgs.append(subgoal)
        next_fin_sgs.append(subgoal.copy())

        r = 0
        active_dists_now = []
        for n in range(1, Tr + 1):
            future_step = step + n
            fut_fl_world, fut_fr_world = _finger_world(future_step)
            active_dists = []
            if subgoal[6] > 0.5:
                active_dists.append(np.linalg.norm(fut_fl_world - subgoal[:3]))
            if subgoal[7] > 0.5:
                active_dists.append(np.linalg.norm(fut_fr_world - subgoal[3:6]))
            if len(active_dists) > 0 and max(active_dists) < goal_thresh:
                r = max_reward
                break
        else:
            if reward_mode == 'only_success':
                r = 0
            elif reward_mode == 'tanh':
                fl_world, fr_world = _finger_world(step)
                if subgoal[6] > 0.5:
                    active_dists_now.append(np.linalg.norm(fl_world - subgoal[:3]))
                if subgoal[7] > 0.5:
                    active_dists_now.append(np.linalg.norm(fr_world - subgoal[3:6]))
                if len(active_dists_now) == 0:
                    r = 0
                else:
                    reward_weights = 3.0
                    r = 1.0 - np.tanh(np.mean(active_dists_now) * reward_weights)
            else:
                raise ValueError("reward_mode must be 'only_success' or 'tanh'")

        rewards.append(r)

    return {
        'subgoal': np.asarray(fin_sgs, dtype=np.float32),
        'next_subgoal': np.asarray(next_fin_sgs, dtype=np.float32),
        'reward': np.asarray(rewards, dtype=np.float32)
    }

import numpy as np
from typing import Dict, List
from scipy.spatial.transform import Rotation as R


def get_subgoals_from_gripper(
    state_ee: np.ndarray,          # (T, 7) 末端位姿 [x, y, z, qx, qy, qz, qw]
    gripper: np.ndarray,           # (T, 1) 夹爪开度，通常归一化到 [0, 1]，1 表示张开，0 表示闭合
    Tr: int = 1,                   # 奖励计算时向未来看的步数

    # ---------- 夹爪事件检测参数 ----------
    open_th: float = 0.8,          # 判定为“张开”的夹爪开度阈值
    close_th: float = 0.35,        # 判定为“闭合”的夹爪开度阈值
    grip_change_th: float = 0.03,  # 夹爪开度变化阈值，用于检测开始闭合/开始张开事件
    min_event_gap: int = 5,        # 两个关键点之间的最小时间间隔，避免关键点过密

    # ---------- 轨迹关键点参数 ----------
    speed_low_quantile: float = 0.25,   # 低速点分位数，用于找停顿/接触/调整点
    curvature_quantile: float = 0.85,   # 高曲率点分位数，用于找转向/轨迹变化点
    max_keypoints: int = 12,            # 最多保留多少个关键点

    # ---------- 夹爪几何参数 ----------
    max_gripper_width: float = 0.085,   # 夹爪最大宽度，例如 Robotiq 2F-85 是 85mm
    tcp_to_gripper_z: float = 0.20,     # TCP 到夹爪指尖坐标系的 z 方向偏移，需按真实机器人标定

    # ---------- 奖励参数 ----------
    goal_dist_thresh: float = 0.01,     # 到达子目标的距离阈值，单位 m
    max_reward: float = 10.0,           # 成功到达子目标时的最大奖励
    reward_mode: str = "tanh",          # "only_success" 或 "tanh"
    reward_scale: float = 80.0,         # dense reward 距离缩放系数
) -> Dict[str, np.ndarray]:
    """
    根据机械臂末端轨迹和夹爪开度，提取抓取任务中的关键子目标。

    主要思想：
        1. 根据夹爪开度变化，提取语义关键点：
           - 开始闭合 close_start
           - 完全闭合/抓稳 grasp
           - 抓取后抬升 lift
           - 开始张开 open_start
           - 释放 release

        2. 根据末端运动轨迹，补充运动关键点：
           - 低速点：可能对应接触、停顿、精细调整
           - 高曲率点：可能对应转向、避障、放置姿态调整

        3. 将每个关键点转换成左右指尖位置，作为模仿学习的 subgoal。

    返回：
        subgoal:
            shape = (T - Tr, 6)
            当前时刻应该追踪的左右指尖目标位置：
            [left_x, left_y, left_z, right_x, right_y, right_z]

        next_subgoal:
            shape = (T - Tr, 6)
            当前子目标之后的下一个子目标。

        reward:
            shape = (T - Tr,)
            未来 Tr 步内是否接近当前子目标的奖励。
    """

    # -----------------------------
    # 0. 输入检查
    # -----------------------------
    assert state_ee.ndim == 2 and state_ee.shape[1] == 7, \
        "state_ee should have shape (T, 7)"

    T = state_ee.shape[0]

    # 将 gripper 转成一维数组，方便后续处理
    grip = gripper.reshape(-1).astype(np.float32)

    assert len(grip) == T, \
        "gripper length should be the same as state_ee length"

    # 如果轨迹太短，无法提取有效关键点，直接返回零
    if T <= Tr + 2:
        valid_len = max(T - Tr, 0)
        return {
            "subgoal": np.zeros((valid_len, 6), dtype=np.float32),
            "next_subgoal": np.zeros((valid_len, 6), dtype=np.float32),
            "reward": np.zeros((valid_len,), dtype=np.float32),
        }

    # -----------------------------
    # 1. 根据末端位姿和夹爪开度，计算左右指尖世界坐标
    # -----------------------------
    def compute_fingers(
        ee_pos_quat: np.ndarray,
        grip_val: np.ndarray,
    ):
        """
        将末端位姿 + 夹爪开度转换为左右指尖的世界坐标。

        输入：
            ee_pos_quat:
                (T, 7)，每帧末端位姿 [x, y, z, qx, qy, qz, qw]

            grip_val:
                (T,)，夹爪开度，通常 1 为张开，0 为闭合

        输出：
            left:
                (T, 3)，左指尖世界坐标

            right:
                (T, 3)，右指尖世界坐标
        """

        left = np.zeros((T, 3), dtype=np.float32)
        right = np.zeros((T, 3), dtype=np.float32)

        for t in range(T):
            # 末端位置
            pos = ee_pos_quat[t, :3]

            # 四元数，注意 scipy 使用 [qx, qy, qz, qw]
            qx, qy, qz, qw = ee_pos_quat[t, 3:]

            # 四元数转旋转矩阵
            rot = R.from_quat([qx, qy, qz, qw]).as_matrix()

            # 夹爪半宽
            # grip=1 表示最大张开，grip=0 表示闭合
            half_w = 0.5 * max_gripper_width * np.clip(grip_val[t], 0.0, 1.0)

            # 在夹爪局部坐标系下，左右指尖坐标
            left_local = np.array([-half_w, 0.0, 0.0, 1.0], dtype=np.float32)
            right_local = np.array([half_w, 0.0, 0.0, 1.0], dtype=np.float32)

            # 构造 TCP 到世界坐标系的齐次变换
            T_tool = np.eye(4, dtype=np.float32)
            T_tool[:3, :3] = rot
            T_tool[:3, 3] = pos

            # TCP 到夹爪中心的偏移
            # 这里默认沿 tool z 方向偏移 tcp_to_gripper_z
            # 真实机器人中建议用标定值替换
            T_gripper = np.eye(4, dtype=np.float32)
            T_gripper[2, 3] = tcp_to_gripper_z

            # 世界坐标系下的夹爪坐标变换
            Twg = T_tool @ T_gripper

            # 计算左右指尖世界坐标
            left[t] = (Twg @ left_local)[:3]
            right[t] = (Twg @ right_local)[:3]

        return left, right

    left_fingers, right_fingers = compute_fingers(state_ee, grip)

    # 每一帧的完整指尖状态，6 维：
    # [left_x, left_y, left_z, right_x, right_y, right_z]
    finger_state = np.concatenate([left_fingers, right_fingers], axis=1)

    # 末端位置轨迹
    pos = state_ee[:, :3]

    # -----------------------------
    # 2. 提取夹爪语义关键点
    # -----------------------------

    # 夹爪开度的一阶差分
    # dg < 0 表示夹爪在闭合
    # dg > 0 表示夹爪在张开
    dg = np.diff(grip, prepend=grip[0])

    # 找到明显开始闭合的帧
    closing_candidates = np.where(dg < -grip_change_th)[0]

    # 找到明显开始张开的帧
    opening_candidates = np.where(dg > grip_change_th)[0]

    # 判断每一帧是否处于闭合/张开状态
    closed = grip <= close_th
    opened = grip >= open_th

    # 初始化关键点列表
    # 起点和终点通常都应该保留
    key_ids: List[int] = [0, T - 1]

    def add_event_with_context(idx: int):
        """
        对一个事件点，额外加入事件前后若干帧。

        原因：
            在抓取模仿学习中，事件本身很重要，
            但事件发生前的接近姿态、事件后的稳定姿态也很重要。

        例如：
            idx - 3：pre-grasp / pre-release
            idx：事件帧
            idx + 3：post-grasp / post-release
        """
        for k in [idx - 3, idx, idx + 3]:
            if 0 <= k < T:
                key_ids.append(int(k))

    # ---------- 2.1 闭合/抓取阶段 ----------
    if len(closing_candidates) > 0:
        # 第一次明显闭合，通常对应开始抓取
        close_start = int(closing_candidates[0])
        add_event_with_context(close_start)

        # 从 close_start 之后，找第一次真正闭合的帧
        closed_after = np.where(closed & (np.arange(T) >= close_start))[0]

        if len(closed_after) > 0:
            # grasp_idx 可理解为夹爪闭合稳定或抓住物体的时间
            grasp_idx = int(closed_after[0])
            add_event_with_context(grasp_idx)

            # 抓取后通常会有一个抬升阶段
            # 在 grasp_idx 后的一段窗口中，z 最大的点作为 lift keypoint
            search_end = min(T, grasp_idx + max(10, T // 3))

            if search_end > grasp_idx + 1:
                lift_idx = grasp_idx + int(np.argmax(pos[grasp_idx:search_end, 2]))
                key_ids.append(lift_idx)

    # ---------- 2.2 张开/释放阶段 ----------
    if len(opening_candidates) > 0:
        # 通常最后一次明显张开对应释放物体
        open_start = int(opening_candidates[-1])
        add_event_with_context(open_start)

        # 从 open_start 之后，找第一次真正张开的帧
        opened_after = np.where(opened & (np.arange(T) >= open_start))[0]

        if len(opened_after) > 0:
            release_idx = int(opened_after[0])
            add_event_with_context(release_idx)

    # -----------------------------
    # 3. 提取运动学关键点
    # -----------------------------

    # 速度：相邻帧末端位置变化量
    vel = np.linalg.norm(
        np.diff(pos, axis=0, prepend=pos[:1]),
        axis=1,
    )

    # 加速度/曲率近似：二阶差分
    # 这里不是严格几何曲率，但能很好地捕捉轨迹突然变化的位置
    acc = np.linalg.norm(
        np.diff(pos, n=2, axis=0, prepend=pos[:1], append=pos[-1:]),
        axis=1,
    )

    # 低速阈值
    # 低速点往往对应：
    #   - 接近物体时的精细调整
    #   - 接触
    #   - 抓取前停顿
    #   - 放置前停顿
    speed_th = np.quantile(vel, speed_low_quantile)
    pause_candidates = np.where(vel <= speed_th)[0]

    # 高曲率/高加速度阈值
    # 这些点往往对应轨迹方向明显改变
    curvature_th = np.quantile(acc, curvature_quantile)
    turn_candidates = np.where(acc >= curvature_th)[0]

    def sparse_add(candidates: np.ndarray, limit: int):
        """
        从候选关键点中稀疏地加入若干个点。

        这样做的原因：
            轨迹中可能连续很多帧都满足低速或高曲率条件，
            如果全部加入，会导致关键点过密，训练目标不稳定。
        """
        last = -10**9
        count = 0

        for idx in candidates:
            idx = int(idx)

            # 与上一个加入的点保持最小间隔
            if idx - last >= min_event_gap:
                key_ids.append(idx)
                last = idx
                count += 1

                if count >= limit:
                    break

    # 加入最多 4 个停顿点
    sparse_add(pause_candidates, limit=4)

    # 加入最多 4 个转向点
    sparse_add(turn_candidates, limit=4)

    # -----------------------------
    # 4. 清理关键点
    # -----------------------------

    # 去重、裁剪到合法范围、排序
    key_ids = sorted(set(int(np.clip(k, 0, T - 1)) for k in key_ids))

    # 再做一次时间间隔过滤，避免关键点太密
    filtered = []

    for k in key_ids:
        if not filtered:
            filtered.append(k)
            continue

        # 如果和上一个关键点距离足够远，则保留
        if k - filtered[-1] >= min_event_gap:
            filtered.append(k)
        else:
            # 如果距离太近，则保留夹爪变化更明显的那个
            old = filtered[-1]

            if abs(dg[k]) > abs(dg[old]):
                filtered[-1] = k

    key_ids = filtered

    # 如果关键点数量超过上限，则均匀采样中间关键点
    # 起点和终点强制保留
    if len(key_ids) > max_keypoints:
        middle = key_ids[1:-1]
        keep_n = max_keypoints - 2

        if keep_n > 0:
            sampled = np.linspace(
                0,
                len(middle) - 1,
                keep_n,
                dtype=int,
            )
            key_ids = [key_ids[0]] + [middle[i] for i in sampled] + [key_ids[-1]]
        else:
            key_ids = [key_ids[0], key_ids[-1]]

    key_ids = sorted(set(key_ids))

    # 若没有关键点，返回零
    if len(key_ids) == 0:
        return {
            "subgoal": np.zeros((T - Tr, 6), dtype=np.float32),
            "next_subgoal": np.zeros((T - Tr, 6), dtype=np.float32),
            "reward": np.zeros((T - Tr,), dtype=np.float32),
        }

    # 将关键帧转换为指尖子目标
    # shape = (K, 6)
    subgoal_fingers = finger_state[key_ids].astype(np.float32)

    # -----------------------------
    # 5. 给每一帧分配当前 subgoal 和 next_subgoal
    # -----------------------------

    fin_sgs = []
    next_fin_sgs = []
    rewards = []

    # 当前正在追踪的子目标编号
    cur_sg = 0

    for t in range(T - Tr):
        curr_fg = finger_state[t]

        # 如果已经接近当前子目标，或者时间已经超过当前关键帧，
        # 则切换到下一个子目标
        while cur_sg < len(key_ids) - 1:
            dist_to_cur = np.linalg.norm(curr_fg - subgoal_fingers[cur_sg])

            if dist_to_cur < goal_dist_thresh or t >= key_ids[cur_sg]:
                cur_sg += 1
            else:
                break

        # 下一个子目标编号
        next_sg = min(cur_sg + 1, len(subgoal_fingers) - 1)

        # 当前目标与下一个目标
        target = subgoal_fingers[cur_sg]
        next_target = subgoal_fingers[next_sg]

        fin_sgs.append(target)
        next_fin_sgs.append(next_target)

        # -----------------------------
        # 6. 计算奖励
        # -----------------------------
        # 看未来 Tr 步内是否接近当前子目标
        future = finger_state[t + 1:t + Tr + 1]

        # 未来每一帧到当前子目标的距离
        dists = np.linalg.norm(future - target[None, :], axis=1)

        # 未来窗口内的最小距离
        min_dist = float(np.min(dists))

        if min_dist < goal_dist_thresh:
            # 成功到达子目标
            r = max_reward
        elif reward_mode == "only_success":
            # 稀疏奖励：没到达就是 0
            r = 0.0
        else:
            # dense reward：
            # 距离越近，奖励越接近 1
            # 距离越远，奖励越接近 0
            r = 1.0 - np.tanh(min_dist * reward_scale)

        rewards.append(r)

    return {
        "subgoal": np.asarray(fin_sgs, dtype=np.float32),
        "next_subgoal": np.asarray(next_fin_sgs, dtype=np.float32),
        "reward": np.asarray(rewards, dtype=np.float32),
    }


def get_ring_insertion_event_subgoals(
    state_ee: np.ndarray,
    gripper: np.ndarray,
    Tr: int = 8,
    open_th: float = 0.8,
    close_th: float = 0.35,
    open_change_th: float = 0.03,
    close_change_th: float = 0.03,
    pre_grasp_offset: int = 6,
    min_event_gap: int = 5,
    max_gripper_width: float = 0.085,
    tcp_to_gripper_z: float = 0.20,
    above_search_ratio: float = 0.75,
    insert_drop_ratio: float = 0.65,
    include_return_closed: bool = True,
    goal_dist_thresh: float = 0.015,
    max_reward: float = 10.0,
    reward_mode: str = "tanh",
    reward_scale: float = 80.0,
    return_debug: bool = False,
):
    """
    Event-based subgoal extractor for the handled-ring-on-cylinder task.

    It intentionally avoids generic low-speed / high-curvature waypoints. The
    subgoals are derived from the task events:
        pre_grasp -> grasp_close -> above_cylinder -> insert_down -> release
        -> return_closed, when include_return_closed=True

    Inputs:
        state_ee: (T, 7), [x, y, z, qx, qy, qz, qw]
        gripper: (T, 1) or (T,), normalized opening, 1=open, 0=closed

    Returns compatible with the current 6D subgoal setup:
        subgoal: (T - Tr, 6), [left_xyz, right_xyz]
        next_subgoal: (T - Tr, 6)
        reward: (T - Tr,)

    If return_debug=True, also returns:
        key_ids: (K,), selected frame ids
        key_names: list[str]
        stage_id: (T - Tr,), current stage target id
    """
    assert state_ee.ndim == 2 and state_ee.shape[1] == 7, \
        "state_ee should have shape (T, 7)"

    T = state_ee.shape[0]
    grip = gripper.reshape(-1).astype(np.float32)
    assert grip.shape[0] == T, "gripper length should match state_ee length"

    valid_len = max(T - Tr, 0)
    if T <= Tr + 2:
        out = {
            "subgoal": np.zeros((valid_len, 6), dtype=np.float32),
            "next_subgoal": np.zeros((valid_len, 6), dtype=np.float32),
            "reward": np.zeros((valid_len,), dtype=np.float32),
        }
        if return_debug:
            out.update({
                "key_ids": np.zeros((0,), dtype=np.int64),
                "key_names": [],
                "stage_id": np.zeros((valid_len,), dtype=np.int64),
            })
        return out

    def compute_fingers(ee_pos_quat, grip_val):
        left = np.zeros((T, 3), dtype=np.float32)
        right = np.zeros((T, 3), dtype=np.float32)
        for t in range(T):
            pos = ee_pos_quat[t, :3]
            quat = ee_pos_quat[t, 3:7]
            quat_norm = np.linalg.norm(quat)
            if quat_norm < 1e-8 or not np.isfinite(quat_norm):
                quat = np.array([0, 0, 0, 1], dtype=np.float32)
            else:
                quat = quat / quat_norm

            rot = R.from_quat(quat).as_matrix()
            half_w = 0.5 * max_gripper_width * np.clip(grip_val[t], 0.0, 1.0)
            left_local = np.array([-half_w, 0.0, 0.0, 1.0], dtype=np.float32)
            right_local = np.array([half_w, 0.0, 0.0, 1.0], dtype=np.float32)

            T_tool = np.eye(4, dtype=np.float32)
            T_tool[:3, :3] = rot
            T_tool[:3, 3] = pos
            T_gripper = np.eye(4, dtype=np.float32)
            T_gripper[2, 3] = tcp_to_gripper_z
            T_world_gripper = T_tool @ T_gripper

            left[t] = (T_world_gripper @ left_local)[:3]
            right[t] = (T_world_gripper @ right_local)[:3]
        return np.concatenate([left, right], axis=1)

    finger_state = compute_fingers(state_ee, grip)
    pos = state_ee[:, :3]
    dg = np.diff(grip, prepend=grip[0])

    def first_after(mask, start):
        ids = np.where(mask & (np.arange(T) >= start))[0]
        return int(ids[0]) if len(ids) > 0 else None

    def last_after(mask, start):
        ids = np.where(mask & (np.arange(T) >= start))[0]
        return int(ids[-1]) if len(ids) > 0 else None

    opened = grip >= open_th
    closed = grip <= close_th

    # K0: initial opening event. It is useful for debug but usually not used as
    # a target because it belongs to the reset/approach transition.
    open_start = first_after(dg > open_change_th, 1)
    if open_start is None:
        open_start = first_after(opened, 0)
    if open_start is None:
        open_start = 0

    # K2: close event at the handle.
    close_candidates = np.where((dg < -close_change_th) & (np.arange(T) >= open_start + min_event_gap))[0]
    close_start = int(close_candidates[0]) if len(close_candidates) > 0 else None
    if close_start is None:
        close_start = first_after(closed, open_start + min_event_gap)
    if close_start is None:
        close_start = min(T - 1, open_start + max(min_event_gap, pre_grasp_offset))

    # K1: pre-grasp immediately before close, while the gripper is still open.
    pre_grasp = max(open_start, close_start - pre_grasp_offset)
    open_before_close = np.where(opened & (np.arange(T) < close_start))[0]
    if len(open_before_close) > 0:
        pre_grasp = int(open_before_close[np.argmin(np.abs(open_before_close - pre_grasp))])

    grasp_close = first_after(closed, close_start)
    if grasp_close is None:
        grasp_close = close_start

    # K5: release after insertion. Use the last opening after grasp to avoid
    # picking the initial open_approach event.
    release_start = last_after(dg > open_change_th, grasp_close + min_event_gap)
    if release_start is None:
        release_start = first_after(opened, grasp_close + min_event_gap)
    if release_start is None:
        release_start = T - 1

    return_closed = None
    if include_return_closed:
        return_closed = first_after(closed, release_start + min_event_gap)
        if return_closed is None:
            close_after_release = np.where((dg < -close_change_th) & (np.arange(T) >= release_start + min_event_gap))[0]
            if len(close_after_release) > 0:
                return_closed = int(close_after_release[0])
        if return_closed is None and release_start < T - 1:
            tail_ids = np.arange(release_start, T)
            tail_score = np.abs(grip[tail_ids] - close_th)
            return_closed = int(tail_ids[np.argmin(tail_score)])
        if return_closed is None:
            return_closed = T - 1

    # K3/K4 are between grasp and release. Without ring/cylinder pose, infer
    # them from the EE trajectory: above_cylinder is a high-z pause/alignment
    # candidate, insert_down is the lower-z point after that and before release.
    search_start = int(np.clip(grasp_close + min_event_gap, 0, T - 1))
    search_end = int(np.clip(max(search_start + 1, release_start), 0, T))
    mid_ids = np.arange(search_start, search_end)

    if len(mid_ids) > 0:
        vel = np.linalg.norm(np.diff(pos, axis=0, prepend=pos[:1]), axis=1)
        z = pos[mid_ids, 2]
        high_z_th = np.quantile(z, above_search_ratio)
        high_ids = mid_ids[z >= high_z_th]
        if len(high_ids) > 0:
            above_cylinder = int(high_ids[np.argmin(vel[high_ids])])
        else:
            above_cylinder = int(mid_ids[np.argmax(z)])
    else:
        above_cylinder = grasp_close

    insert_start = int(np.clip(above_cylinder + 1, 0, T - 1))
    insert_end = int(np.clip(max(above_cylinder + 2, release_start), 0, T))
    insert_ids = np.arange(insert_start, insert_end)
    if len(insert_ids) > 0:
        z_after = pos[insert_ids, 2]
        low_z_th = np.quantile(z_after, 1.0 - insert_drop_ratio)
        low_ids = insert_ids[z_after <= low_z_th]
        insert_down = int(low_ids[-1]) if len(low_ids) > 0 else int(insert_ids[np.argmin(z_after)])
    else:
        insert_down = above_cylinder

    raw_keys = [
        ("pre_grasp", pre_grasp),
        ("grasp_close", grasp_close),
        ("above_cylinder", above_cylinder),
        ("insert_down", insert_down),
        ("release", release_start),
    ]
    if include_return_closed:
        raw_keys.append(("return_closed", return_closed))

    # Enforce monotonic ids while preserving stage names. If two events collapse
    # to the same frame, keep one target and avoid zero-length stages.
    key_names = []
    key_ids = []
    last_id = -1
    for name, idx in raw_keys:
        idx = int(np.clip(idx, 0, T - 1))
        if idx <= last_id:
            idx = min(T - 1, last_id + 1)
        if idx > last_id:
            key_names.append(name)
            key_ids.append(idx)
            last_id = idx

    if len(key_ids) == 0:
        key_names = ["fallback"]
        key_ids = [min(T - 1, Tr)]

    key_ids = np.asarray(key_ids, dtype=np.int64)
    key_targets = finger_state[key_ids].astype(np.float32)

    fin_sgs = []
    next_fin_sgs = []
    rewards = []
    stage_ids = []

    cur_stage = 0
    for t in range(valid_len):
        while cur_stage < len(key_ids) - 1 and t >= key_ids[cur_stage]:
            cur_stage += 1

        next_stage = min(cur_stage + 1, len(key_ids) - 1)
        target = key_targets[cur_stage]
        next_target = key_targets[next_stage]
        fin_sgs.append(target)
        next_fin_sgs.append(next_target)
        stage_ids.append(cur_stage)

        future = finger_state[t + 1:t + Tr + 1]
        dists = np.linalg.norm(future - target[None, :], axis=1)
        min_dist = float(np.min(dists)) if len(dists) > 0 else np.inf
        if min_dist < goal_dist_thresh:
            r = max_reward
        elif reward_mode == "only_success":
            r = 0.0
        else:
            r = 1.0 - np.tanh(min_dist * reward_scale)
        rewards.append(r)

    out = {
        "subgoal": np.asarray(fin_sgs, dtype=np.float32),
        "next_subgoal": np.asarray(next_fin_sgs, dtype=np.float32),
        "reward": np.asarray(rewards, dtype=np.float32),
    }
    if return_debug:
        out.update({
            "key_ids": key_ids,
            "key_names": key_names,
            "stage_id": np.asarray(stage_ids, dtype=np.int64),
        })
    return out


def get_block_stack_event_subgoals(
    state_ee: np.ndarray,
    gripper: np.ndarray,
    Tr: int = 8,
    open_th: float = 0.8,
    close_th: float = 0.35,
    open_change_th: float = 0.005,
    close_change_th: float = 0.005,
    pre_grasp_offset: int = 6,
    pre_place_offset: int = 4,
    min_event_gap: int = 5,
    max_gripper_width: float = 0.085,
    tcp_to_gripper_z: float = 0.20,
    lift_search_ratio: float = 0.75,
    place_drop_ratio: float = 0.65,
    include_return_closed: bool = False,
    goal_dist_thresh: float = 0.015,
    max_reward: float = 10.0,
    reward_mode: str = "tanh",
    reward_scale: float = 80.0,
    return_debug: bool = False,
):
    """
    Event-based subgoal extractor for block stacking.

    Stages:
        pre_grasp -> grasp_close -> lift_above -> pre_place -> place_down
        -> release_start -> place_open
        -> return_closed, when include_return_closed=True.

    The output is compatible with the current 6D fingertip subgoal format:
        [left_finger_xyz, right_finger_xyz].
    """
    assert state_ee.ndim == 2 and state_ee.shape[1] == 7, \
        "state_ee should have shape (T, 7)"

    T = state_ee.shape[0]
    grip = gripper.reshape(-1).astype(np.float32)
    assert grip.shape[0] == T, "gripper length should match state_ee length"

    valid_len = max(T - Tr, 0)
    if T <= Tr + 2:
        out = {
            "subgoal": np.zeros((valid_len, 6), dtype=np.float32),
            "next_subgoal": np.zeros((valid_len, 6), dtype=np.float32),
            "reward": np.zeros((valid_len,), dtype=np.float32),
        }
        if return_debug:
            out.update({
                "key_ids": np.zeros((0,), dtype=np.int64),
                "key_names": [],
                "stage_id": np.zeros((valid_len,), dtype=np.int64),
            })
        return out

    def compute_fingers(ee_pos_quat, grip_val):
        left = np.zeros((T, 3), dtype=np.float32)
        right = np.zeros((T, 3), dtype=np.float32)
        for t in range(T):
            pos = ee_pos_quat[t, :3]
            quat = ee_pos_quat[t, 3:7]
            quat_norm = np.linalg.norm(quat)
            if quat_norm < 1e-8 or not np.isfinite(quat_norm):
                quat = np.array([0, 0, 0, 1], dtype=np.float32)
            else:
                quat = quat / quat_norm

            rot = R.from_quat(quat).as_matrix()
            half_w = 0.5 * max_gripper_width * np.clip(grip_val[t], 0.0, 1.0)
            left_local = np.array([-half_w, 0.0, 0.0, 1.0], dtype=np.float32)
            right_local = np.array([half_w, 0.0, 0.0, 1.0], dtype=np.float32)

            T_tool = np.eye(4, dtype=np.float32)
            T_tool[:3, :3] = rot
            T_tool[:3, 3] = pos
            T_gripper = np.eye(4, dtype=np.float32)
            T_gripper[2, 3] = tcp_to_gripper_z
            T_world_gripper = T_tool @ T_gripper

            left[t] = (T_world_gripper @ left_local)[:3]
            right[t] = (T_world_gripper @ right_local)[:3]
        return np.concatenate([left, right], axis=1)

    finger_state = compute_fingers(state_ee, grip)
    pos = state_ee[:, :3]
    dg = np.diff(grip, prepend=grip[0])
    frame_ids = np.arange(T)

    def first_after(mask, start):
        ids = np.where(mask & (frame_ids >= start))[0]
        return int(ids[0]) if len(ids) > 0 else None

    def last_after(mask, start):
        ids = np.where(mask & (frame_ids >= start))[0]
        return int(ids[-1]) if len(ids) > 0 else None

    opened = grip >= open_th
    closed = grip <= close_th

    open_start = first_after(dg > open_change_th, 1)
    if open_start is None:
        open_start = first_after(opened, 0)
    if open_start is None:
        open_start = 0

    close_candidates = np.where((dg < -close_change_th) & (frame_ids >= open_start + min_event_gap))[0]
    close_start = int(close_candidates[0]) if len(close_candidates) > 0 else None
    if close_start is None:
        close_start = first_after(grip < open_th, open_start + min_event_gap)
    if close_start is None:
        close_start = min(T - 1, open_start + max(min_event_gap, pre_grasp_offset))

    pre_grasp = max(open_start, close_start - pre_grasp_offset)
    open_before_close = np.where(opened & (frame_ids < close_start))[0]
    if len(open_before_close) > 0:
        pre_grasp = int(open_before_close[np.argmin(np.abs(open_before_close - pre_grasp))])

    # Block grasp demos often close only to a stable partial width, not below
    # close_th. Use the minimum gripper width between close start and release.
    release_start = first_after(dg > open_change_th, close_start + min_event_gap)
    if release_start is None:
        release_start = first_after(opened, close_start + min_event_gap)
    if release_start is None:
        release_start = T - 1

    place_open = first_after(opened, release_start)
    if place_open is None:
        open_window = np.arange(release_start, T)
        if len(open_window) > 0:
            place_open = int(open_window[np.argmax(grip[open_window])])
        else:
            place_open = release_start

    grasp_end = max(close_start + 1, release_start)
    grasp_window = np.arange(close_start, grasp_end)
    if len(grasp_window) > 0:
        grasp_close = int(grasp_window[np.argmin(grip[grasp_window])])
    else:
        grasp_close = close_start

    search_start = int(np.clip(grasp_close + min_event_gap, 0, T - 1))
    search_end = int(np.clip(max(search_start + 1, release_start), 0, T))
    carry_ids = np.arange(search_start, search_end)

    if len(carry_ids) > 0:
        vel = np.linalg.norm(np.diff(pos, axis=0, prepend=pos[:1]), axis=1)
        z = pos[carry_ids, 2]
        high_z_th = np.quantile(z, lift_search_ratio)
        high_ids = carry_ids[z >= high_z_th]
        if len(high_ids) > 0:
            lift_above = int(high_ids[np.argmin(vel[high_ids])])
        else:
            lift_above = int(carry_ids[np.argmax(z)])
    else:
        lift_above = grasp_close

    pre_place = max(lift_above, release_start - pre_place_offset)
    pre_place = int(np.clip(pre_place, 0, T - 1))

    place_start = int(np.clip(lift_above + 1, 0, T - 1))
    place_end = int(np.clip(max(lift_above + 2, release_start), 0, T))
    place_ids = np.arange(place_start, place_end)
    if len(place_ids) > 0:
        z_after = pos[place_ids, 2]
        low_z_th = np.quantile(z_after, 1.0 - place_drop_ratio)
        low_ids = place_ids[z_after <= low_z_th]
        place_down = int(low_ids[-1]) if len(low_ids) > 0 else int(place_ids[np.argmin(z_after)])
    else:
        place_down = pre_place

    return_closed = None
    if include_return_closed:
        return_closed = first_after(closed, release_start + min_event_gap)
        if return_closed is None:
            close_after_release = np.where((dg < -close_change_th) & (frame_ids >= release_start + min_event_gap))[0]
            if len(close_after_release) > 0:
                return_closed = int(close_after_release[0])
        if return_closed is None:
            return_closed = T - 1

    raw_keys = [
        ("pre_grasp", pre_grasp),
        ("grasp_close", grasp_close),
        ("lift_above", lift_above),
        ("pre_place", pre_place),
        ("place_down", place_down),
        ("release_start", release_start),
        ("place_open", place_open),
    ]
    if include_return_closed:
        raw_keys.append(("return_closed", return_closed))

    key_names = []
    key_ids = []
    last_id = -1
    for name, idx in raw_keys:
        idx = int(np.clip(idx, 0, T - 1))
        if idx <= last_id:
            idx = min(T - 1, last_id + 1)
        if idx > last_id:
            key_names.append(name)
            key_ids.append(idx)
            last_id = idx

    if len(key_ids) == 0:
        key_names = ["fallback"]
        key_ids = [min(T - 1, Tr)]

    key_ids = np.asarray(key_ids, dtype=np.int64)
    key_targets = finger_state[key_ids].astype(np.float32)

    fin_sgs = []
    next_fin_sgs = []
    rewards = []
    stage_ids = []

    cur_stage = 0
    for t in range(valid_len):
        while cur_stage < len(key_ids) - 1 and t >= key_ids[cur_stage]:
            cur_stage += 1

        next_stage = min(cur_stage + 1, len(key_ids) - 1)
        target = key_targets[cur_stage]
        next_target = key_targets[next_stage]
        fin_sgs.append(target)
        next_fin_sgs.append(next_target)
        stage_ids.append(cur_stage)

        future = finger_state[t + 1:t + Tr + 1]
        dists = np.linalg.norm(future - target[None, :], axis=1)
        min_dist = float(np.min(dists)) if len(dists) > 0 else np.inf
        if min_dist < goal_dist_thresh:
            r = max_reward
        elif reward_mode == "only_success":
            r = 0.0
        else:
            r = 1.0 - np.tanh(min_dist * reward_scale)
        rewards.append(r)

    out = {
        "subgoal": np.asarray(fin_sgs, dtype=np.float32),
        "next_subgoal": np.asarray(next_fin_sgs, dtype=np.float32),
        "reward": np.asarray(rewards, dtype=np.float32),
    }
    if return_debug:
        out.update({
            "key_ids": key_ids,
            "key_names": key_names,
            "stage_id": np.asarray(stage_ids, dtype=np.int64),
        })
    return out
