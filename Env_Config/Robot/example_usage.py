"""
使用示例：如何使用合并后的USD文件和新的BimanualRobot类
"""

import numpy as np
from isaacsim.core.api import World
from Env_Config.Robot.BimanualRobot import BimanualRobot

def example_usage():
    # 创建世界
    world = World()
    
    # 定义机器人位置和方向
    # 注意：现在只需要一个位置和方向，因为两个手臂在同一个USD文件中
    robot_pos = np.array([0.0, 0.0, 0.0])  # 机器人基座位置
    robot_ori = np.array([0.0, 0.0, 0.0])  # 机器人基座方向（欧拉角，度）
    
    # 创建双手机器人实例
    bimanual_robot = BimanualRobot(world, robot_pos, robot_ori)
    
    # 初始化机器人
    world.scene.add(bimanual_robot)
    world.reset()
    
    # 示例1：设置手部状态
    print("设置手部状态...")
    bimanual_robot.set_both_hand_state(left_hand_state="open", right_hand_state="close")
    
    # 示例2：同时移动两个手臂
    print("移动两个手臂...")
    left_target_pos = np.array([0.5, 0.3, 0.5])   # 左手目标位置
    left_target_ori = np.array([1.0, 0.0, 0.0, 0.0])  # 左手目标方向（四元数）
    
    right_target_pos = np.array([0.5, -0.3, 0.5])  # 右手目标位置
    right_target_ori = np.array([1.0, 0.0, 0.0, 0.0])  # 右手目标方向（四元数）
    
    success = bimanual_robot.dense_move_both_ik(
        left_pos=left_target_pos,
        left_ori=left_target_ori,
        right_pos=right_target_pos,
        right_ori=right_target_ori,
        angular_type="quat",
        dense_sample_scale=0.01
    )
    
    if success:
        print("移动成功！")
    else:
        print("移动失败！")
    
    # 示例3：使用欧拉角指定方向
    print("使用欧拉角移动...")
    left_target_pos_euler = np.array([0.3, 0.2, 0.4])
    left_target_ori_euler = np.array([0.0, 90.0, 0.0])  # 欧拉角，度
    
    right_target_pos_euler = np.array([0.3, -0.2, 0.4])
    right_target_ori_euler = np.array([0.0, -90.0, 0.0])  # 欧拉角，度
    
    success = bimanual_robot.dense_move_both_ik(
        left_pos=left_target_pos_euler,
        left_ori=left_target_ori_euler,
        right_pos=right_target_pos_euler,
        right_ori=right_target_ori_euler,
        angular_type="euler",
        degree=True,
        dense_sample_scale=0.01
    )
    
    print("示例完成！")

if __name__ == "__main__":
    example_usage() 