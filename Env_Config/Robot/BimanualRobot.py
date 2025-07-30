import os
import sys
import numpy as np
import torch
from termcolor import cprint

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles, quat_to_rot_matrix, rot_matrix_to_quat
from isaacsim.core.prims import SingleXFormPrim, SingleRigidPrim
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.kinematics import KinematicsSolver

sys.path.append(os.getcwd())
from Env_Config.Utils_Project.Transforms import quat_diff_rad, Rotation, get_pose_relat, get_pose_world
from Env_Config.Utils_Project.Code_Tools import float_truncate, dense_trajectory_points_generation


class BimanualRobot(Robot):
    def __init__(self, world: World, translation: np.ndarray, orientation: np.ndarray):
        # define world
        self.world = world
        # define robot name
        self._name = "BimanualRobot"
        # define robot prim path
        self._prim_path = "/World/BimanualRobot"
        # get merged USD file path
        self.asset_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../Assets/Robots/assembly/ur10e_shadow/dual_ur10e.usd")
        # define robot position
        self.translation = translation
        # define robot orientation
        self.orientation = euler_angles_to_quat(orientation, degrees=True)
        
        # add merged USD to stage
        add_reference_to_stage(self.asset_file, self._prim_path)
        # initialize Robot according to USD file loaded in stage
        super().__init__(
            prim_path=self._prim_path,
            name=self._name,
            translation=self.translation,
            orientation=self.orientation,
            articulation_controller=None
        )
        # add robot to the scene
        self.world.scene.add(self)

        # check whether pick point is reachable or not
        self.pre_distance = 0
        self.distance_nochange_epoch = 0
        
    def initialize(self, physics_sim_view):
        # initialize robot
        super().initialize(physics_sim_view)
        # reset default status
        self.set_default_state(position=self.translation, orientation=self.orientation)
        
        # 设置默认关节状态 - 根据dual_ur10e.usd的结构
        # 结构是: [left_arm(6), left_hand(24), right_arm(6), right_hand(24)]
        self.set_joints_default_state(
            np.array([
                # Left arm joints (与DexLeft_Ur10e相同)
                -1.57, -1.84, -2.5, -1.89, -1.57, 0.0,
                # Left hand joints (24个，与DexLeft_Ur10e相同)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                # Right arm joints (与DexRight_Ur10e相同)
                1.57, -1.3, 2.5, -1.25, 1.57, 0.0,
                # Right hand joints (24个，与DexRight_Ur10e相同)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ])
        )
        
        # 获取关节名称和索引 - 保持原来的名称结构（没有前缀）
        # Left arm DOF names and indices (与DexLeft_Ur10e相同)
        self.left_arm_dof_names = [
            "shoulder_pan_joint", 
            "shoulder_lift_joint", 
            "elbow_joint",
            "wrist_1_joint", 
            "wrist_2_joint", 
            "wrist_3_joint"
        ]
        self.left_arm_dof_indices = [self.get_dof_index(dof_name) for dof_name in self.left_arm_dof_names]
        
        # Left hand DOF names and indices (与DexLeft_Ur10e相同)
        self.left_hand_dof_names = [
            "WRJ2", "WRJ1", 
            "FFJ4", "FFJ3", "FFJ2", "FFJ1", 
            "MFJ4", "MFJ3", "MFJ2", "MFJ1", 
            "RFJ4", "RFJ3", "RFJ2", "RFJ1", 
            "LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1", 
            "THJ5", "THJ4", "THJ3", "THJ2", "THJ1"
        ]
        self.left_hand_dof_indices = [self.get_dof_index(dof_name) for dof_name in self.left_hand_dof_names]
        
        # Right arm DOF names and indices (与DexRight_Ur10e相同)
        self.right_arm_dof_names = [
            "shoulder_pan_joint", 
            "shoulder_lift_joint", 
            "elbow_joint",
            "wrist_1_joint", 
            "wrist_2_joint", 
            "wrist_3_joint"
        ]
        self.right_arm_dof_indices = [self.get_dof_index(dof_name) for dof_name in self.right_arm_dof_names]
        
        # Right hand DOF names and indices (与DexRight_Ur10e相同)
        self.right_hand_dof_names = [
            "WRJ2", "WRJ1", 
            "FFJ4", "FFJ3", "FFJ2", "FFJ1", 
            "MFJ4", "MFJ3", "MFJ2", "MFJ1", 
            "RFJ4", "RFJ3", "RFJ2", "RFJ1", 
            "LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1", 
            "THJ5", "THJ4", "THJ3", "THJ2", "THJ1"
        ]
        self.right_hand_dof_indices = [self.get_dof_index(dof_name) for dof_name in self.right_hand_dof_names]

        # kinematics control for both arms
        # 由于两个手臂有相同的关节名称，我们需要为每个手臂创建独立的求解器
        # 左手逆运动学求解器 - 只使用左手的手臂关节
        self.left_ki_solver = KinematicsSolver(
            self, 
            end_effector_frame_name="ee_link",
            joint_names=self.left_arm_dof_names,
            joint_indices=self.left_arm_dof_indices
        )
        
        # 右手逆运动学求解器 - 只使用右手的手臂关节
        self.right_ki_solver = KinematicsSolver(
            self, 
            end_effector_frame_name="ee_link",
            joint_names=self.right_arm_dof_names,
            joint_indices=self.right_arm_dof_indices
        )
        
        # 使用不同的prim路径来区分左右手的末端执行器
        self.left_end_effector = SingleRigidPrim(self._prim_path + "/ur10e_shadow_left_hand_glb/ee_link", "left_end_effector")
        self.right_end_effector = SingleRigidPrim(self._prim_path + "/ur10e_shadow_right_hand_glb/ee_link", "right_end_effector")
        
        self.left_end_effector.initialize(physics_sim_view)
        self.right_end_effector.initialize(physics_sim_view)
        
        self.disable_gravity()
        
        # 由于两个手臂有相同的关节名称，我们需要通过索引范围来区分
        # 假设关节顺序是: [left_arm(6), left_hand(24), right_arm(6), right_hand(24)]
        total_dofs = len(self.dof_names)
        print(f"Total DOFs: {total_dofs}")
        print(f"DOF names: {self.dof_names}")
        
        # 重新计算关节索引，基于位置而不是名称
        # 左手关节索引：前30个关节 (6个手臂 + 24个手部)
        self.left_arm_dof_indices = list(range(0, 6))  # 前6个是左手手臂关节
        self.left_hand_dof_indices = list(range(6, 30))  # 接下来24个是左手手部关节
        
        # 右手关节索引：后30个关节 (6个手臂 + 24个手部)
        self.right_arm_dof_indices = list(range(30, 36))  # 接下来6个是右手手臂关节
        self.right_hand_dof_indices = list(range(36, 60))  # 最后24个是右手手部关节
        
        print(f"Left arm indices: {self.left_arm_dof_indices}")
        print(f"Left hand indices: {self.left_hand_dof_indices}")
        print(f"Right arm indices: {self.right_arm_dof_indices}")
        print(f"Right hand indices: {self.right_hand_dof_indices}")
        
    # //////////////////////////////////////////////////////////////
    # /                                                            /
    # /********              Hand Pose Control             ********/
    # /                                                            /
    # //////////////////////////////////////////////////////////////
    
    def set_both_hand_state(self, left_hand_state: str = "None", right_hand_state: str = "None"):
        # set hand pose according to given hand state ('open' / 'close')
        if left_hand_state == "close":
            left_hand_pose = ArticulationAction(
                joint_positions=np.array([ 
                        7.93413107e-02,-1.50470986e-02 ,2.67736827e-01 ,7.11539682e-01,
                        9.51350867e-01 ,8.78066358e-01,-8.53160175e-02 ,8.53321711e-01,
                        7.87673219e-01 ,7.87673219e-01 ,3.40252007e-02 ,6.76240842e-01,
                        1.12421269e+00 ,8.73928860e-01 ,2.19334288e-01 ,1.76221645e-04,
                        4.14303315e-01 ,1.31339149e+00 ,7.58440614e-01 ,9.76555113e-01,
                        1.10293828e+00 ,5.89046458e-02,-1.27955038e-02 ,5.75099223e-02]),
                joint_indices=np.array(self.left_hand_dof_indices)
            )
        elif left_hand_state == "open":
            left_hand_pose = ArticulationAction(
                joint_positions=np.array([
                -0.06150788, -0.07899461,
                -0.34906599, 0.02261488 , 0.01851636,0.01851636,
                -0.08347217, 0.         , 0.00755569,0.00755569 ,
                -0.09509672, 0.00797776, 0.0236096 ,0.0236096   ,
                0.06855482, -0.34906599, 0.04799218 ,0.03465942, 0.03446003 ,
                -0.06890292, 0.63903833, -0.15110777,-0.08261109  ,0.02802108
                ]),
                joint_indices=np.array(self.left_hand_dof_indices)
            )
        elif left_hand_state == "pinch":
            left_hand_pose = ArticulationAction(
                joint_positions=np.array([
                    -1.35955052e-01,  5.75466704e-02, -3.49065989e-01,  5.68935978e-01,
                    1.15562234e+00,  7.44316154e-01, -1.70137126e-01,  6.15123597e-02,
                    7.19645386e-02,  7.19645386e-02,  3.24903752e-02,  6.18193961e-19,
                    3.44893551e-02,  3.44893551e-02,  4.06464774e-20, -3.49065989e-01,
                    8.79347072e-02,  5.14625171e-02,  5.14625171e-02,  3.32827328e-01,
                    5.36995551e-01,  7.89638528e-02,  3.2958011e-01,  2.48388920e-01,
                ]),
                joint_indices=np.array(self.left_hand_dof_indices)
            )
        elif left_hand_state == "cradle":
            left_hand_pose = ArticulationAction(
                joint_positions=np.array([
                    -0.16060828 , 0.09542108, -0.32950141 , 0.00419018 , 0.06623378 , 0.06622696,
                    -0.09273153 , 0.0146647  , 0.04232529 , 0.04232529, -0.0497151  , 0.01105581,
                    0.07081381 , 0.07081381 , 0.06527984, -0.34906599 , 0.04872768 , 0.05029842,
                    0.05029842 , 0.53537881 , 0.8075118  , 0.19626232 , 0.58037802 , 0.0140745,
                ]),
                joint_indices=np.array(self.left_hand_dof_indices)
            )
        elif left_hand_state == "smooth":
            left_hand_pose = ArticulationAction(
                joint_positions=np.array(
                    [-6.67141718e-02,  1.66426553e-01, -2.43158132e-01,  2.33355593e-01,
                    6.73720170e-01,  6.44790593e-02, -1.62202123e-01,  5.12784203e-01,
                    1.65908404e-01,  8.80870383e-02,  1.94555304e-01,  3.39687364e-01,
                    4.71966296e-01,  6.82939946e-21,  1.26723228e-01,  9.82577176e-02,
                    3.40769729e-01,  1.61393453e-01,  5.65633490e-02,  6.14344438e-01,
                    5.08354918e-01, -3.19813694e-02,  5.31338993e-02,  7.06677286e-02,]
                ),
                joint_indices=np.array(self.left_hand_dof_indices)
            )

        
        if right_hand_state == "close":
            right_hand_pose = ArticulationAction(
                joint_positions=np.array([ 
                        7.93413107e-02,-1.50470986e-02 ,2.67736827e-01 ,7.11539682e-01,
                        9.51350867e-01 ,8.78066358e-01,-8.53160175e-02 ,8.53321711e-01,
                        7.87673219e-01 ,7.87673219e-01 ,3.40252007e-02 ,6.76240842e-01,
                        1.12421269e+00 ,8.73928860e-01 ,2.19334288e-01 ,1.76221645e-04,
                        4.14303315e-01 ,1.31339149e+00 ,7.58440614e-01 ,9.76555113e-01,
                        1.10293828e+00 ,5.89046458e-02,-1.27955038e-02 ,5.75099223e-02]),
                joint_indices=np.array(self.right_hand_dof_indices)
            )
        elif right_hand_state == "open":
            right_hand_pose = ArticulationAction(
                joint_positions=np.array([
                -0.06150788, -0.07899461,
                -0.34906599, 0.02261488 , 0.01851636,0.01851636,
                -0.08347217, 0.         , 0.00755569,0.00755569 ,
                -0.09509672, 0.00797776, 0.0236096 ,0.0236096   ,
                0.06855482, -0.34906599, 0.04799218 ,0.03465942, 0.03446003 ,
                -0.06890292, 0.63903833, -0.15110777,-0.08261109  ,0.02802108
                ]),
                joint_indices=np.array(self.right_hand_dof_indices)
            )
        elif right_hand_state == "pinch":
            right_hand_pose = ArticulationAction(
                joint_positions=np.array([
                    -1.35955052e-01,  5.75466704e-02, -3.49065989e-01,  5.68935978e-01,
                    1.15562234e+00,  7.44316154e-01, -1.70137126e-01,  6.15123597e-02,
                    7.19645386e-02,  7.19645386e-02,  3.24903752e-02,  6.18193961e-19,
                    3.44893551e-02,  3.44893551e-02,  4.06464774e-20, -3.49065989e-01,
                    8.79347072e-02,  5.14625171e-02,  5.14625171e-02,  3.32827328e-01,
                    5.36995551e-01,  7.89638528e-02,  3.2958011e-01,  2.48388920e-01,
                ]),
                joint_indices=np.array(self.right_hand_dof_indices)
            )
        elif right_hand_state == "cradle":
            right_hand_pose = ArticulationAction(
                joint_positions=np.array([
                    -0.16060828 , 0.09542108, -0.32950141 , 0.00419018 , 0.06623378 , 0.06622696,
                    -0.09273153 , 0.0146647  , 0.04232529 , 0.04232529, -0.0497151  , 0.01105581,
                    0.07081381 , 0.07081381 , 0.06527984, -0.34906599 , 0.04872768 , 0.05029842,
                    0.05029842 , 0.53537881 , 0.8075118  , 0.19626232 , 0.58037802 , 0.0140745,
                ]),
                joint_indices=np.array(self.right_hand_dof_indices)
            )
        elif right_hand_state == "smooth":
            right_hand_pose = ArticulationAction(
                joint_positions=np.array([
                    -6.67141718e-02,  1.66426553e-01, -2.43158132e-01,  2.33355593e-01,
                    6.73720170e-01,  6.44790593e-02, -1.62202123e-01,  5.12784203e-01,
                    1.65908404e-01,  8.80870383e-02,  1.94555304e-01,  3.39687364e-01,
                    4.71966296e-01,  6.82939946e-21,  1.26723228e-01,  9.82577176e-02,
                    3.40769729e-01,  1.61393453e-01,  5.65633490e-02,  6.14344438e-01,
                    5.08354918e-01, -3.19813694e-02,  5.31338993e-02,  7.06677286e-02,
                    ]
                ),
                joint_indices=np.array(self.right_hand_dof_indices)
            )

        # apply action
        if left_hand_state != "None":
            self.apply_action(left_hand_pose)
        if right_hand_state != "None":
            self.apply_action(right_hand_pose)
        
        # wait action to be done
        for i in range(20):
            self.world.step(render=True)
       
    # //////////////////////////////////////////////////////////////
    # /                                                            /
    # /********         Inverse Kinematics Control         ********/
    # /                                                            /
    # //////////////////////////////////////////////////////////////      
            
    def dense_move_both_ik(self, left_pos, left_ori, right_pos, right_ori, angular_type="quat", degree=True, dense_sample_scale:int=0.01):
        '''
        Move both arms simultaneously once and use dense trajectory to guarantee smoothness.
        '''
        assert angular_type in ["quat", "euler"]
        if angular_type == "euler" and left_ori is not None:
            if degree:
                left_ori = euler_angles_to_quat(left_ori, degrees=True)
            else:
                left_ori = euler_angles_to_quat(left_ori)
        if angular_type == "euler" and right_ori is not None:
            if degree:
                right_ori = euler_angles_to_quat(right_ori, degrees=True)
            else:
                right_ori = euler_angles_to_quat(right_ori)
        
        ee_left_pos = left_pos + Rotation(left_ori, np.array([-0.37, -0.025, 0.025]))
        ee_right_pos = right_pos + Rotation(right_ori, np.array([-0.37, -0.025, -0.025]))
        
        ee_left_ori = quat_to_rot_matrix(left_ori)
        ee_right_ori = quat_to_rot_matrix(right_ori)
        
        base_pose, base_ori = self.get_world_pose()
        base_ori = quat_to_rot_matrix(base_ori)
        
        current_ee_left_pos, current_ee_left_ori = self.left_end_effector.get_world_pose()
        current_ee_right_pos, current_ee_right_ori = self.right_end_effector.get_world_pose()
        
        dense_sample_num = int(max(np.linalg.norm(current_ee_left_pos - ee_left_pos), np.linalg.norm(current_ee_right_pos - ee_right_pos)) // dense_sample_scale)

        left_interp_pos = dense_trajectory_points_generation(
            start_pos=current_ee_left_pos, 
            end_pos=ee_left_pos,
            num_points=dense_sample_num,
        )
        
        right_interp_pos = dense_trajectory_points_generation(
            start_pos=current_ee_right_pos, 
            end_pos=ee_right_pos,
            num_points=dense_sample_num,
        )

        for i in range(dense_sample_num):
            left_pose_in_local, left_ori_in_local = get_pose_relat(left_interp_pos[i], ee_left_ori, base_pose, base_ori)
            left_ori_in_local = rot_matrix_to_quat(left_ori_in_local)
            
            right_pose_in_local, right_ori_in_local = get_pose_relat(right_interp_pos[i], ee_right_ori, base_pose, base_ori)
            right_ori_in_local = rot_matrix_to_quat(right_ori_in_local)
            
            left_action, left_succ = self.left_ki_solver.compute_inverse_kinematics(
                left_pose_in_local, 
                left_ori_in_local,
                position_tolerance=0.06
            )
            right_action, right_succ = self.right_ki_solver.compute_inverse_kinematics(
                right_pose_in_local, 
                right_ori_in_local,
                position_tolerance=0.06
            )
            
            if left_succ and right_succ:            
                # 创建完整的关节动作，包括所有关节
                full_joint_positions = np.zeros(len(self.dof_names))
                
                # 设置左手手臂关节位置
                for i, joint_idx in enumerate(self.left_arm_dof_indices):
                    full_joint_positions[joint_idx] = left_action.joint_positions[i]
                
                # 设置右手手臂关节位置
                for i, joint_idx in enumerate(self.right_arm_dof_indices):
                    full_joint_positions[joint_idx] = right_action.joint_positions[i]
                
                # 保持手部关节的当前位置
                current_state = self.get_joints_state()
                if current_state is not None:
                    # 保持左手手部关节位置
                    for i, joint_idx in enumerate(self.left_hand_dof_indices):
                        full_joint_positions[joint_idx] = current_state.positions[joint_idx]
                    # 保持右手手部关节位置
                    for i, joint_idx in enumerate(self.right_hand_dof_indices):
                        full_joint_positions[joint_idx] = current_state.positions[joint_idx]
                
                # 创建完整的动作
                combined_action = ArticulationAction(
                    joint_positions=full_joint_positions,
                    joint_indices=list(range(len(self.dof_names)))
                )
                self.apply_action(combined_action)
                self.world.step(render=True)
        
        if left_succ and right_succ:
            print("\033[32mfinish moving!\033[0m")
        else:
            if not left_succ and not right_succ:
                print("\033[31mboth hand failed to move completely!\033[0m")
            elif not left_succ:
                print("\033[31mleft hand failed to move completely!\033[0m")
            elif not right_succ:
                print("\033[31mright hand failed to move completely!\033[0m")
        
        return left_succ and right_succ
    
    def move_both_with_blocks(self, left_pos, left_ori, right_pos, right_ori,
                            angular_type="quat", degree=None, dense_sample_scale:int=0.01,
                            attach=None, indices=None):
        '''
        Move both arms simultaneously once and use dense trajectory to guarantee smoothness.
        '''
        assert angular_type in ["quat", "euler"]
        if angular_type == "euler" and left_ori is not None:
            if degree:
                left_ori = euler_angles_to_quat(left_ori, degrees=True)
            else:
                left_ori = euler_angles_to_quat(left_ori)
        if angular_type == "euler" and right_ori is not None:
            if degree:
                right_ori = euler_angles_to_quat(right_ori, degrees=True)
            else:
                right_ori = euler_angles_to_quat(right_ori)
            
        base_pose, base_ori = self.get_world_pose()
        base_ori = quat_to_rot_matrix(base_ori)
        
        ee_left_pos, ee_left_ori = left_pos + Rotation(left_ori, np.array([-0.35, -0.055, 0.025])), left_ori
        ee_right_pos, ee_right_ori = right_pos + Rotation(right_ori, np.array([-0.35, -0.055, -0.025])), right_ori
        
        current_ee_left_pos, current_ee_left_ori = self.left_end_effector.get_world_pose()
        current_ee_right_pos, current_ee_right_ori = self.right_end_effector.get_world_pose()
        
        dense_sample_num = int(max(np.linalg.norm(current_ee_left_pos - ee_left_pos), np.linalg.norm(current_ee_right_pos - ee_right_pos)) // dense_sample_scale)

        left_interp_pos = dense_trajectory_points_generation(
            start_pos=current_ee_left_pos, 
            end_pos=ee_left_pos,
            num_points=dense_sample_num,
        )

        right_interp_pos = dense_trajectory_points_generation(
            start_pos=current_ee_right_pos, 
            end_pos=ee_right_pos,
            num_points=dense_sample_num,
        )

        left_finger = SingleXFormPrim("/World/BimanualRobot/ur10e_shadow_left_hand_glb/fftip")
        right_finger = SingleXFormPrim("/World/BimanualRobot/ur10e_shadow_right_hand_glb/fftip")

        for i in range(dense_sample_num):
            left_pose_in_local, left_ori_in_local = get_pose_relat(left_interp_pos[i], ee_left_ori, base_pose, base_ori)
            left_ori_in_local = rot_matrix_to_quat(left_ori_in_local)
            
            right_pose_in_local, right_ori_in_local = get_pose_relat(right_interp_pos[i], ee_right_ori, base_pose, base_ori)
            right_ori_in_local = rot_matrix_to_quat(right_ori_in_local)
            
            left_action, left_succ = self.left_ki_solver.compute_inverse_kinematics(
                left_pose_in_local, 
                left_ori_in_local,
                position_tolerance=0.06
            )
            right_action, right_succ = self.right_ki_solver.compute_inverse_kinematics(
                right_pose_in_local, 
                right_ori_in_local,
                position_tolerance=0.06
            )
            
            if left_succ and right_succ:            
                # 创建完整的关节动作，包括所有关节
                full_joint_positions = np.zeros(len(self.dof_names))
                
                # 设置左手手臂关节位置
                for i, joint_idx in enumerate(self.left_arm_dof_indices):
                    full_joint_positions[joint_idx] = left_action.joint_positions[i]
                
                # 设置右手手臂关节位置
                for i, joint_idx in enumerate(self.right_arm_dof_indices):
                    full_joint_positions[joint_idx] = right_action.joint_positions[i]
                
                # 保持手部关节的当前位置
                current_state = self.get_joints_state()
                if current_state is not None:
                    # 保持左手手部关节位置
                    for i, joint_idx in enumerate(self.left_hand_dof_indices):
                        full_joint_positions[joint_idx] = current_state.positions[joint_idx]
                    # 保持右手手部关节位置
                    for i, joint_idx in enumerate(self.right_hand_dof_indices):
                        full_joint_positions[joint_idx] = current_state.positions[joint_idx]
                
                # 创建完整的动作
                combined_action = ArticulationAction(
                    joint_positions=full_joint_positions,
                    joint_indices=list(range(len(self.dof_names)))
                )
                self.apply_action(combined_action)
                self.world.step(render=True)

            if attach is not None and indices is not None:
                block1_pos = left_finger.get_world_pose()[0]
                block0_pos = right_finger.get_world_pose()[0]
                block_pos = []
                block_pos.append(block0_pos)
                block_pos.append(block1_pos)
                gripper_ori = []
                gripper_ori.append([1.0, 0.0, 0.0, 0.0])
                gripper_ori.append([1.0, 0.0, 0.0, 0.0])
                for i in indices:
                    attach.block_list[i].set_world_pose(block_pos[i], gripper_ori[i])
                self.world.step(render=True)

        if left_succ and right_succ:
            print("\033[32mfinish moving!\033[0m")
        else:
            if not left_succ and not right_succ:
                print("\033[31mboth hand failed to move completely!\033[0m")
            elif not left_succ:
                print("\033[31mleft hand failed to move completely!\033[0m")
            elif not right_succ:
                print("\033[31mright hand failed to move completely!\033[0m")
        
        return left_succ and right_succ 