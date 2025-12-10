from isaacsim import SimulationApp
# 创建 Isaac Sim 仿真应用，headless=True 表示使用无界面模式（适合批量验证 / 集群）
# - 若需要在本地可视化调试，可以改为 headless=False；但服务器/集群上建议保持为 True
simulation_app = SimulationApp({"headless": True})

# ------------------------- #
#   加载 Python 外部依赖    #
# ------------------------- #
import os
import sys
import time
import numpy as np
import open3d as o3d
from termcolor import cprint
import threading

# ------------------------- #
#   加载 Isaac 相关依赖     #
# ------------------------- #
import omni.replicator.core as rep
import isaacsim.core.utils.prims as prims_utils
from pxr import UsdGeom,UsdPhysics,PhysxSchema, Gf
from isaacsim.core.api import World
from isaacsim.core.api import SimulationContext
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualCuboid
from isaacsim.core.utils.prims import is_prim_path_valid, set_prim_visibility
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading
from isaacsim.core.prims import SingleXFormPrim, SingleClothPrim, SingleRigidPrim, SingleGeometryPrim, SingleParticleSystem, SingleDeformablePrim
from isaacsim.core.prims import XFormPrim, ClothPrim, RigidPrim, GeometryPrim, ParticleSystem
from isaacsim.core.utils.types import ArticulationAction, ArticulationActions
from omni.physx.scripts import deformableUtils,particleUtils,physicsUtils

# ------------------------- #
#   加载本工程自定义模块    #
# ------------------------- #
sys.path.append(os.getcwd())
from Env_StandAlone.BaseEnv import BaseEnv
from Env_Config.Garment.Particle_Garment import Particle_Garment
from Env_Config.Garment.Deformable_Garment import Deformable_Garment
from Env_Config.Robot.BimanualDex_Ur10e import Bimanual_Ur10e
from Env_Config.Camera.Recording_Camera import Recording_Camera
from Env_Config.Room.Real_Ground import Real_Ground
from Env_Config.Utils_Project.Code_Tools import get_unique_filename, normalize_columns
from Env_Config.Utils_Project.Parse import parse_args_val
from Env_Config.Utils_Project.Position_Judge import judge_pcd
from Env_Config.Room.Object_Tools import set_prim_visible_group, delete_prim_group
from Model_HALO.GAM.GAM_Encapsulation import GAM_Encapsulation
from Model_HALO.SADP_G.SADP_G import SADP_G

class FoldTops_Env(BaseEnv):
    """
    上衣折叠任务（Fold Tops）环境，基于 BaseEnv 搭建：
    - 加载地面、衣服（布料）、双臂机器人、相机等仿真资产
    - 初始化 HALO 中的 GAM（几何可操作性模型）与 SADP_G（三阶段策略模型）
    - 对外提供 step / 相机观测 / 机器人控制 等接口

    整体流程：
    1. 在场景中放置地面、衣服、双臂机器人和两个相机
    2. 调用 GAM 模型从衣服点云中提取关键可操作点（manipulation points）以及对应特征
    3. 按照 HALO 设定的 3 个阶段，依次调用 SADP_G 策略网络生成机械臂关节动作序列
    4. 完成折叠后，基于点云与几何规则自动评估折叠效果，并在需要时记录日志/图片/视频

    该类本身只负责“仿真环境 + 策略推理”的封装，不负责策略训练过程。
    """
    def __init__(
        self, 
        pos:np.ndarray=None, 
        ori:np.ndarray=None, 
        usd_path:str=None, 
        ground_material_usd:str=None,
        record_video_flag:bool=False, 
        training_data_num:int=100,
        stage_1_checkpoint_num:int=1500, 
        stage_2_checkpoint_num:int=1500, 
        stage_3_checkpoint_num:int=1500, 
    ):
        """
        初始化 Fold Tops 仿真环境。

        参数：
        - pos: np.ndarray，衣服初始位置 (x, y, z)，这里只真正使用 x、y，z 会在内部重设为 0.2
        - ori: np.ndarray，衣服初始欧拉角朝向 (roll, pitch, yaw)
        - usd_path: str，自定义衣服 USD 路径；若为 None，则使用默认的长袖上衣资产
        - ground_material_usd: str，地面材质 USD 路径，用于设置地面的纹理与外观
        - record_video_flag: bool，是否在仿真过程中录制视频（由 env_camera 生成 mp4）
        - training_data_num: int，训练数据量，用于构造/加载对应的 SADP_G 模型配置
        - stage_1_checkpoint_num / stage_2_checkpoint_num / stage_3_checkpoint_num: int，
          三个阶段策略网络分别使用的 checkpoint 编号，用于从对应目录中加载权重。

        通常不直接实例化本类，而是通过下方的 FoldTops 函数统一创建和调用。
        """
        # 先初始化基础仿真环境（时间步长、物理世界、场景等）
        super().__init__()
        
        # ------------------------------------ #
        # ---        Add Env Assets        --- #
        # ------------------------------------ #
        # 加载真实地面（带材质），主要用作布料和机器人放置的基础平面
        self.ground = Real_Ground(
            self.scene, 
            visual_material_usd = ground_material_usd,
            # you can use materials in 'Assets/Material/Floor' to change the texture of ground.
        )
        
        # -------------------- #
        #   加载布料（上衣）    #
        # -------------------- #
        # 这里使用基于粒子的布料仿真模型 Particle_Garment
        # usd_path 可以在 main 函数或上层传入自定义衣服路径，否则使用默认衣服
        self.garment = Particle_Garment(
            self.world, 
            pos=np.array([0, 3.0, 0.6]),
            ori=np.array([0.0, 0.0, 0.0]),
            usd_path="Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_018/TCLC_018_obj.usd" if usd_path is None else usd_path,
            contact_offset=0.012,             
            rest_offset=0.010,                
            particle_contact_offset=0.012,    
            fluid_rest_offset=0.010,
            solid_rest_offset=0.010,
        )
        # 下面给出了一些可替换的 Garment usd 路径示例，可在实验中手动替换测试泛化能力：
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Jacket032/TCLC_Jacket032_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Jacket152/TCLC_Jacket152_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Top566/TCLC_Top566_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Top584/TCLC_Top584_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_top118/TCLC_top118_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Top476/TCLC_Top476_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Top030/TCLC_Top030_obj.usd",  

        # ------------------------------ #
        #   加载双臂 Dexterous Hand 机器人 #
        # ------------------------------ #
        # Bimanual_Ur10e 内部包含两个 UR10e 机械臂 + Dexterous Hand
        # 这里设置左右机械臂的初始位姿（在世界坐标系中的位置与欧拉角）
        self.bimanual_dex = Bimanual_Ur10e(
            self.world,
            dexleft_pos=np.array([-0.8, 0.0, 0.5]),
            dexleft_ori=np.array([0.0, 0.0, 0.0]),
            dexright_pos=np.array([0.8, 0.0, 0.5]),
            dexright_ori=np.array([0.0, 0.0, 0.0]),
        )

        # ------------------- #
        #   加载相机（两个）  #
        # ------------------- #
        # garment_camera：只看衣服区域，用于 GAM 模型输入（点云 + 分割）
        self.garment_camera = Recording_Camera(
            camera_position=np.array([0.0, 1.0, 6.75]), 
            camera_orientation=np.array([0, 90.0, 90.0]),
            prim_path="/World/garment_camera",
        )
        
        # env_camera：俯视环境相机，用于策略（SADP_G）的环境点云观测与录制视频
        self.env_camera = Recording_Camera(
            camera_position=np.array([0.0, 4.0, 6.0]),
            camera_orientation=np.array([0, 60, -90.0]),
            prim_path="/World/env_camera",
        )
        
        # 用于缓存：初始衣服点云 & HALO 输出的可操作性特征
        self.garment_pcd = None
        self.points_affordance_feature = None
        
        # ----------------------------- #
        #   加载 HALO 中的 GAM 模型     #
        # ----------------------------- #
        # GAM_Encapsulation：给定衣服点云，预测关键可操作点（manipulation points）
        self.model = GAM_Encapsulation(catogory="Tops_LongSleeve") 
        
        # ----------------------------- #
        #   加载三阶段策略模型 SADP_G   #
        # ----------------------------- #
        # stage_1：抓取 / 摊平领口等第一阶段动作
        self.sadp_g_stage_1 = SADP_G(task_name="Fold_Tops_stage_1", data_num=training_data_num, checkpoint_num=stage_1_checkpoint_num)

        # stage_2：另一只手的抓取和拉拽等第二阶段动作
        self.sadp_g_stage_2 = SADP_G(task_name="Fold_Tops_stage_2", data_num=training_data_num, checkpoint_num=stage_2_checkpoint_num)

        # stage_3：双手协同将底部折到上方等第三阶段动作
        self.sadp_g_stage_3 = SADP_G(task_name="Fold_Tops_stage_3", data_num=training_data_num, checkpoint_num=stage_3_checkpoint_num)  
        
        # ------------------------------------ #
        # --- Initialize World to be Ready --- #
        # ------------------------------------ #
        # 初始化物理世界：重置场景中所有物体并稳定一段时间
        self.reset()
        
        # 将衣服放置到指定位置和姿态（pos/ori 一般在 main 中根据参数随机化）
        # 这里只改 z（高度）为 0.2，保证衣服初始时略高于地面，便于布料下落到稳定状态
        self.garment.set_pose(pos=np.array([pos[0], pos[1], 0.2]), ori=ori)
        self.position = [pos[0], pos[1], 0.2]
        self.orientation = ori
        
        # 初始化衣服相机，只渲染并分割 Garment 这一类物体，用于获取纯衣服点云
        self.garment_camera.initialize(
            segment_pc_enable=True, 
            segment_prim_path_list=[
                "/World/Garment/garment"
            ]
        )
        
        # 初始化环境相机，开启深度图输出，后续可以转换为环境点云
        self.env_camera.initialize(depth_enable=True)
        
        # 若需要录制视频，则开启独立线程持续采集 rgb 帧，后续再合成为 mp4
        # 这里记录的是 env_camera 的图像
        if record_video_flag:
            self.thread_record = threading.Thread(target=self.env_camera.collect_rgb_graph_for_vedio)
            self.thread_record.daemon = True
        
        # 初始状态先打开双手，避免一开始就抓取布料
        self.bimanual_dex.set_both_hand_state("open", "open")

        # 预先仿真 100 步，让场景中的布料与机器人等充分稳定到物理合理状态
        for i in range(100):
            self.step()
            
        cprint("----------- World Configuration -----------", color="magenta", attrs=["bold"])
        cprint(f"usd_path: {usd_path}", "magenta")
        cprint(f"pos_x: {pos[0]}", "magenta")
        cprint(f"pos_y: {pos[1]}", "magenta")
        cprint("----------- World Configuration -----------", color="magenta", attrs=["bold"])
        
        cprint("World Ready!", "green", "on_green")


def FoldTops(pos, ori, usd_path, ground_material_usd, validation_flag, record_video_flag, training_data_num, stage_1_checkpoint_num, stage_2_checkpoint_num, stage_3_checkpoint_num):
    """
    单次 Fold Tops 任务入口：
    - 构建环境（FoldTops_Env）
    - 使用 GAM 模型预测关键抓取点
    - 按照 HALO 设计的 3 个阶段调用 SADP_G 策略生成机器人动作
    - 最后根据折叠结果计算成功率，并在需要时写入日志 / 保存图像 / 视频

    参数说明：
    - pos: np.ndarray，衣服初始位置 (x, y, z)，这里只主要关心 x、y，z 会在环境内部重设为 0.2
    - ori: np.ndarray，衣服初始欧拉角朝向
    - usd_path: str，衣服 USD 路径；为 None 时使用默认上衣
    - ground_material_usd: str，地面材质路径
    - validation_flag: bool，是否为“批量验证模式”
        * True  ：每次运行会将结果写入 log 文件，并在结束后自动关闭仿真
        * False ：仅运行一次任务，结束后保持仿真运行，方便人工观察
    - record_video_flag: bool，是否记录验证过程视频
    - training_data_num: int，SADP_G 模型对应的训练数据规模（影响模型配置/权重）
    - stage_1_checkpoint_num / stage_2_checkpoint_num / stage_3_checkpoint_num: int，
      三个阶段策略网络所加载的 checkpoint 编号

    返回：
    - 无显式返回值，最终结果通过日志与终端输出体现（success / fail）。
    """
    
    # 构建上衣折叠环境
    env = FoldTops_Env(pos, ori, usd_path, ground_material_usd, record_video_flag, training_data_num, stage_1_checkpoint_num, stage_2_checkpoint_num, stage_3_checkpoint_num)
    
    # 若需要录制视频，则启动采集线程（异步）
    if record_video_flag:
        env.thread_record.start()
    
    # ------------------------------- #
    #   1. 获取初始衣服点云（GAM输入）  #
    # ------------------------------- #
    # 为了获取“只有衣服”的点云，我们暂时隐藏两只手臂（DexLeft / DexRight）
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight"],
        visible=False,
    )
    for i in range(50):
        env.step()
        
    # 从衣服相机获取分割后的点云（pcd 为 Nx3，color 为 Nx3 颜色）
    pcd, color = env.garment_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".ply"),
        real_time_watch=False,
    )
    # 将初始衣服点云缓存到环境中，供后续策略模型使用
    env.garment_pcd = pcd
    
    # 取消隐藏，将两只手重新显示出来，后续执行抓取与折叠动作
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight"],
        visible=True,
    )
    for i in range(50):
        env.step()
    
    # 使用 GAM 模型在输入点云上预测关键操作点，以及每个点的可操作性评分：
    # - manipulation_points：关键点在 3D 空间中的坐标（这里 index_list 指定使用哪些点）
    # - indices：对应点在点云中的索引
    # - points_similarity：每个点的可操作性（affordance）或相似性特征，用于后续策略条件
    #   注意：index_list 中的索引是基于 GAM 训练时固定下来的，语义上对应衣服的关键部位
    manipulation_points, indices, points_similarity = env.model.get_manipulation_points(
        input_pcd=pcd,
        index_list=[957, 501, 1902, 448, 1196, 422],
    )

    # 轻微调整关键点的高度：
    # - 前 4 个点的 z 坐标设为 0.02（略高于衣服表面，避免初次插入过深）
    # - 后 2 个点设为 0.0，主要用于后续折叠时的参考
    manipulation_points[0:4, 2] = 0.02
    manipulation_points[4:, 2] = 0.0
    
    # ====================================================== #
    #                     Stage 1：左手                      #
    #   目标：将左手移动到 GAM 给出的第一个关键点附近，并让
    #         SADP_G 第一阶段策略根据当前观测生成动作序列。
    # ====================================================== #
        
    # 为左手阶段构造点的可操作性特征：
    # 这里取 points_similarity 中第 0 个点的特征，并在通道维上复制一次，
    # 然后做列归一化，得到供策略模型使用的特征矩阵
    env.points_affordance_feature = normalize_columns(
        np.concatenate([points_similarity[0:1], points_similarity[0:1]], axis=0).T
    )
            
    # 先通过逆解控制将“左手”末端执行器移动到第一个关键点附近
    env.bimanual_dex.dexleft.dense_step_action(
        target_pos=manipulation_points[0],
        target_ori=np.array([0.579, -0.579, -0.406, 0.406]),
        angular_type="quat",
    )
    
    for i in range(20):
        env.step()
    
    # 循环调用策略 8 次，每次都产生一个长为 4 的 action 序列
    for i in range(8):
        
        print(f"Stage_1_Step: {i}")

        # --------------------------- #
        #   1. 构造当前观测 obs       #
        # --------------------------- #
        joint_pos_L = env.bimanual_dex.dexleft.get_joint_positions()
        joint_pos_R = env.bimanual_dex.dexright.get_joint_positions()
        # joint_state: [左臂关节角, 右臂关节角]，长度 60
        joint_state = np.concatenate([joint_pos_L, joint_pos_R])
        
        obs = dict()
        # 机器人当前关节角
        obs['agent_pos'] = joint_state
        # 环境点云：由 env_camera 深度图转换而来
        obs['environment_point_cloud'] = env.env_camera.get_pointcloud_from_depth()
        # 初始衣服点云（保持不变）
        obs['garment_point_cloud'] = env.garment_pcd
        # 可操作性特征（左手当前使用的那一个）
        obs['points_affordance_feature'] = env.points_affordance_feature
        
        # --------------------------- #
        #   2. 通过 Stage_1 策略推理   #
        # --------------------------- #
        # SADP_G 输出形状为 [4, 60] 的关节角序列（4 步，每一步控制 60 个关节）
        action = env.sadp_g_stage_1.get_action(obs)
        
        print("action_shape:",action.shape)
        
        # 对每个时间步（共 4 步）依次施加动作
        for j in range(4):
            
            # 将一帧动作拆分为左臂和右臂的关节角（各 30 维）
            action_L = ArticulationAction(joint_positions=action[j][:30])
            action_R = ArticulationAction(joint_positions=action[j][30:])

            # 将动作应用到机器人
            env.bimanual_dex.dexleft.apply_action(action_L)
            env.bimanual_dex.dexright.apply_action(action_R)
            
            for _ in range(5):    
                env.step()
                
            # 重新采样最新的观测，并告知策略模型当前的 obs，以便其内部维护轨迹等信息
            joint_pos_L = env.bimanual_dex.dexleft.get_joint_positions()
            joint_pos_R = env.bimanual_dex.dexright.get_joint_positions()
            joint_state = np.concatenate([joint_pos_L, joint_pos_R])
            
            obs = dict()
            obs['agent_pos'] = joint_state
            obs['environment_point_cloud'] = env.env_camera.get_pointcloud_from_depth()
            obs['garment_point_cloud'] = env.garment_pcd
            obs['points_affordance_feature'] = env.points_affordance_feature
            
            env.sadp_g_stage_1.update_obs(obs)
    
    # 在阶段之间，短暂增大布料重力（重力 scale=10），让布料快速“落稳”，减少中间漂浮
    env.garment.particle_material.set_gravity_scale(10.0)
    for i in range(200):
        env.step()
    env.garment.particle_material.set_gravity_scale(1.0) 
    
    
    
    # 将左手先移走到远处，避免干扰右手后续抓取和拉拽动作
    env.bimanual_dex.dexleft.dense_step_action(
        target_pos=np.array([-0.6, 0.8, 0.5]),
        target_ori=np.array([0.579, -0.579, -0.406, 0.406]),
        angular_type="quat",
    )

    
    # ====================================================== #
    #                    Stage 2：右手                       #
    #   目标：右手抓取衣服另一侧，对称完成第一步拉拽 / 摊平。 #
    # ====================================================== #
    
    # 为第二阶段构造新的可操作性特征，这里使用第 2 个关键点的特征
    env.points_affordance_feature = normalize_columns(
        np.concatenate([points_similarity[2:3], points_similarity[2:3]], axis=0).T
    )
                
    # 将右手移动到第二个关键点附近
    env.bimanual_dex.dexright.dense_step_action(
        target_pos=manipulation_points[2],
        target_ori=np.array([0.406, -0.406, -0.579, 0.579]),
        angular_type="quat",
    )

    for i in range(20):
        env.step()
    
    # 同样循环 8 次阶段二策略推理
    for i in range(8):
        
        print(f"Stage_2_Step: {i}")

        # 构造观测并通过 Stage_2 策略网络得到动作序列
        joint_pos_L = env.bimanual_dex.dexleft.get_joint_positions()
        joint_pos_R = env.bimanual_dex.dexright.get_joint_positions()
        joint_state = np.concatenate([joint_pos_L, joint_pos_R])
        
        obs = dict()
        obs['agent_pos'] = joint_state
        obs['environment_point_cloud'] = env.env_camera.get_pointcloud_from_depth()
        obs['garment_point_cloud'] = env.garment_pcd
        obs['points_affordance_feature'] = env.points_affordance_feature
        
        action = env.sadp_g_stage_2.get_action(obs)
        
        print("action_shape:",action.shape)
        
        for j in range(4):
            # 同 Stage_1，将每个动作帧拆分成左右臂的关节角并执行
            action_L = ArticulationAction(joint_positions=action[j][:30])
            action_R = ArticulationAction(joint_positions=action[j][30:])

            env.bimanual_dex.dexleft.apply_action(action_L)
            env.bimanual_dex.dexright.apply_action(action_R)
            
            for _ in range(5):    
                env.step()
                
            # 更新观察并缓存给策略
            joint_pos_L = env.bimanual_dex.dexleft.get_joint_positions()
            joint_pos_R = env.bimanual_dex.dexright.get_joint_positions()
            joint_state = np.concatenate([joint_pos_L, joint_pos_R])
            
            obs = dict()
            obs['agent_pos'] = joint_state
            obs['environment_point_cloud'] = env.env_camera.get_pointcloud_from_depth()
            obs['garment_point_cloud'] = env.garment_pcd
            obs['points_affordance_feature'] = env.points_affordance_feature
            
            env.sadp_g_stage_2.update_obs(obs)
    
    # 同样在阶段之间加速布料稳定过程
    env.garment.particle_material.set_gravity_scale(10.0)
    for i in range(200):
        env.step()
    env.garment.particle_material.set_gravity_scale(1.0) 
    
    
    
    # 将右手也暂时移走，给第三阶段双手协同折叠留出空间
    env.bimanual_dex.dexright.dense_step_action(
        target_pos=np.array([0.6, 0.8, 0.5]),
        target_ori=np.array([0.406, -0.406, -0.579, 0.579]),
        angular_type="quat",
    )
    
    # ====================================================== #
    #                 Stage 3：双手协同折叠                   #
    #   目标：双手分别抓取衣服底部两点，将其折到上半部分。   #
    # ====================================================== #     
    
    # 对底部两个点（索引 4 和 5）构造可操作性特征
    env.points_affordance_feature = normalize_columns(points_similarity[4:6].T)   
        
    # 同时移动左右手到对应的两个关键点位置，准备执行折叠动作
    env.bimanual_dex.dense_move_both_ik(
        left_pos=manipulation_points[4], 
        left_ori=np.array([0.579, -0.579, -0.406, 0.406]),
        right_pos=manipulation_points[5],
        right_ori=np.array([0.406, -0.406, -0.579, 0.579]),
    )
    
    for i in range(20):
        env.step()
    
    # 第三阶段循环 12 次，每次仍然会产生 4 帧动作
    for i in range(12):
        
        print(f"Stage_3_Step: {i}")

        # 构造观测并调用 Stage_3 策略网络
        joint_pos_L = env.bimanual_dex.dexleft.get_joint_positions()
        joint_pos_R = env.bimanual_dex.dexright.get_joint_positions()
        joint_state = np.concatenate([joint_pos_L, joint_pos_R])
        
        obs = dict()
        obs['agent_pos'] = joint_state
        obs['environment_point_cloud'] = env.env_camera.get_pointcloud_from_depth()
        obs['garment_point_cloud'] = env.garment_pcd
        obs['points_affordance_feature'] = env.points_affordance_feature

        action = env.sadp_g_stage_3.get_action(obs)
        
        print("action_shape:",action.shape)
        
        for j in range(4):
            # 将动作序列应用到双臂，完成折叠过程中的具体轨迹
            action_L = ArticulationAction(joint_positions=action[j][:30])
            action_R = ArticulationAction(joint_positions=action[j][30:])

            env.bimanual_dex.dexleft.apply_action(action_L)
            env.bimanual_dex.dexright.apply_action(action_R)
            
            for _ in range(5):    
                env.step()
                
            # 更新观测并回传给策略
            joint_pos_L = env.bimanual_dex.dexleft.get_joint_positions()
            joint_pos_R = env.bimanual_dex.dexright.get_joint_positions()
            joint_state = np.concatenate([joint_pos_L, joint_pos_R])
            
            obs = dict()
            obs['agent_pos'] = joint_state
            obs['environment_point_cloud'] = env.env_camera.get_pointcloud_from_depth()
            obs['garment_point_cloud'] = env.garment_pcd
            obs['points_affordance_feature'] = env.points_affordance_feature
            
            env.sadp_g_stage_3.update_obs(obs)
    
    # 第三阶段结束后再让布料稳定一次
    env.garment.particle_material.set_gravity_scale(10.0)
    for i in range(100):
        env.step()
    env.garment.particle_material.set_gravity_scale(1.0)
    
    # 折叠完成后，隐藏双臂，只保留衣服与环境，方便进行结果可视化和评估
    dexleft_prim = prims_utils.get_prim_at_path("/World/DexLeft")
    dexright_prim = prims_utils.get_prim_at_path("/World/DexRight")
    set_prim_visibility(dexleft_prim, False)
    set_prim_visibility(dexright_prim, False)
    
    for i in range(50):
        env.step()

    # 如果需要生成 mp4 视频，这里会将线程采集到的 rgb 序列写成视频文件
    if record_video_flag:
        if not os.path.exists("Data/Fold_Tops_Validation_HALO/vedio"):
            os.makedirs("Data/Fold_Tops_Validation_HALO/vedio")
        env.env_camera.create_mp4(get_unique_filename("Data/Fold_Tops_Validation_HALO/vedio/vedio", ".mp4"))
   
        
    # ----------------------------- #
    #   折叠结果自动评估（success）   #
    # ----------------------------- #
    success = True
    # 使用 GAM 再次在当前点云上采样四个关键点，用于构造评估区域 boundary：
    # boundary = [min_x, max_x, min_y, max_y]
    # 直观理解：用 4 个点确定一个“理想折叠区域”的矩形包围盒
    points, *_ = env.model.get_manipulation_points(pcd, [554, 1540, 1014, 1385])
    boundary = [points[0][0] - 0.05, points[1][0] + 0.05, points[3][1] - 0.1, points[2][1] + 0.1]
    # 获取折叠结束后的衣服点云
    pcd_end, _ = env.garment_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".ply"),
        real_time_watch=False,
    )
    # 使用 Position_Judge 中的 judge_pcd 函数判断折叠质量：
    # - 若在 boundary 区域内的点云分布满足阈值要求（覆盖度/致密度达到阈值），则视为成功
    success = judge_pcd(pcd_end, boundary, threshold=0.12)
    cprint(f"final result: {success}", color="green", on_color="on_green")

    
    # 若处于“批量验证模式”，则将结果写入日志，便于统计成功率
    if validation_flag:
        if not os.path.exists("Data/Fold_Tops_Validation_HALO"):
            os.makedirs("Data/Fold_Tops_Validation_HALO")
        # write into .log file
        with open("Data/Fold_Tops_Validation_HALO/validation_log.txt", "a") as f:
            f.write(f"result:{success}  usd_path:{env.garment.usd_path}  pos_x:{pos[0]}  pos_y:{pos[1]}\n")
        if not os.path.exists("Data/Fold_Tops_Validation_HALO/final_state_pic"):
            os.makedirs("Data/Fold_Tops_Validation_HALO/final_state_pic")
        env.env_camera.get_rgb_graph(save_or_not=True,save_path=get_unique_filename("Data/Fold_Tops_Validation_HALO/final_state_pic/img",".png"))

   
if __name__=="__main__":
    
    # 从命令行解析验证参数（如：是否随机衣服、是否记录视频、训练数据量等）
    args = parse_args_val()
    
    # --------------------------- #
    #     设置默认初始位姿        #
    # --------------------------- #
    pos = np.array([0.0, 0.8, 0.2])
    ori = np.array([0.0, 0.0, 0.0])
    usd_path = None
    
    # 如果开启随机衣服与随机摆放位置：
    # - 随机 x,y（落在一定范围内）
    # - 从 assets_list.txt 中随机选取一个衣服 usd 作为测试对象
    if args.garment_random_flag:
        np.random.seed(int(time.time()))
        x = np.random.uniform(-0.1, 0.1) # changeable
        y = np.random.uniform(0.7, 0.9) # changeable
        pos = np.array([x,y,0.0])
        ori = np.array([0.0, 0.0, 0.0])
        # Base_dir 为整个仓库的根目录
        Base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # assets_list.txt 中保存了所有可用的长袖上衣 usd 路径
        assets_lists = os.path.join(Base_dir, "Model_HALO/GAM/checkpoints/Tops_LongSleeve/assets_list.txt")
        assets_list = []
        with open(assets_lists, "r", encoding='utf-8') as f:
            for line in f:
                clean_line = line.rstrip('\n')
                assets_list.append(clean_line)
        # 随机选择一个衣服 usd 路径作为测试对象
        usd_path = np.random.choice(assets_list)
    
    # 启动一次完整的上衣折叠任务
    FoldTops(
        pos,
        ori,
        usd_path,
        args.ground_material_usd,
        args.validation_flag,
        args.record_video_flag,
        args.training_data_num,
        args.stage_1_checkpoint_num,
        args.stage_2_checkpoint_num,
        args.stage_3_checkpoint_num,
    )
    
    # 若是批量验证模式，单次任务完成后即可关闭仿真；
    # 否则保持 Isaac Sim 运行状态，便于交互式观察结果
    if args.validation_flag:
        simulation_app.close()
    else:
        while simulation_app.is_running():
            simulation_app.update()
    
# 安全起见，在脚本结束前再次调用 close（若已关闭则不影响）
simulation_app.close()

