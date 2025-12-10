from isaacsim import SimulationApp

# 创建 Isaac Sim 仿真应用
# 如果你在服务器上只想无界面运行，可把 False 改成 True
simulation_app = SimulationApp({"headless": False})

# --------------------------- #
# 外部依赖 / 标准库
# --------------------------- #
import os
import sys
import time
import numpy as np
import open3d as o3d
from termcolor import cprint
import threading
import cv2

# --------------------------- #
# Isaac Sim / Omniverse 相关依赖
# --------------------------- #
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
from omni.physx.scripts import deformableUtils,particleUtils,physicsUtils

# --------------------------- #
# 本项目自定义模块
# --------------------------- #
sys.path.append(os.getcwd())
from Env_StandAlone.BaseEnv import BaseEnv
from Env_Config.Garment.Particle_Garment import Particle_Garment
from Env_Config.Garment.Deformable_Garment import Deformable_Garment
from Env_Config.Robot.BimanualDex_Ur10e import Bimanual_Ur10e
from Env_Config.Camera.Recording_Camera import Recording_Camera
from Env_Config.Room.Real_Ground import Real_Ground
from Env_Config.Utils_Project.Code_Tools import get_unique_filename, normalize_columns
from Env_Config.Utils_Project.Point_Cloud_Manip import compute_similarity, get_surface_vertices
from Env_Config.Utils_Project.Parse import parse_args_record
from Env_Config.Utils_Project.Position_Judge import judge_pcd
from Env_Config.Room.Object_Tools import set_prim_visible_group, delete_prim_group
from Model_HALO.GAM.GAM_Encapsulation import GAM_Encapsulation


"""
FoldTops_Env
这个类封装了一整套「长袖上衣折叠」任务环境：
- 搭建场景（地面、衣服、双臂机器人、相机）
- 通过 GAM 模型从点云中预测抓取/操作点
- 控制双臂完成折叠动作
- 根据结果打分、保存数据和视频
"""

VERTEX_WORLD_SCALE = np.array([0.0085, 0.0085, 0.0085])

class FoldTops_Env(BaseEnv):
    def __init__(
        self, 
        pos:np.ndarray=None, 
        ori:np.ndarray=None, 
        usd_path:str=None, 
        ground_material_usd:str=None,
        record_video_flag:bool=False, 
    ):
        # 初始化基础环境（加载 world、场景、记录工具等）
        super().__init__()
        
        # ------------------------------------ #
        # ---        Add Env Assets        --- #
        # ---   环境中固定的物体和组件加载   --- #
        # ------------------------------------ #
        self.ground = Real_Ground(
            self.scene, 
            visual_material_usd = ground_material_usd,
            # you can use materials in 'Assets/Material/Floor' to change the texture of ground.
        )
        
        # load garment
        # 加载衣物（粒子布料形式的上衣）
        self.garment = Particle_Garment(
            self.world, 
            pos=np.array([0, 3.0, 0.6]),
            ori=np.array([0.0, 0.0, 0.0]),
            # 如果没有传入 usd_path，则使用默认的一件长袖上衣
            usd_path=os.getcwd() + "/" + "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_018/TCLC_018_obj.usd" if usd_path is None else usd_path,
            contact_offset=0.012,             
            rest_offset=0.010,                
            particle_contact_offset=0.012,    
            fluid_rest_offset=0.010,
            solid_rest_offset=0.010,
        )
        # Here are some example garments you can try:
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Jacket032/TCLC_Jacket032_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Jacket152/TCLC_Jacket152_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Top566/TCLC_Top566_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Top584/TCLC_Top584_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_top118/TCLC_top118_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Top476/TCLC_Top476_obj.usd",
        # "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Top030/TCLC_Top030_obj.usd",  

        # 加载双臂机器人（左右各一只 DexHand + UR10e）
        # 这里手的pose是固定的，后面可能需要改
        self.bimanual_dex = Bimanual_Ur10e(
            self.world,
            dexleft_pos=np.array([-0.8, 0.0, 0.5]),
            dexleft_ori=np.array([0.0, 0.0, 0.0]),
            dexright_pos=np.array([0.8, 0.0, 0.5]),
            dexright_ori=np.array([0.0, 0.0, 0.0]),
        )

        # 加载两个相机：
        # 1. garment_camera：主要看衣服，用于分割出衣服点云
        # 2. env_camera：俯视环境，用于录制可视化视频
        self.garment_camera = Recording_Camera(
            camera_position=np.array([0.0, 1.0, 6.75]), 
            camera_orientation=np.array([0, 90.0, 90.0]),
            prim_path="/World/garment_camera",
        )
        
        self.env_camera = Recording_Camera(
            camera_position=np.array([0.0, 4.0, 6.0]),
            camera_orientation=np.array([0, 60, -90.0]),
            prim_path="/World/env_camera",
        )
        
        # 运行时会存放「当前衣物点云」和「当前点的可操作性特征」
        self.garment_pcd = None
        self.points_affordance_feature = None
        
        # 加载 GAM 模型（长袖上衣类别），用于从点云预测关键操作点
        self.model = GAM_Encapsulation(catogory="Tops_LongSleeve")   
        
        # ------------------------------------ #
        # --- Initialize World to be Ready --- #
        # ------------------------------------ #
        # 初始化 world（把物体/机器人放到默认状态）
        self.reset()
        
        # 把衣物移动到指定位置（pos）和姿态（ori）
        self.garment.set_pose(pos=np.array([pos[0], pos[1], 0.2]), ori=ori)
        self.position = [pos[0], pos[1], 0.2]
        self.orientation = ori
        
        # 设置衣物相机：只分割 /World/Garment/garment 这一类 prim
        self.garment_camera.initialize(
            segment_pc_enable=True, 
            segment_prim_path_list=[
                "/World/Garment/garment"
            ]
        )
        
        # 环境相机：打开深度图，用于生成环境点云
        self.env_camera.initialize(depth_enable=True)
        
        # 如需录制视频，会起一个线程不断收集 env_camera 的 RGB 帧
        #（主线程只负责控制机器人和仿真步进）
        if record_video_flag:
            self.thread_record = threading.Thread(target=self.env_camera.collect_rgb_graph_for_video)
            self.thread_record.daemon = True
        
        # 让双手张开作为初始状态
        self.bimanual_dex.set_both_hand_state("open", "open")

        # 先仿真 100 步，让布料和机器人都「稳定」下来
        for i in range(100):
            self.step()
            
        cprint("----------- World Configuration -----------", color="magenta", attrs=["bold"])
        cprint(f"usd_path: {usd_path}", "magenta")
        cprint(f"pos_x: {pos[0]}", "magenta")
        cprint(f"pos_y: {pos[1]}", "magenta")
        cprint("----------- World Configuration -----------", color="magenta", attrs=["bold"])
        
        cprint("World Ready!", "green", "on_green")
    
    def record_callback(self, step_size):
        """
        这个函数会在数据采集时被 BaseEnv 的 recorder 调用：
        - 每隔 5 步记录一次：
          - 双臂关节状态
          - 环境 RGB 图像
          - 环境点云
          - 当前衣物点云和可操作性特征
        """

        if self.step_num % 5 == 0:
        
            # 读取左右机械臂的关节角
            joint_pos_L = self.bimanual_dex.dexleft.get_joint_positions()
            joint_pos_R = self.bimanual_dex.dexright.get_joint_positions()
            
            joint_state = np.array([*joint_pos_L, *joint_pos_R])

            # 获取环境相机的 RGB 图像
            rgb = self.env_camera.get_rgb_graph(save_or_not=False)

            # 通过深度图生成环境点云
            point_cloud = self.env_camera.get_pointcloud_from_depth(
                show_original_pc_online=False,
                show_downsample_pc_online=False,
            )
            
            # 存入缓冲区，之后统一写入 .npz
            self.saving_data.append({ 
                "joint_state": joint_state,
                "image": rgb,
                "env_point_cloud": point_cloud,
                "garment_point_cloud":self.garment_pcd,# 这里的garment_pcd和points_affordance_feature是不变的
                "points_affordance_feature": self.points_affordance_feature,
            })
        
        self.step_num += 1


def FoldTops(pos, ori, usd_path, ground_material_usd, record_video_flag, data_collection_flag=True):
    """
    核心任务流程：给定衣物初始位姿和材质，执行一整套「折叠上衣」动作。
    参数：
    - pos, ori: 衣服初始位置、姿态
    - usd_path: 衣服模型路径（未指定则用默认）
    - ground_material_usd: 地面材质
    - data_collection_flag: 是否进行数据采集（保存 npz / log 等）
    - record_video_flag: 是否录制视频（由 env_camera 线程生成 mp4）
    """
    
    # 创建环境（其中会完成场景搭建 + 初始仿真）
    env = FoldTops_Env(pos, ori, usd_path, ground_material_usd, record_video_flag)
    
    # 如果需要录制视频，启动 RGB 采集线程
    if record_video_flag:
        env.thread_record.start()
    
    def _get_world_garment_vertices():
        vertices = env.garment.get_vertice_positions()
        vertices = vertices * VERTEX_WORLD_SCALE
        vertices += env.garment.get_garment_center_pos()
        return vertices
    
    def _refresh_manipulation_points_from_vertices():
        env.manipulation_points = _get_world_garment_vertices()[env.manipulation_vertex_indices]
        env.manipulation_points[0:4, 2] = 0.02
        env.manipulation_points[4:, 2] = 0.0
        return env.manipulation_points
    
    def _capture_stage_state():
        pcd, _ = env.garment_camera.get_point_cloud_data_from_segment(
            save_or_not=False,
            save_path=get_unique_filename("data", extension=".ply"),
            real_time_watch=False,
        )
        env.garment_pcd = pcd
        _refresh_manipulation_points_from_vertices()
        return env.manipulation_points.copy()

    def _capture_top_image(stage_name: str):
        """
        使用与 Deformable_Tops_Collect.py 一致的方式拍照：
        - 只保留衣服（临时隐藏左右机械臂）
        - 从 garment_camera 获取 RGB，并保存到本地
        """
        # 临时隐藏左右机械臂，只显示衣服和地面
        set_prim_visible_group(
            prim_path_list=["/World/DexLeft", "/World/DexRight"],
            visible=False,
        )
        # 让仿真前进几步，保证可见性状态已经在渲染中生效
        for _ in range(50):
            env.step()

        # 获取顶部（衣物）相机 RGB 图像
        rgb = env.garment_camera.get_rgb_graph(save_or_not=False)
        
        # 拍完后把机械臂重新显示出来，避免影响后续可视化
        set_prim_visible_group(
            prim_path_list=["/World/DexLeft", "/World/DexRight"],
            visible=True,
        )
        for _ in range(5):
            env.step()
        
        # 保存目录（如不存在则自动创建）
        save_dir = "Data/Fold_Tops/stage_images"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 使用 get_unique_filename 生成不重名的文件路径
        prefix = os.path.join(save_dir, f"{stage_name}_img")
        output_image_path = get_unique_filename(prefix, extension=".png")
        
        # 注意 Isaac Sim 输出为 RGB，需要转为 BGR 再用 cv2 保存
        cv2.imwrite(output_image_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    
    def _resolve_point_reference(ref):
        if ref is None:
            return None
        if isinstance(ref, dict):
            if "point" in ref:
                return ref["point"]
            if "index" in ref:
                return env.manipulation_points[ref["index"]]
        if isinstance(ref, int):
            return env.manipulation_points[ref]
        return ref  # assume直接传入坐标
    
    def _build_similarity_columns(pcd, refs):
        columns = []
        for ref in refs:
            point = _resolve_point_reference(ref)
            if point is None:
                columns.append(np.zeros((len(pcd), 1)))
            else:
                columns.append(compute_similarity(pcd, point))
        return np.hstack(columns) if columns else np.zeros((len(pcd), 0))
    
    def _update_stage_feature(affordance_refs, goal_refs):
        affordance_block = _build_similarity_columns(env.garment_pcd, affordance_refs)
        if affordance_block.size:
            affordance_block = normalize_columns(affordance_block)
        goal_block = _build_similarity_columns(env.garment_pcd, goal_refs)
        if affordance_block.size and goal_block.size:
            env.points_affordance_feature = np.hstack([affordance_block, goal_block])
        elif affordance_block.size:
            env.points_affordance_feature = affordance_block
        else:
            env.points_affordance_feature = goal_block
        return env.points_affordance_feature
    
    # -------- 获取初始衣物点云 --------
    # 把双臂隐藏，只看衣物本身的点云
    # 对一组给定路径的 prim 统一设置“可见 / 不可见”
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight"],
        visible=False,
    )
    for i in range(50):
        env.step()
        
    # 利用衣物相机，对分割出来的衣物生成点云
    pcd, color = env.garment_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".ply"),
        real_time_watch=False,
    )
    env.garment_pcd = pcd
    
    # unhide
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight"],
        visible=True,
    )
    for i in range(50):
        env.step()
    
    # 通过 GAM 模型，从点云中选出 6 个关键操作点：
    # index_list 里对应的是预先标好的 6 个语义点（如左右袖口、下摆等）
    manipulation_points, indices, _ = env.model.get_manipulation_points(
        input_pcd=pcd,
        index_list=[957, 501, 1902, 448, 1196, 422],
    )

    garment_vertices_world = _get_world_garment_vertices()
    _, garment_indices = get_surface_vertices(garment_vertices_world, env.garment_pcd)
    env.manipulation_vertex_indices = garment_indices[indices]
    _refresh_manipulation_points_from_vertices()
    
    # ---------------------- left hand ---------------------- #
    # 阶段 1：左手负责整理左袖子
    manipulation_points = _capture_stage_state()
    
    # 估算左袖提升高度：与袖中点距离的一部分，上限 0.3
    left_sleeve_height = min(np.linalg.norm(manipulation_points[0][:2] - manipulation_points[3][:2]), 0.3)
    # 左手最终的目标是「抬到第二个关键点上方」
    lift_point_1 = np.array([manipulation_points[0][0], manipulation_points[0][1], left_sleeve_height])
    lift_point_2 = np.array([manipulation_points[1][0], manipulation_points[1][1], left_sleeve_height])
    # 初始就把 affordance（抓取点）和 goal（目标点）编码成高斯衰减特征：左手 goal 是 lift_point_2，右手为空
    _update_stage_feature([0, None], [{"index": 1}, None])
    
    # 基于inverse kinematics，ori固定，应该怎么选择这个值？
    env.bimanual_dex.dexleft.dense_step_action(target_pos=manipulation_points[0], target_ori=np.array([0.579, -0.579, -0.406, 0.406]), angular_type="quat")
    
    # 若采集数据，则在抓取前多走几步并记录第 1 阶段
    if data_collection_flag:
        for i in range(20):
            env.step()
        env.record(task_name="Fold_Tops", stage_index=1)
    
    # 左手闭合 / 抓住布料
    env.bimanual_dex.set_both_hand_state(left_hand_state="close", right_hand_state="None")

    env.bimanual_dex.dexleft.dense_step_action(target_pos=lift_point_1, target_ori=np.array([0.579, -0.579, -0.406, 0.406]), angular_type="quat")
    
    # 再移动到第二个点上方，形成一个「绕袖子提拉」的动作
    lift_point_2 = np.array([manipulation_points[1][0], manipulation_points[1][1], left_sleeve_height])
    
    env.bimanual_dex.dexleft.dense_step_action(target_pos=lift_point_2, target_ori=np.array([0.579, -0.579, -0.406, 0.406]), angular_type="quat")

    # 放开左手
    env.bimanual_dex.set_both_hand_state(left_hand_state="open", right_hand_state="None")
    
    if data_collection_flag:
        env.stop_record()
    
    # 暂时放大重力，让布料自然下落稳定到桌面
    env.garment.particle_material.set_gravity_scale(10.0)
    for i in range(200):
        env.step()
    env.garment.particle_material.set_gravity_scale(1.0)


    # 把左手收回到一侧的「休息位」
    env.bimanual_dex.dexleft.dense_step_action(
        target_pos=np.array([-0.6, 0.8, 0.5]),
        target_ori=np.array([0.579, -0.579, -0.406, 0.406]),
        angular_type="quat",
    )
    # 若开启数据采集，则使用顶部相机记录左袖整理完成后的环境状态
    if data_collection_flag:
        _capture_top_image("stage1_left_sleeve")
    
    # --------------------- right hand --------------------- #
    # 阶段 2：右手负责整理右袖子
    manipulation_points = _capture_stage_state()
    
    # 估算右袖提升高度
    right_sleeve_height = min(np.linalg.norm(manipulation_points[2][:2] - manipulation_points[1][:2]), 0.3)
    lift_point_1 = np.array([manipulation_points[2][0], manipulation_points[2][1], right_sleeve_height])
    lift_point_2 = np.array([manipulation_points[3][0], manipulation_points[3][1], right_sleeve_height])
    # 初始就编码右手的 affordance（第 2 个关键点）和 goal（lift_point_2），左手保持空
    _update_stage_feature([None, 2], [None, {"index": 3}])
            
    env.bimanual_dex.dexright.dense_step_action(target_pos=manipulation_points[2], target_ori=np.array([0.406, -0.406, -0.579, 0.579]), angular_type="quat")

    # 若采集数据，则记录第 2 阶段
    if data_collection_flag:
        for i in range(20):
            env.step()
        env.record(task_name="Fold_Tops", stage_index=2)
            
    # 右手闭合 / 抓住右袖
    env.bimanual_dex.set_both_hand_state(left_hand_state="None", right_hand_state="close")
    
    env.bimanual_dex.dexright.dense_step_action(target_pos=lift_point_1, target_ori=np.array([0.406, -0.406, -0.579, 0.579]), angular_type="quat")
    
    lift_point_2 = np.array([manipulation_points[3][0], manipulation_points[3][1], right_sleeve_height])
    
    env.bimanual_dex.dexright.dense_step_action(target_pos=lift_point_2, target_ori=np.array([0.406, -0.406, -0.579, 0.579]), angular_type="quat")

    env.bimanual_dex.set_both_hand_state(left_hand_state="None", right_hand_state="open")
    
    if data_collection_flag:
        env.stop_record()
    
    # 同样，增加重力让布料自然归位
    env.garment.particle_material.set_gravity_scale(10.0)
    for i in range(200):
        env.step()
    env.garment.particle_material.set_gravity_scale(1.0) 



    # 把右手收回到另一侧的「休息位」
    env.bimanual_dex.dexright.dense_step_action(
        target_pos=np.array([0.6, 0.8, 0.5]),
        target_ori=np.array([0.406, -0.406, -0.579, 0.579]),
        angular_type="quat",
    )
    # 若开启数据采集，则使用顶部相机记录右袖整理完成后的环境状态
    # if data_collection_flag:
    #     _capture_top_image("stage2_right_sleeve")
    
    # --------------------- bottom-top --------------------- #    
    # 阶段 3：双手同时从下摆往上折
    manipulation_points = _capture_stage_state()
    
    
    # 计算从下摆到肩部（大概是 manipulation_points[3]）的高度差，
    # 用于决定向上抬起的距离
    lift_height = manipulation_points[3][1] - manipulation_points[4][1]
    
    # print("lift_height: ", lift_height)
    
    # 把下摆往上抬到一半高度
    lift_point_1 = np.array([manipulation_points[4][0], manipulation_points[4][1], lift_height/2])
    lift_point_2 = np.array([manipulation_points[5][0], manipulation_points[5][1], lift_height/2])
    
    # 再往上（肩部附近）推一点，使得衣身覆盖在主体上
    push_point_1 = np.array([manipulation_points[3][0], manipulation_points[3][1]+0.1, min(lift_height/2, 0.2)])
    push_point_2 = np.array([manipulation_points[1][0], manipulation_points[1][1]+0.1, min(lift_height/2, 0.2)])
    # 第 3 阶段：在动作开始前一次性把 affordance（下摆点 4/5）和 goal（push_point_1/2）编码好
    _update_stage_feature([4, 5], [{"index": 3}, {"index": 1}])
   
    env.bimanual_dex.dense_move_both_ik(
        left_pos=manipulation_points[4], 
        left_ori=np.array([0.579, -0.579, -0.406, 0.406]),
        right_pos=manipulation_points[5],
        right_ori=np.array([0.406, -0.406, -0.579, 0.579]),
    )

    # 若采集数据，则记录第 3 阶段
    if data_collection_flag:
        for i in range(20):
            env.step()
        env.record(task_name="Fold_Tops", stage_index=3)
    
    # 双手夹住下摆两侧
    env.bimanual_dex.set_both_hand_state(left_hand_state="close", right_hand_state="close")
    
    env.bimanual_dex.dense_move_both_ik(
        left_pos=lift_point_1,
        left_ori=np.array([0.579, -0.579, -0.406, 0.406]),
        right_pos=lift_point_2,
        right_ori=np.array([0.406, -0.406, -0.579, 0.579]),
    )
    
    env.bimanual_dex.dense_move_both_ik(
        left_pos=push_point_1,
        left_ori=np.array([0.579, -0.579, -0.406, 0.406]),
        right_pos=push_point_2,
        right_ori=np.array([0.406, -0.406, -0.579, 0.579]),
    )
    
    # 放手
    env.bimanual_dex.set_both_hand_state(left_hand_state="open", right_hand_state="open")
    
    if data_collection_flag:
        env.stop_record()
    
    # 让布料再次自然下落、稳定形态
    env.garment.particle_material.set_gravity_scale(10.0)
    for i in range(100):
        env.step()
    env.garment.particle_material.set_gravity_scale(1.0)

    # 把双臂在可视化中隐藏，便于只观察最终衣服形态
    dexleft_prim = prims_utils.get_prim_at_path("/World/DexLeft")
    dexright_prim = prims_utils.get_prim_at_path("/World/DexRight")
    set_prim_visibility(dexleft_prim, False)
    set_prim_visibility(dexright_prim, False)
    
    for i in range(50):
        env.step()   
        
    # -------- 评估折叠效果 --------
    # 使用另外一组语义点，围出一个「理想折叠区域」
    success = True
    points, *_ = env.model.get_manipulation_points(pcd, [554, 1540, 1014, 1385])
    boundary = [points[0][0]-0.05, points[1][0]+0.05, points[3][1]-0.1, points[2][1]+0.1]

    # 再次从分割点云中获取最终衣服点云
    pcd_end, _ = env.garment_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".ply"),
        real_time_watch=False,
    )
    # judge_pcd 会判断大部分衣服点是否落在 boundary 内部，从而给出成功/失败
    success = judge_pcd(pcd_end, boundary, threshold=0.12)
    cprint(f"final result: {success}", color="green", on_color="on_green")
    
    # 如果想要生成 mp4 视频（类似 gif 效果）
    if record_video_flag and success:
        if not os.path.exists("Data/Fold_Tops/video"):
            os.makedirs("Data/Fold_Tops/video")
        env.env_camera.create_mp4(get_unique_filename("Data/Fold_Tops/video/video", ".mp4"))

    if data_collection_flag:
        # write into .log file
        with open("Data/Fold_Tops/data_collection_log.txt", "a") as f:
            f.write(f"result:{success}  usd_path:{env.garment.usd_path}  pos_x:{pos[0]}  pos_y:{pos[1]}\n")
        if success:
            env.record_to_npz()
            if not os.path.exists("Data/Fold_Tops/final_state_pic"):
                os.makedirs("Data/Fold_Tops/final_state_pic")
            env.env_camera.get_rgb_graph(save_or_not=True,save_path=get_unique_filename("Data/Fold_Tops/final_state_pic/img",".png"))

   
if __name__ == "__main__":
    
    args = parse_args_record()

    # ---------------------- #
    # 1. 决定初始位姿 & 地面材质（单次运行）
    # ---------------------- #
    # 默认固定一个初始位姿和默认地面
    pos = np.array([0.0, 0.8, 0.2])
    ori = np.array([0.0, 0.0, 0.0])
    ground_material_usd = args.ground_material_usd if hasattr(args, "ground_material_usd") else None

    # 如果希望环境随机：
    # - 衣服初始位置随机（桌面上的 x, y）
    # - 若未显式指定 ground_material_usd，则从 Floor 材质库中随机选一块地面贴图
    if args.env_random_flag:
        np.random.seed(int(time.time()))
        # 随机初始位姿
        x = np.random.uniform(-0.1, 0.1)  # changeable
        y = np.random.uniform(0.7, 0.9)   # changeable
        pos = np.array([x, y, 0.0])

        # 随机地面材质（仅当用户没有在命令行里指定 ground_material_usd 时）
        if ground_material_usd is None:
            floor_dir = os.path.join(os.getcwd(), "Assets/Material/Floor")
            try:
                candidates = [
                    os.path.join(floor_dir, f)
                    for f in os.listdir(floor_dir)
                    if f.endswith(".usd")
                ]
                if len(candidates) > 0:
                    ground_material_usd = np.random.choice(candidates)
            except Exception as e:
                cprint(f"[Warning] Failed to randomize ground material: {e}", "yellow")

    # ---------------------- #
    # 2. 决定本次要用哪一件衣服（只跑 1 次）
    # ---------------------- #
    usd_path = None

    # 优先级 1：命令行直接指定 --usd_path
    if hasattr(args, "usd_path") and args.usd_path is not None:
        # 既支持绝对路径，也支持相对于工程根目录的相对路径
        if os.path.isabs(args.usd_path):
            usd_path = args.usd_path
        else:
            usd_path = os.path.join(os.getcwd(), args.usd_path)

    # 优先级 2：如果有 garment_random_flag（用于验证脚本的别名），且被置为 True，
    #           则从训练集列表中随机选 1 件（在当前记录脚本中通常不用）
    elif hasattr(args, "garment_random_flag") and args.garment_random_flag:
        Base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assets_lists = os.path.join(
            Base_dir,
            "Model_HALO/GAM/checkpoints/Tops_LongSleeve/assets_train.txt",
        )
        assets_list = []
        with open(assets_lists, "r", encoding="utf-8") as f:
            for line in f:
                clean_line = line.strip()
                if clean_line:
                    assets_list.append(clean_line)

        if len(assets_list) > 0:
            np.random.seed(int(time.time()))
            garment_rel_path = np.random.choice(assets_list)
            usd_path = os.path.join(os.getcwd(), garment_rel_path)

    # 否则：保持 usd_path = None，使用 FoldTops_Env 中的默认那件长袖上衣

    # ---------------------- #
    # 3. 单次运行 FoldTops
    # ---------------------- #
    try:
        FoldTops(
            pos,
            ori,
            usd_path,
            ground_material_usd,
            args.data_collection_flag,
            args.record_video_flag,
        )
    except ValueError as e:
        # 专门捕获 furthest_point_sampling 的「空点云」报错：
        # 把这一轮视为无效轮次，打印提示但不让程序崩溃。
        if "furthest_point_sampling 收到空点云" in str(e):
            cprint(
                "[Warning] 本轮 FoldTops 因上游返回空点云而被跳过（已安全结束本轮）。",
                "yellow",
            )
        else:
            # 其它 ValueError 依旧抛出，避免掩盖真正的 bug
            raise

    if args.data_collection_flag:
        # 数据收集模式：当前这条轨迹跑完就直接关闭 app
        simulation_app.close()
    else:
        # 交互 / 可视化模式：保持仿真窗口，直到用户手动关闭
        while simulation_app.is_running():
            simulation_app.update()
        simulation_app.close()
