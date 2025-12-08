from isaacsim import SimulationApp

# 启动 Isaac Sim 应用
# - headless=False 表示启用图形界面，便于观察仿真效果与调试
simulation_app = SimulationApp({"headless": False})

# ==============================
# 加载外部 Python 依赖
# ==============================
import os
import sys
import time
import numpy as np
import open3d as o3d
from termcolor import cprint
import threading
import cv2

# ==============================
# 加载 Isaac / USD 相关依赖
# ==============================
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

# ==============================
# 加载本项目自定义模块
# ==============================
sys.path.append(os.getcwd())
from Env_StandAlone.BaseEnv import BaseEnv
from Env_Config.Garment.Particle_Garment import Particle_Garment
from Env_Config.Garment.Deformable_Garment import Deformable_Garment
from Env_Config.Robot.BimanualDex_Ur10e import Bimanual_Ur10e
from Env_Config.Camera.Recording_Camera import Recording_Camera
from Env_Config.Room.Real_Ground import Real_Ground
from Env_Config.Utils_Project.Code_Tools import get_unique_filename, normalize_columns
from Env_Config.Utils_Project.Parse import parse_args_record
from Env_Config.Utils_Project.Position_Judge import judge_pcd
from Env_Config.Room.Object_Tools import set_prim_visible_group, delete_prim_group
from Model_HALO.GAM.GAM_Encapsulation import GAM_Encapsulation
from Env_Config.Utils_Project.Point_Cloud_Manip import furthest_point_sampling, get_surface_vertices

import json


class FoldTops_Env(BaseEnv):
    """
    长袖上衣折叠环境：
    - 继承自 BaseEnv，负责创建场景（地面、衣物、相机、机器人）并管理仿真 world
    - 提供 apply() 用于重置环境、加载新的衣物和地面材质
    - 成员中保存了各个语义部位（袖口、领口、下摆、腋下、肩、腰等）的模板顶点索引，
      用于后续通过 GAM 模型在任意姿态下追踪对应区域。
    """
    def __init__(self):
        # -------- 场景基础环境 --------
        # 调用 BaseEnv.__init__() 完成 Isaac Sim world、场景等基础初始化
        super().__init__()

        # -------- 机器人：双臂 UR10e --------
        # 这里实例化一个双臂机械臂环境：
        # - 左手基坐标大致在 (-0.8, 0.0, 0.5)
        # - 右手基坐标大致在 ( 0.8, 0.0, 0.5)
        # 具体封装见 Env_Config.Robot.BimanualDex_Ur10e
        self.bimanual_dex = Bimanual_Ur10e(
            self.world,
            dexleft_pos=np.array([-0.8, 0.0, 0.5]),
            dexleft_ori=np.array([0.0, 0.0, 0.0]),
            dexright_pos=np.array([0.8, 0.0, 0.5]),
            dexright_ori=np.array([0.0, 0.0, 0.0]),
        )

        # -------- 录制相机 --------
        # 用于采集 RGB / 深度 / 分割点云等信息
        # 相机大致位于衣物上方，朝下俯视
        self.garment_camera = Recording_Camera(
            camera_position=np.array([0.0, 0.8, 6]), 
            camera_orientation=np.array([0, 90.0, 90.0]),
            prim_path="/World/garment_camera",
        )
        
        # -------- GAM 模型 --------
        # GAM_Encapsulation 用于从点云中根据模板索引找到语义一致的关键点，
        # 此处类别为长袖上衣 Tops_LongSleeve
        self.model = GAM_Encapsulation(catogory="Tops_LongSleeve")
        
        # -------- 运行时变量（后续在 apply / FoldTops 中填充） --------
        self.garment = None
        self.ground = None
        self.garment_pcd = None
        self.points_affordance_feature = None

        # -------- 语义模板顶点索引（基于“模板网格”编号） --------
        # 这些索引在模板网格上手工标注好，代表对应部位附近的一簇顶点。
        # 在真实仿真中，GAM 模型会从当前点云中找出“最相似”的点集合，
        # 再通过最近邻对应回当前仿真的变形网格，从而得到语义上对齐的关键点位置。

        # 左右袖口
        self.left_cuff = [8,45,70,144,270,296,351,386,409,522,636,675,710,785,830,957,1091,1144,1224,1470,1475,1635,1761,1762,1763,1839,1895,1959,1990,2027]
        self.right_cuff = [1,38,120,153,233,302,362,364,394,425,564,571,599,775,878,954,977,1017,1036,1039,1043,1134,1139,1176,1202,1321,1506,1629,1709,1755,1902,1916]

        # 左右及中间领口
        self.left_collar = [132,163,288,305,519,548,573,686,1003,1019,1125,1148,1329,1334,1455,1592,1690]
        self.right_collar = [76,135,149,355,454,660,740,970,1069,1078,1096,1114,1178,1261,1264,1287,1799,1906,2008]
        self.center_collar = [10,150,340,404,465,584,780,811,866,887,1006,1010,1093,1118,1333,1519,1537,1576,1589,1597,1713,1723,1980,2004,2034]

        # 左右及中间下摆
        self.left_hem = [7,106,147,231,236,382,387,532,563,609,673,958,1082,1146,1161,1196,1241,1418,1661,1687,1697,1760,2031]
        self.right_hem = [50,152,201,369,422,471,567,585,873,1018,1081,1220,1308,1341,1370,1605,1615,1751,2025]
        self.center_hem = [46,128,137,214,223,326,327,395,439,597,623,726,809,983,1021,1063,1073,1206,1237,1371,1376,1509,1548,1566,1567,1568,1580,1764,1800]

        # 左腋下（下面两行是历史版本/备选索引，最终又被第三行覆盖）
        # self.left_armpit = [...]
        # self.left_armpit = [...]
        # 注意：最后这一行才是实际生效的 left_armpit 模板
        self.left_armpit = [12, 78, 99, 234, 307, 432, 435, 474, 493, 556, 593, 649, 687, 745, 923, 930, 943, 1059, 1123, 1228, 1239, 1313, 1346, 1474, 1508, 1872, 1877, 1909, 1948, 1975, 1979, 1997]

        # 右腋下
        self.right_armpit = [60,79,164,170,172,228,348,458,467,535,543,637,654,824,972,986,1016,1066,1136,1143,1172,1246,1262,1382,1477,1479,1526,1533,1619,1677,1692,1837,1865,1922,1952,1962,1965,2013,2032,2037,2041]

        # 左右肩部
        self.left_shoulder = [4,89,113,119,175,297,374,455,483,509,512,555,627,630,757,791,802,864,876,990,1020,1044,1064,1212,1219,1266,1284,1318,1335,1410,1428,1478,1559,1601,1747,1773,1831,1957,2040]
        self.right_shoulder = [2,95,98,182,215,310,363,418,454,502,518,523,660,669,743,750,752,789,793,907,918,941,963,1103,1150,1233,1240,1253,1270,1281,1286,1305,1399,1547,1570,1689,1770,1781,1799,1850,1854,1982,2029]

        # 左右腰线
        self.left_waist =  [14,64,110,143,241,357,398,410,490,758,760,766,837,945,974,1005,1132,1173,1244,1247,1272,1468,1481,1574,1683,1759,1926,1928,2046]
        self.right_waist = [68,117,202,315,323,459,537,633,648,659,708,720,855,919,944,1166,1256,1291,1296,1494,1512,1587,1598,1633,1836,1898,1920,1924,1972,1995,2030]
        # self.left_cuff = [70,710,1470,1635,1763,1990]
        # self.right_cuff = [120,233,302,954,1139,1755]
        # self.left_collar = [132,457,1148,1702,2011]
        # self.right_collar = [513,728,1621,1905,1919]
        # self.center_collar = [173,625,1114,1517,1728]
        # self.left_hem = [147,382,958,1146,1661]
        # self.right_hem = [16,201,585,1018,1751]
        # self.center_hem = [46,809,983,1073,1376,1566,1567]
        # self.left_armpit = [99,307,1614,1665]
        # self.right_armpit = [94,198,1159,1489]
        # self.left_shoulder = [455,990,1044,1212,1266]
        # self.right_shoulder = [215,363,793,1281,1850]
        # self.left_waist =  [112,357,974,1173]
        # self.right_waist = [459,1587,1898,1950]
    def cleanup(self):
        """清理上一次仿真中创建的衣物和地面对象，为重新加载做准备"""
        # Delete previous garment if it exists
        if hasattr(self, 'garment') and self.garment is not None:
            try:
                # Delete the garment prim and its components
                delete_prim_group([self.garment.garment_prim_path])
                delete_prim_group([self.garment.particle_system_path])
                delete_prim_group([self.garment.particle_material_path])
                # if hasattr(self.garment, 'visual_material_path'):
                #     delete_prim_group([self.garment.visual_material_path])
            except:
                pass
            self.garment = None
        
        # Delete previous ground if it exists
        if hasattr(self, 'ground') and self.ground is not None:
            try:
                # Delete ground prims
                delete_prim_group(["/World/defaultGroundPlane"])
            except:
                pass
            self.ground = None
        
        # Reset world to clear any remaining objects
        # self.reset()
        
        # Step a few times to ensure cleanup
        # for i in range(10):
        #     self.step()
        
    def apply(self, pos:np.ndarray=None,
              ori:np.ndarray=None,
              ground_material_usd:str=None,
              usd_path:str=None):
        """
        将当前环境重置为新的场景配置：
        - 删除旧的衣物和地面；
        - 加载新的地面材质和衣物 USD；
        - 初始化相机与机器人手势；
        - 将衣物移动到指定位置/姿态。

        参数：
        - pos: 目标衣物中心世界坐标 (x, y, z)
        - ori: 目标衣物欧拉角姿态
        - ground_material_usd: 地面材质 usd 路径
        - usd_path: 衣物资产 usd 路径
        """
        
        # Clean up previous objects first
        self.cleanup()
        
        # 1) 先创建新的地面
        self.ground = Real_Ground(
                self.scene, 
                visual_material_usd = ground_material_usd,
                # you can use materials in 'Assets/Material/Floor' to change the texture of ground.
            )
            
        # 2) 加载衣物（粒子布料）对象
        self.garment = Particle_Garment(
                self.world, 
                pos=np.array([0, 3.0, 0.6]),
                ori=np.array([0.0, 0.0, 0.0]),
                # 若未给定 usd_path，则使用默认长袖衣物资产
                usd_path="Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_018/TCLC_018_obj.usd" if usd_path is None else usd_path,
                contact_offset=0.012,             
                rest_offset=0.010,                
                particle_contact_offset=0.012,    
                fluid_rest_offset=0.010,
                solid_rest_offset=0.010,
                visual_material_usd = np.random.choice(["Assets/Material/Garment/linen_Pumpkin.usd", "Assets/Material/Garment/linen_Blue.usd"])
            )
        
        # 追踪点云与 GAM 特征相关缓存重置
        self.garment_pcd = None
        self.points_affordance_feature = None

        # 3) 初始化 world（内含 BaseEnv 中的默认初始化逻辑）
        self.reset()
        
        # 4) 将衣物移动到给定的 pos / ori
        self.garment.set_pose(pos=np.array([pos[0], pos[1], 0.2]), ori=ori)
        self.position = [pos[0], pos[1], 0.2]
        self.orientation = ori

        # 5) 初始化相机：开启分割点云与相机内参与外参
        self.garment_camera.initialize(
            segment_pc_enable=True, 
            # depth_enable=True,  # 启用深度功能
            segment_prim_path_list=[
                "/World/Garment/garment"
            ],
            camera_params_enable=True
        )

        # 让双手保持张开状态
        self.bimanual_dex.set_both_hand_state("open", "open")

        # 6) 预先仿真若干步，使衣物从初始状态稳定落下
        for i in range(100):
            self.step()
        


def FoldTops(env):
    """
    核心折叠流程：
    - 使用当前相机点云和 GAM 模型，追踪并记录各部位（袖口、领口、下摆等）的
      网格顶点索引，方便后续在图像中生成语义标注；
    - 按步骤执行：
        1) 左手整理左袖
        2) 右手整理右袖
        3) 双手将下摆向上折叠
    - 在每个阶段结束后调用 save_jsonl() 保存 RGB + 标注结果。
    """
    # ---------------------- initial ---------------------- #
    # 全程隐藏机械臂（包括仿真与采集可视化），只保留衣物和环境
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight"],
        visible=False,
    )
    # 预仿真 50 步，让衣物进一步稳定
    for i in range(50):
        env.step()

    scale:np.ndarray=np.array([0.0085, 0.0085, 0.0085])
    # 1) 从布料仿真中取出当前所有网格顶点的位置，并缩放 + 平移到世界坐标
    garment_vertices = env.garment.get_vertice_positions()
    garment_vertices = garment_vertices * scale
    garment_vertices += env.garment.get_garment_center_pos()

    # 2) 从相机获取基于语义分割的点云（只包含衣物部分）
    #    若点云为空（例如分割失败 / 物体完全不在视野中），则直接跳过当前衣物
    try:
        pcd, color = env.garment_camera.get_point_cloud_data_from_segment(
            save_or_not=False,
            save_path=get_unique_filename("data", extension=".ply"),
            real_time_watch=False,
        )
    except ValueError as e:
        # 上游 furthest_point_sampling 会在空点云时抛出 ValueError
        cprint(
            f"[WARNING] 初始点云获取失败（空点云），跳过当前衣物。本次错误信息：{e}",
            color="yellow",
        )
        return

    if pcd is None or len(pcd) == 0:
        cprint("[WARNING] 初始点云为空，跳过当前衣物。", color="yellow")
        return

    # 3) 通过最近邻将相机点云与原始布料网格建立一一对应关系，
    #    garment_indices 记录了“点云中第 i 个点”对应“原网格的哪个顶点”
    garment_vertices, garment_indices = get_surface_vertices(garment_vertices, pcd)

    # Preview
    # pcd_vis = o3d.geometry.PointCloud()
    # pcd_vis.points = o3d.utility.Vector3dVector(garment_vertices)
    # pcd_vis.paint_uniform_color([0.8, 0.8, 0.8])  # Set base color to light gray
    # o3d.visualization.draw_geometries([pcd_vis])

    # pcd, garment_indices = furthest_point_sampling(garment_vertices, n_samples=2048, indices=True)
    
    # 4) 针对每个部位调用 GAM 模型：
    #    - 从当前点云 pcd 中找到和模板索引最匹配的一簇点
    #    - 用 garment_indices[indices] 映射回当前网格顶点，实现语义追踪
    manipulation_points, indices, points_similarity = env.model.get_manipulation_points(input_pcd=pcd, index_list=env.left_cuff)
    env.left_cuff_indices = garment_indices[indices]
    manipulation_points, indices, points_similarity = env.model.get_manipulation_points(input_pcd=pcd, index_list=env.right_cuff)
    env.right_cuff_indices = garment_indices[indices]
    manipulation_points, indices, points_similarity = env.model.get_manipulation_points(input_pcd=pcd, index_list=env.left_collar)
    env.left_collar_indices = garment_indices[indices]
    manipulation_points, indices, points_similarity = env.model.get_manipulation_points(input_pcd=pcd, index_list=env.right_collar)
    env.right_collar_indices = garment_indices[indices]
    manipulation_points, indices, points_similarity = env.model.get_manipulation_points(input_pcd=pcd, index_list=env.center_collar)
    env.center_collar_indices = garment_indices[indices]
    manipulation_points, indices, points_similarity = env.model.get_manipulation_points(input_pcd=pcd, index_list=env.left_hem)
    env.left_hem_indices = garment_indices[indices]
    manipulation_points, indices, points_similarity = env.model.get_manipulation_points(input_pcd=pcd, index_list=env.right_hem)
    env.right_hem_indices = garment_indices[indices]
    manipulation_points, indices, points_similarity = env.model.get_manipulation_points(input_pcd=pcd, index_list=env.center_hem)
    env.center_hem_indices = garment_indices[indices]
    manipulation_points, indices, points_similarity = env.model.get_manipulation_points(input_pcd=pcd, index_list=env.left_armpit)
    env.left_armpit_indices = garment_indices[indices]
    manipulation_points, indices, points_similarity = env.model.get_manipulation_points(input_pcd=pcd, index_list=env.right_armpit)
    env.right_armpit_indices = garment_indices[indices]
    manipulation_points, indices, points_similarity = env.model.get_manipulation_points(input_pcd=pcd, index_list=env.left_shoulder)
    env.left_shoulder_indices = garment_indices[indices]
    manipulation_points, indices, points_similarity = env.model.get_manipulation_points(input_pcd=pcd, index_list=env.right_shoulder)
    env.right_shoulder_indices = garment_indices[indices]
    manipulation_points, indices, points_similarity = env.model.get_manipulation_points(input_pcd=pcd, index_list=env.left_waist)
    env.left_waist_indices = garment_indices[indices]
    manipulation_points, indices, points_similarity = env.model.get_manipulation_points(input_pcd=pcd, index_list=env.right_waist)
    env.right_waist_indices = garment_indices[indices]

    # 5) 额外获取用于折叠动作的 6 个关键点：
    #    （此处与另一个脚本 Fold_Tops_Env 中逻辑保持一致）
    fold_manipulation_points, fold_indices, fold_points_similarity = env.model.get_manipulation_points(
        input_pcd=pcd,
        index_list=[957, 501, 1902, 448, 1196, 422],
    )
    fold_manipulation_points[0:4, 2] = 0.02
    fold_manipulation_points[4:, 2] = 0.0

    # ---------------------------------------- #
    # 将 Fold_Tops_Env.py 中的机械臂折叠策略迁移过来
    # 使用与 Fold_Tops_Env 相同的 6 个关键点语义：
    # 0,1：左袖两点；2,3：右袖两点；4,5：下摆两点
    # ---------------------------------------- #

    # 在网格顶点上记录这 6 个点对应的顶点索引，
    # 方便后续根据网格状态刷新这些关键点的 3D 位置
    env.manipulation_vertex_indices = garment_indices[fold_indices]

    def _get_world_garment_vertices():
        """从布料仿真得到当前世界坐标系下的所有顶点位置"""
        vertices = env.garment.get_vertice_positions()
        vertices = vertices * scale
        vertices += env.garment.get_garment_center_pos()
        return vertices

    def _refresh_manipulation_points_from_vertices():
        """根据当前布料网格刷新 6 个关键点的位置（与 Fold_Tops_Env 中逻辑保持一致）"""
        env.manipulation_points = _get_world_garment_vertices()[env.manipulation_vertex_indices]
        env.manipulation_points[0:4, 2] = 0.02
        env.manipulation_points[4:, 2] = 0.0
        return env.manipulation_points

    def _capture_stage_state():
        """在每个阶段前刷新关键点，返回拷贝，用于后续规划该阶段动作"""
        _refresh_manipulation_points_from_vertices()
        return env.manipulation_points.copy()

    # 先用当前网格状态初始化一次 manipulation_points
    _refresh_manipulation_points_from_vertices()

    # ====================== left hand（整理左袖） ====================== #
    manipulation_points = _capture_stage_state()

    # 估算左袖提升高度：与躯干关键点之间的距离，上限 0.3（照搬 Fold_Tops_Env）
    left_sleeve_height = min(
        np.linalg.norm(manipulation_points[0][:2] - manipulation_points[3][:2]),
        0.3,
    )
    lift_point_1 = np.array(
        [manipulation_points[0][0], manipulation_points[0][1], left_sleeve_height]
    )
    lift_point_2 = np.array(
        [manipulation_points[1][0], manipulation_points[1][1], left_sleeve_height]
    )

    # 左手移动到左袖抓取点
    env.bimanual_dex.dexleft.dense_step_action(
        target_pos=manipulation_points[0],
        target_ori=np.array([0.579, -0.579, -0.406, 0.406]),
        angular_type="quat",
    )

    # 闭合左手夹爪
    env.bimanual_dex.set_both_hand_state(
        left_hand_state="close", right_hand_state="None"
    )

    # 抬起到预设高度
    env.bimanual_dex.dexleft.dense_step_action(
        target_pos=lift_point_1,
        target_ori=np.array([0.579, -0.579, -0.406, 0.406]),
        angular_type="quat",
    )

    # 沿着袖子方向移动，帮助理顺衣袖
    env.bimanual_dex.dexleft.dense_step_action(
        target_pos=lift_point_2,
        target_ori=np.array([0.579, -0.579, -0.406, 0.406]),
        angular_type="quat",
    )

    # 松开左手
    env.bimanual_dex.set_both_hand_state(
        left_hand_state="open", right_hand_state="None"
    )

    # 提高重力系数，使布料更快“垂落”稳定
    env.garment.particle_material.set_gravity_scale(5.0)
    for i in range(200):
        env.step()
    env.garment.particle_material.set_gravity_scale(1.0)

    # 左手动作结束后拍一张照片并保存标注（含所有部位的点/框）
    # pcd, color = env.garment_camera.get_point_cloud_data_from_segment(
    #     save_or_not=False,
    #     save_path=get_unique_filename("data", extension=".ply"),
    #     real_time_watch=False,
    # )
    # save_jsonl(env)

    # 左手收回到一个“休息位姿”，避免挡住相机或影响右手动作
    env.bimanual_dex.dexleft.dense_step_action(
        target_pos=np.array([-0.6, 0.8, 0.5]),
        target_ori=np.array([0.579, -0.579, -0.406, 0.406]),
        angular_type="quat",
    )

    # ====================== right hand（整理右袖） ====================== #
    manipulation_points = _capture_stage_state()

    # 估算右袖提升高度：同样用袖子与躯干之间的距离
    right_sleeve_height = min(
        np.linalg.norm(manipulation_points[2][:2] - manipulation_points[1][:2]),
        0.3,
    )
    lift_point_1 = np.array(
        [manipulation_points[2][0], manipulation_points[2][1], right_sleeve_height]
    )
    lift_point_2 = np.array(
        [manipulation_points[3][0], manipulation_points[3][1], right_sleeve_height]
    )

    # 右手移动到右袖抓取点
    env.bimanual_dex.dexright.dense_step_action(
        target_pos=manipulation_points[2],
        target_ori=np.array([0.406, -0.406, -0.579, 0.579]),
        angular_type="quat",
    )

    # 闭合右手夹爪
    env.bimanual_dex.set_both_hand_state(
        left_hand_state="None", right_hand_state="close"
    )

    # 抬起袖子
    env.bimanual_dex.dexright.dense_step_action(
        target_pos=lift_point_1,
        target_ori=np.array([0.406, -0.406, -0.579, 0.579]),
        angular_type="quat",
    )

    # 沿袖子方向移动，整理右袖
    env.bimanual_dex.dexright.dense_step_action(
        target_pos=lift_point_2,
        target_ori=np.array([0.406, -0.406, -0.579, 0.579]),
        angular_type="quat",
    )

    # 松开右手
    env.bimanual_dex.set_both_hand_state(
        left_hand_state="None", right_hand_state="open"
    )
    
    # 同样短时间内增大重力，加快布料回落
    env.garment.particle_material.set_gravity_scale(5.0)
    for i in range(200):
        env.step()
    env.garment.particle_material.set_gravity_scale(1.0)
    #右手动作结束后拍一张照片并保存标注
    # pcd, color = env.garment_camera.get_point_cloud_data_from_segment(
    #     save_or_not=False,
    #     save_path=get_unique_filename("data", extension=".ply"),
    #     real_time_watch=False,
    # )
    # save_jsonl(env)

    # 右手收回到“休息位姿”
    env.bimanual_dex.dexright.dense_step_action(
        target_pos=np.array([0.6, 0.8, 0.5]),
        target_ori=np.array([0.406, -0.406, -0.579, 0.579]),
        angular_type="quat",
    )


    # ====================== bottom-top（下摆往上折） ====================== #
    manipulation_points = _capture_stage_state()

    # 计算 lift_height：从下摆到肩部的竖直距离
    lift_height = manipulation_points[3][1] - manipulation_points[4][1]

    lift_point_1 = np.array(
        [manipulation_points[4][0], manipulation_points[4][1], lift_height / 2]
    )
    lift_point_2 = np.array(
        [manipulation_points[5][0], manipulation_points[5][1], lift_height / 2]
    )

    push_point_1 = np.array(
        [
            manipulation_points[3][0],
            manipulation_points[3][1] + 0.1,
            min(lift_height / 2, 0.2),
        ]
    )
    push_point_2 = np.array(
        [
            manipulation_points[1][0],
            manipulation_points[1][1] + 0.1,
            min(lift_height / 2, 0.2),
        ]
    )

    # 双手先移动到下摆两侧的抓取点
    env.bimanual_dex.dense_move_both_ik(
        left_pos=manipulation_points[4],
        left_ori=np.array([0.579, -0.579, -0.406, 0.406]),
        right_pos=manipulation_points[5],
        right_ori=np.array([0.406, -0.406, -0.579, 0.579]),
    )

    # 闭合双手夹爪抓住下摆两侧
    env.bimanual_dex.set_both_hand_state(
        left_hand_state="close", right_hand_state="close"
    )

    # 先抬起到中间高度（lift_point_1 / lift_point_2）
    env.bimanual_dex.dense_move_both_ik(
        left_pos=lift_point_1,
        left_ori=np.array([0.579, -0.579, -0.406, 0.406]),
        right_pos=lift_point_2,
        right_ori=np.array([0.406, -0.406, -0.579, 0.579]),
    )

    # 再推向上方（push_point_1 / push_point_2），
    # 使下摆覆盖到衣身，完成“向上折叠”的动作
    env.bimanual_dex.dense_move_both_ik(
        left_pos=push_point_1,
        left_ori=np.array([0.579, -0.579, -0.406, 0.406]),
        right_pos=push_point_2,
        right_ori=np.array([0.406, -0.406, -0.579, 0.579]),
    )

    # 松开双手
    env.bimanual_dex.set_both_hand_state(
        left_hand_state="open", right_hand_state="open"
    )

    # 再次短时间增大重力，加快折叠后布料稳定
    env.garment.particle_material.set_gravity_scale(10.0)
    for i in range(100):
        env.step()
    env.garment.particle_material.set_gravity_scale(1.0)

    # 使用与 Fold_Tops_Env.py 中相同的策略评估折叠是否成功：
    # 1）用初始衣物点云 pcd 和一组模板点计算理想折叠区域边界
    points, *_ = env.model.get_manipulation_points(
        input_pcd=pcd,
        index_list=[554, 1540, 1014, 1385],
    )
    boundary = [
        points[0][0] - 0.05,
        points[1][0] + 0.05,
        points[3][1] - 0.1,
        points[2][1] + 0.1,
    ]

    # 2）获取当前最终衣物点云，并用 judge_pcd 判断是否大部分点都落在 boundary 内
    #    若此时相机分割返回空点云，则认为本次折叠结果无效，直接跳过当前衣物
    try:
        pcd_end, color = env.garment_camera.get_point_cloud_data_from_segment(
            save_or_not=False,
            save_path=get_unique_filename("data", extension=".ply"),
            real_time_watch=False,
        )
    except ValueError as e:
        cprint(
            f"[WARNING] 最终点云获取失败（空点云），跳过当前衣物。本次错误信息：{e}",
            color="yellow",
        )
        return

    if pcd_end is None or len(pcd_end) == 0:
        cprint("[WARNING] 最终点云为空，跳过当前衣物。", color="yellow")
        return

    success = judge_pcd(pcd_end, boundary, threshold=0.1)
    cprint(f"final result: {success}", color="green" if success else "red")

    # 只对折叠成功的样本保存 jsonl / 图片
    if success:
        save_jsonl(env)


def save_jsonl(env):
    """
    将追踪得到的关键点重新投影到 RGB 图像并输出 jsonl / 可视化结果：
    - 输入：包含当前仿真状态和相机的 env
    - 步骤：
        1) 根据 env.<area>_indices 从当前衣物网格中取出对应部位的 3D 顶点集合；
        2) 对每个部位进行聚类 + 投影，得到一个中心点 (u, v) 和一个包围框 bbox；
        3) 将上述信息写入 jsonl 文件，同时在 RGB 图像上画点和框用于可视化。
    """
    # 再次获取布料网格顶点（和 FoldTops 中保持一致的缩放/平移），
    # 结合 env.<area>_indices 就能锁定每个部位在当前帧实际的 3D 位置
    scale:np.ndarray=np.array([0.0085, 0.0085, 0.0085])
    garment_vertices = env.garment.get_vertice_positions()
    garment_vertices = garment_vertices * scale
    garment_vertices += env.garment.get_garment_center_pos()

    data = {}
    
    # 采集与点云对应的 RGB 图像，后续会投影关键点并保存可视化
    rgb = env.garment_camera.get_rgb_graph(save_or_not=False
                                        ,save_path=get_unique_filename("data", extension=".png"))
    
    # 将 RGB 图像写入到预设目录下
    output_image_path = get_unique_filename(f"Preprocess/data/{project_name}/images/image", extension=".png")
    cv2.imwrite(output_image_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    data["rgb"] = output_image_path.split("/")[-1]
    # data["not_exist_point"] = None

    # 保存pcd
    # pcd_path = get_unique_filename("Preprocess/data/pcd/pcd", extension=".ply")
    # o3d_pcd = o3d.geometry.PointCloud()
    # o3d_pcd.points = o3d.utility.Vector3dVector(garment_vertices)
    # o3d.io.write_point_cloud(pcd_path, o3d_pcd)
    # data["pcd"] = pcd_path.split("/")[-1]

    # 针对每个部位：取出当前帧被追踪到的顶点，计算聚类中心并投影成像素点/包围框
    point, bbox = get_area_info(env, rgb, garment_vertices[env.left_cuff_indices], "left_cuff")
    data["left_cuff"] = {
        "point": point,
        "bbox": bbox,
    }

    point, bbox = get_area_info(env, rgb, garment_vertices[env.right_cuff_indices], "right_cuff")
    data["right_cuff"] = {
        "point": point,
        "bbox": bbox,
    }

    point, bbox = get_area_info(env, rgb, garment_vertices[env.left_collar_indices], "left_collar")
    data["left_collar"] = {
        "point": point,
        "bbox": bbox,
    }

    point, bbox = get_area_info(env, rgb, garment_vertices[env.right_collar_indices], "right_collar")
    data["right_collar"] = {
        "point": point,
        "bbox": bbox,
    }

    point, bbox = get_area_info(env, rgb, garment_vertices[env.center_collar_indices], "center_collar")
    data["center_collar"] = {
        "point": point,
        "bbox": bbox,
    }
    
    point, bbox = get_area_info(env, rgb, garment_vertices[env.left_hem_indices], "left_hem")
    data["left_hem"] = {
        "point": point,
        "bbox": bbox,
    }
    
    point, bbox = get_area_info(env, rgb, garment_vertices[env.right_hem_indices], "right_hem")
    data["right_hem"] = {
        "point": point,
        "bbox": bbox,
    }
    
    point, bbox = get_area_info(env, rgb, garment_vertices[env.center_hem_indices], "center_hem")
    data["center_hem"] = {
        "point": point,
        "bbox": bbox,
    }
    
    point, bbox = get_area_info(env, rgb, garment_vertices[env.left_armpit_indices], "left_armpit")
    data["left_armpit"] = {
        "point": point,
        "bbox": bbox,
    }
    
    point, bbox = get_area_info(env, rgb, garment_vertices[env.right_armpit_indices], "right_armpit")
    data["right_armpit"] = {
        "point": point,
        "bbox": bbox,
    }
    
    point, bbox = get_area_info(env, rgb, garment_vertices[env.left_shoulder_indices], "left_shoulder")
    data["left_shoulder"] = {
        "point": point,
        "bbox": bbox,
    }

    point, bbox = get_area_info(env, rgb, garment_vertices[env.right_shoulder_indices], "right_shoulder")
    data["right_shoulder"] = {
        "point": point,
        "bbox": bbox,
    }
    
    point, bbox = get_area_info(env, rgb, garment_vertices[env.left_waist_indices], "left_waist")
    data["left_waist"] = {
        "point": point,
        "bbox": bbox,
    }
    
    point, bbox = get_area_info(env, rgb, garment_vertices[env.right_waist_indices], "right_waist")
    data["right_waist"] = {
        "point": point,
        "bbox": bbox,
    }
    
    # 确保 jsonl 文件所在目录存在（防止因目录缺失导致 FileNotFoundError）
    jsonl_path = f"Preprocess/data/{project_name}/garments_box.jsonl"
    jsonl_dir = os.path.dirname(jsonl_path)
    if not os.path.exists(jsonl_dir):
        os.makedirs(jsonl_dir, exist_ok=True)
    
    # 以追加模式写入 jsonl，每一行是一个样本的一整套标注
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(data) + "\n")

    rgb_image = rgb.copy().astype(np.uint8)

    # 在 RGB 图像上画出点和框，方便肉眼检查标注质量
    for key, value in data.items():
        if key != "rgb" and key != "pcd":
            point = value["point"]
            bbox = value["bbox"]
            
            # Check if point is valid (not None)
            if point is not None and point[0] is not None and point[1] is not None:
                cv2.circle(rgb_image, (int(point[0]), int(point[1])), radius=1, color=(0, 255, 0), thickness=-1)
            
            # Check if bbox is valid and has proper coordinates
            if (bbox is not None and len(bbox) == 4 and 
                bbox[0] is not None and bbox[1] is not None and
                bbox[2] is not None and bbox[3] is not None):
                
                pt1 = (int(bbox[0]), int(bbox[1]))
                pt2 = (int(bbox[2]), int(bbox[3]))
                cv2.rectangle(rgb_image, pt1, pt2, color=(255, 0, 0), thickness=1)
                
                # Only add text if point is valid
                if point is not None and point[0] is not None and point[1] is not None:
                    cv2.putText(rgb_image, key, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    output_image_path = get_unique_filename(f"Preprocess/data/{project_name}/data_vis/vis", extension=".png")
    cv2.imwrite(output_image_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))


def get_area_info(env, rgb, manipulation_points, area_name):
    """
    对某个语义区域的一批 3D 点进行聚类筛选，并将其投影到像素坐标：
    - 输入：
        env: 环境对象（提供投影函数）
        rgb: 当前帧 RGB 图像
        manipulation_points: (N, 3) 一批 3D 顶点
        area_name: 区域名称（仅用于 log 打印）
    - 输出：
        centroid: (u, v) 像素坐标的中心点（若无效则返回 (0, 0)）
        bbox: [min_x, min_y, max_x, max_y] 形式的像素包围框（若无效则为 [0, 0, 0, 0]）
    """
    n_points = len(manipulation_points)
    
    # 如果该区域没有任何点，直接返回默认值，避免后续计算出错
    if n_points == 0:
        print(f"No manipulation points for {area_name}, skipping.")
        # 返回一个默认的中心和 bbox，后续会在可视化时被当成无效点处理
        return (0, 0), [0, 0, 0, 0]
    
    # 用距离阈值做连通域聚类，尽量滤掉误匹配的离散点
    # 下面先计算所有点对之间的欧式距离
    distance_matrix = np.zeros((n_points, n_points))
    
    for i in range(n_points):
        for j in range(i+1, n_points):
            dist = np.linalg.norm(manipulation_points[i] - manipulation_points[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    # 取非零距离，用于估计聚类的阈值
    non_zero_distances = distance_matrix[distance_matrix > 0]
    
    # 如果所有距离都为 0（或者只有 1 个点），则无法计算分位数，直接跳过聚类步骤
    if non_zero_distances.size > 0:
        # 使用非零距离的百分位数（70%）作为聚类的距离阈值
        # 这个参数可以根据衣物大小、点云密度进行微调
        threshold = np.percentile(non_zero_distances, 70)
    else:
        threshold = None
    
    # Find clusters using connected components
    clusters = []
    visited = [False] * n_points
    
    for i in range(n_points):
        if not visited[i]:
            # Start a new cluster
            cluster = [i]
            visited[i] = True
            
            if threshold is not None:
                # Find all points connected to this point
                stack = [i]
                while stack:
                    current = stack.pop()
                    for j in range(n_points):
                        if not visited[j] and distance_matrix[current, j] <= threshold:
                            cluster.append(j)
                            visited[j] = True
                            stack.append(j)
            
            clusters.append(cluster)
    
    # Select the largest cluster
    if clusters:
        largest_cluster_idx = max(range(len(clusters)), key=lambda i: len(clusters[i]))
        largest_cluster = clusters[largest_cluster_idx]
        
        # If the largest cluster has at least 3 points, use it
        if len(largest_cluster) >= 3 and len(largest_cluster) < n_points:
            manipulation_points = manipulation_points[largest_cluster]
            print(f"Selected cluster with {len(largest_cluster)} points from {n_points} total points for {area_name}")
        else:
            print(f"Largest cluster too small ({len(largest_cluster)} points), using all {n_points} points for {area_name}")
    else:
        print(f"No clusters found, using all {n_points} points for {area_name}")
    
    # 将选出的 3D 点求质心，并投影成像素坐标，得到单点标注
    centroid = np.mean(manipulation_points, axis=0)

    centroid = get_rgb_index(env, rgb, centroid)

    # 进一步将每个 3D 点都投影到像素坐标，用于计算包围框
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    valid_points = []

    for point in manipulation_points:
        pixel_x, pixel_y = get_rgb_index(env, rgb, point)
        if pixel_x is not None and pixel_y is not None:
            min_x = min(min_x, pixel_x)
            max_x = max(max_x, pixel_x)
            min_y = min(min_y, pixel_y)
            max_y = max(max_y, pixel_y)
            valid_points.append((pixel_x, pixel_y))

    # 检查是否有合法的投影点来构成 bbox
    if valid_points and min_x != float('inf') and max_x != float('-inf'):
        # Add the overall bounding box
        bbox = [min_x, min_y, max_x, max_y]
    else:
        # If no valid points, set default values
        bbox = [0, 0, 0, 0]
        if centroid[0] is None or centroid[1] is None:
            centroid = (0, 0)

    return centroid, bbox


def get_rgb_index(env, rgb, point):
    """
    将 3D 点（世界坐标）投影到相机平面，返回像素坐标 (u, v)：
    - 若点在相机前方且位于视锥内，则返回其在图像中的整数像素坐标；
    - 否则返回 (None, None) 表示该点不可见。
    """
    # 获取相机的 view / projection 矩阵
    view_matrix, projection_matrix = env.garment_camera.get_camera_matrices()
    
    # 获取 RGB 图像大小（用于规范化坐标到像素）
    height, width, _ = rgb.shape

    # 将点从世界坐标扩展为齐次坐标
    point_world = np.append(point, 1.0)
    
    # 先通过 view_matrix 变换到相机坐标系 / 视图坐标
    point_camera_view = point_world @ view_matrix

    # 再通过 projection_matrix 将点投影到裁剪空间 (clip space)
    point_clip = point_camera_view @ projection_matrix
    
    # w>0 表示点位于相机前方
    if point_clip[3] > 0:  # check if the point is in front of the camera
        point_ndc = point_clip[:3] / point_clip[3]
        
        # 检查点是否在 NDC 范围内（-1~1），否则说明超出相机视锥
        if -1 <= point_ndc[0] <= 1 and -1 <= point_ndc[1] <= 1:
            pixel_x = int((point_ndc[0] + 1) * width / 2)
            pixel_y = int((1 - point_ndc[1]) * height / 2)  # y is inverted in image coordinates

            return pixel_x, pixel_y
    return None, None


if __name__=="__main__":
    args=parse_args_record()
    # ===================================================================================== #
    project_name = "stage3_1207"
    # ===================================================================================== #
    # 若数据输出主目录不存在，则创建
    if not os.path.exists("Preprocess/data"):
        os.makedirs("Preprocess/data")

    # 存放 RGB 原图的目录
    if not os.path.exists(f"Preprocess/data/{project_name}/images"):
        os.makedirs(f"Preprocess/data/{project_name}/images")

    # if not os.path.exists("Preprocess/data/pcd"):
    #     os.makedirs("Preprocess/data/pcd")

    # 存放可视化图（画有点和 bbox）的目录
    if not os.path.exists(f"Preprocess/data/{project_name}/data_vis"):
        os.makedirs(f"Preprocess/data/{project_name}/data_vis")

    # 衣物初始位姿（后续在循环中会被覆盖，这里只是默认值）
    pos = np.array([0.0, 0.8, 0.2])
    ori = np.array([0.0, 0.0, 0.0])
    usd_path = None
    
    # Base_dir 为项目根目录（当前文件的上一级目录的上一级）
    Base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ==============================
    # 原始逻辑：从 assets_list.txt 中读取多件衣物资产并遍历采集
    # ==============================
    # assets_list.txt 中存储了可用的 Tops_LongSleeve 衣物资产 usd 路径
    assets_lists = os.path.join(Base_dir,"Model_HALO/GAM/checkpoints/Tops_LongSleeve/assets_list.txt")
    assets_list = []
    with open(assets_lists,"r",encoding='utf-8') as f:
        for line in f:
            clean_line = line.rstrip('\n')
            assets_list.append(clean_line)
    # 可以随机抽一件：
    # usd_path = os.path.join(Base_dir, np.random.choice(assets_list))

    # ==============================
    # 以前的“只用一件固定衣物资产”的逻辑，先保留成注释，方便以后再切换
    # ==============================
    # fixed_usd_rel_path = "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_018/TCLC_018_obj.usd"
    # usd_path = os.path.join(Base_dir, fixed_usd_rel_path)

    # floors_list.txt 中存储了可用的地面材质 usd 路径
    floors_lists = os.path.join(Base_dir,"Preprocess/floors_list.txt")
    floors_list = []
    with open(floors_lists,"r",encoding='utf-8') as f:
        for line in f:
            clean_line = line.rstrip('\n')
            floors_list.append(clean_line)

    # 先随机选一个地面材质（后面循环中每次仍会随机选一次）
    ground_material_usd = np.random.choice(floors_list)

    # 创建折叠环境（内部会初始化 Isaac world、机器人、相机等）
    env = FoldTops_Env()
    
    # assets_list = assets_list[::-1]

    # ==============================
    # 遍历资产列表，对每一件衣物进行仿真与数据采集
    # 这里保留你刚才改的 for i in range(5)，表示每件衣服跑 5 次
    # ==============================
    #assets_list = assets_list[103:]
    for usd_path in assets_list:
        for i in range(1):
        # for ground_material_usd in floors_list:
            # 当前正在使用的衣物资产路径
            print(usd_path)
            ground_material_usd = np.random.choice(floors_list)
            np.random.seed(int(time.time()))
            # 为每个样本随机一个衣物初始位置（x, y），增加数据多样性
            x = np.random.uniform(-0.1, 0.1)  # changeable
            y = np.random.uniform(0.7, 0.9)   # changeable
            pos = np.array([x,y,0.0])
            ori = np.array([0.0, 0.0, 0.0])

            # 将场景重置为当前衣物 + 地面材质，并执行折叠流程
            env.apply(pos, ori, ground_material_usd, usd_path)
            FoldTops(env)

    # 所有资产仿真结束后，关闭 Isaac Sim 应用
    simulation_app.close()