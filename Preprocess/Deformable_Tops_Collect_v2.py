from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# load external package
import os
import sys
import time
import numpy as np
import open3d as o3d
from termcolor import cprint
import threading
import cv2

# load isaac-relevant package
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

# load custom package
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

import json

class FoldTops_Env(BaseEnv):
    def __init__(self):
        # load BaseEnv
        super().__init__()

        self.bimanual_dex = Bimanual_Ur10e(
            self.world,
            dexleft_pos=np.array([-0.8, 0.0, 0.5]),
            dexleft_ori=np.array([0.0, 0.0, 0.0]),
            dexright_pos=np.array([0.8, 0.0, 0.5]),
            dexright_ori=np.array([0.0, 0.0, 0.0]),
        )

        # load camera
        self.garment_camera = Recording_Camera(
            camera_position=np.array([0.0, 0.8, 6]), 
            camera_orientation=np.array([0, 90.0, 90.0]),
            prim_path="/World/garment_camera",
        )
        
        # load GAM Model
        self.model = GAM_Encapsulation(catogory="Tops_LongSleeve")
        
        # Initialize variables
        self.garment = None
        self.ground = None
        self.garment_pcd = None
        self.points_affordance_feature = None
        
    def cleanup(self):
        """Clean up previous objects before creating new ones"""
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
        
        # Clean up previous objects first
        self.cleanup()
        
        self.ground = Real_Ground(
                self.scene, 
                visual_material_usd = ground_material_usd,
                # you can use materials in 'Assets/Material/Floor' to change the texture of ground.
            )
            
        # load garment
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
                visual_material_usd = np.random.choice(["Assets/Material/Garment/linen_Pumpkin.usd", "Assets/Material/Garment/linen_Blue.usd"])
            )
        
        self.garment_pcd = None
        self.points_affordance_feature = None

        # initialize world
        self.reset()
        
        # move garment to the target position
        self.garment.set_pose(pos=np.array([pos[0], pos[1], 0.2]), ori=ori)
        self.position = [pos[0], pos[1], 0.2]
        self.orientation = ori

        self.garment_camera.initialize(
            segment_pc_enable=True, 
            # depth_enable=True,  # 启用深度功能
            segment_prim_path_list=[
                "/World/Garment/garment"
            ],
            camera_params_enable=True
        )

        self.bimanual_dex.set_both_hand_state("open", "open")

        # step world to make it ready
        for i in range(100):
            self.step()
        


def FoldTops(env):
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight"],
        visible=False,
    )
     
    for i in range(50):
        env.step()

    pcd, color = env.garment_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".ply"),
        real_time_watch=False,
    )

    save_jsonl(env, pcd)
    
    # ---------------------- left hand ---------------------- #
    
    source_pos = pcd[np.random.randint(0, len(pcd))]
    target_pos = pcd[np.random.randint(0, len(pcd))]

    env.bimanual_dex.dexleft.dense_step_action(target_pos=source_pos, target_ori=np.array([0.579, -0.579, -0.406, 0.406]), angular_type="quat")
    
    env.bimanual_dex.set_both_hand_state(left_hand_state="close", right_hand_state="None")
    
    height = 0.3

    lift_point_1 = np.array([source_pos[0], source_pos[1], height])

    env.bimanual_dex.dexleft.dense_step_action(target_pos=lift_point_1, target_ori=np.array([0.579, -0.579, -0.406, 0.406]), angular_type="quat")
    
    lift_point_2 = np.array([target_pos[0], target_pos[1], height])
    
    env.bimanual_dex.dexleft.dense_step_action(target_pos=lift_point_2, target_ori=np.array([0.579, -0.579, -0.406, 0.406]), angular_type="quat")

    env.bimanual_dex.set_both_hand_state(left_hand_state="open", right_hand_state="None")
    
    env.garment.particle_material.set_gravity_scale(10.0)
    for i in range(200):
        env.step()
    env.garment.particle_material.set_gravity_scale(1.0) 
    
    
    env.bimanual_dex.dexleft.dense_step_action(target_pos=np.array([-0.6, 0.8, 0.5]), target_ori=np.array([0.579, -0.579, -0.406, 0.406]), angular_type="quat")

    for i in range(50):
        env.step()
    
    pcd, color = env.garment_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".ply"),
        real_time_watch=False,
    )

    save_jsonl(env, pcd)

    # --------------------- right hand --------------------- #
    source_pos = pcd[np.random.randint(0, len(pcd))]
    target_pos = pcd[np.random.randint(0, len(pcd))]
            
    env.bimanual_dex.dexright.dense_step_action(target_pos=source_pos, target_ori=np.array([0.406, -0.406, -0.579, 0.579]), angular_type="quat")
            
    env.bimanual_dex.set_both_hand_state(left_hand_state="None", right_hand_state="close")
    
    height = 0.3
    
    lift_point_1 = np.array([source_pos[0], source_pos[1], height])
    
    env.bimanual_dex.dexright.dense_step_action(target_pos=lift_point_1, target_ori=np.array([0.406, -0.406, -0.579, 0.579]), angular_type="quat")
    
    lift_point_2 = np.array([target_pos[0], target_pos[1], height])
    
    env.bimanual_dex.dexright.dense_step_action(target_pos=lift_point_2, target_ori=np.array([0.406, -0.406, -0.579, 0.579]), angular_type="quat")

    env.bimanual_dex.set_both_hand_state(left_hand_state="None", right_hand_state="open")
    
    env.garment.particle_material.set_gravity_scale(10.0)
    for i in range(200):
        env.step()
    env.garment.particle_material.set_gravity_scale(1.0) 
    
    env.bimanual_dex.dexright.dense_step_action(target_pos=np.array([0.6, 0.8, 0.5]), target_ori=np.array([0.406, -0.406, -0.579, 0.579]), angular_type="quat")

    for i in range(50):
        env.step()
    
    pcd, color = env.garment_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".ply"),
        real_time_watch=False,
    )
    save_jsonl(env, pcd)


def save_jsonl(env, pcd):
    left_cuff = [8,45,70,144,270,296,351,386,409,522,636,675,710,785,830,957,1091,1144,1224,1470,1475,1635,1761,1762,1763,1839,1895,1959,1990,2027]
    right_cuff = [1,38,120,153,233,302,362,364,394,425,564,571,599,775,878,954,977,1017,1036,1039,1043,1134,1139,1176,1202,1321,1506,1629,1709,1755,1902,1916]

    left_collar = [132,163,288,305,519,548,573,686,1003,1019,1125,1148,1329,1334,1455,1592,1690]
    right_collar = [76,135,149,355,454,660,740,970,1069,1078,1096,1114,1178,1261,1264,1287,1799,1906,2008]
    center_collar = [10,150,340,404,465,584,780,811,866,887,1006,1010,1093,1118,1333,1519,1537,1576,1589,1597,1713,1723,1980,2004,2034]

    left_hem = [7,106,147,231,236,382,387,532,563,609,673,958,1082,1146,1161,1196,1241,1418,1661,1687,1697,1760,2031]
    right_hem = [50,152,201,369,422,471,567,585,873,1018,1081,1220,1308,1341,1370,1605,1615,1751,2025]
    center_hem = [46,128,137,214,223,326,327,395,439,597,623,726,809,983,1021,1063,1073,1206,1237,1371,1376,1509,1548,1566,1567,1568,1580,1764,1800]

    left_armpit = [12,78,99,234,307,432,435,474,493,556,593,649,687,745,923,930,943,1059,1123,1228,1239,1313,1346,1474,1508,1665,1736,1841,1872,1877,1909,1948,1975,1979,1997]
    right_armpit = [60,79,164,170,172,228,348,458,467,535,543,637,654,824,972,986,1016,1066,1136,1143,1172,1246,1262,1382,1477,1479,1526,1533,1619,1677,1692,1837,1865,1922,1952,1962,1965,2013,2032,2037,2041]
    
    left_shoulder = [4,89,113,119,175,297,374,455,483,509,512,555,627,630,757,791,802,864,876,990,1020,1044,1064,1212,1219,1266,1284,1318,1335,1410,1428,1478,1559,1601,1747,1773,1831,1957,2040]
    right_shoulder = [2,95,98,182,215,310,363,418,454,502,518,523,660,669,743,750,752,789,793,907,918,941,963,1103,1150,1233,1240,1253,1270,1281,1286,1305,1399,1547,1570,1689,1770,1781,1799,1850,1854,1982,2029]

    left_waist =  [14,64,110,143,241,357,398,410,490,758,760,766,837,945,974,1005,1132,1173,1244,1247,1272,1468,1481,1574,1683,1759,1926,1928,2046]
    right_waist = [68,117,202,315,323,459,537,633,648,659,708,720,855,919,944,1166,1256,1291,1296,1494,1512,1587,1598,1633,1836,1898,1920,1924,1972,1995,2030]

    data = {}
    
    rgb = env.garment_camera.get_rgb_graph(save_or_not=False
                                           ,save_path=get_unique_filename("data", extension=".png"))
    
    output_image_path = get_unique_filename("Preprocess/data/image/image", extension=".png")
    cv2.imwrite(output_image_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    data["rgb"] = output_image_path.split("/")[-1]
    # data["not_exist_point"] = None

    # 保存pcd
    pcd_path = get_unique_filename("Preprocess/data/pcd/pcd", extension=".ply")
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd)
    o3d.io.write_point_cloud(pcd_path, o3d_pcd)
    data["pcd"] = pcd_path.split("/")[-1]

    point, bbox = get_area_info(env, rgb, pcd, left_cuff)
    data["left_cuff"] = {
        "point": point,
        "bbox": bbox,
    }

    point, bbox = get_area_info(env, rgb, pcd, right_cuff)
    data["right_cuff"] = {
        "point": point,
        "bbox": bbox,
    }

    # point, bbox = get_area_info(env, rgb, pcd, left_collar)
    # data["left_collar"] = {
    #     "point": point,
    #     "bbox": bbox,
    # }

    # point, bbox = get_area_info(env, rgb, pcd, right_collar)
    # data["right_collar"] = {
    #     "point": point,
    #     "bbox": bbox,
    # }

    # point, bbox = get_area_info(env, rgb, pcd, center_collar)
    # data["center_collar"] = {
    #     "point": point,
    #     "bbox": bbox,
    # }
    
    point, bbox = get_area_info(env, rgb, pcd, left_hem)
    data["left_hem"] = {
        "point": point,
        "bbox": bbox,
    }
    
    point, bbox = get_area_info(env, rgb, pcd, right_hem)
    data["right_hem"] = {
        "point": point,
        "bbox": bbox,
    }
    
    point, bbox = get_area_info(env, rgb, pcd, center_hem)
    data["center_hem"] = {
        "point": point,
        "bbox": bbox,
    }
    
    point, bbox = get_area_info(env, rgb, pcd, left_armpit)
    data["left_armpit"] = {
        "point": point,
        "bbox": bbox,
    }
    
    point, bbox = get_area_info(env, rgb, pcd, right_armpit)
    data["right_armpit"] = {
        "point": point,
        "bbox": bbox,
    }
    
    # point, bbox = get_area_info(env, rgb, pcd, left_shoulder)
    # data["left_shoulder"] = {
    #     "point": point,
    #     "bbox": bbox,
    # }

    # point, bbox = get_area_info(env, rgb, pcd, right_shoulder)
    # data["right_shoulder"] = {
    #     "point": point,
    #     "bbox": bbox,
    # }
    
    point, bbox = get_area_info(env, rgb, pcd, left_waist)
    data["left_waist"] = {
        "point": point,
        "bbox": bbox,
    }
    
    point, bbox = get_area_info(env, rgb, pcd, right_waist)
    data["right_waist"] = {
        "point": point,
        "bbox": bbox,
    }
    
    with open("Preprocess/data/tile.jsonl", "a") as f:
        f.write(json.dumps(data) + "\n")

    rgb_image = rgb.copy().astype(np.uint8)

    for key, value in data.items():
        if key != "rgb" and key != "pcd":
            point = value["point"]
            bbox = value["bbox"]
            cv2.circle(rgb_image, point, radius=1, color=(0, 255, 0), thickness=-1)
            cv2.rectangle(rgb_image, bbox[0], bbox[1], color=(255, 0, 0), thickness=1)
            cv2.putText(rgb_image, key, bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    output_image_path = get_unique_filename("data_vis", extension=".png")
    cv2.imwrite(output_image_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))


def get_area_info(env, rgb, pcd, keypoints):
    manipulation_points, indices, points_similarity = env.model.get_manipulation_points(input_pcd=pcd, index_list=keypoints)
    centroid = np.mean(manipulation_points, axis=0)

    # Calculate distances to centroid and get closest 90% points
    distances = np.linalg.norm(manipulation_points - centroid, axis=1)
    threshold = np.percentile(distances, 50)
    closest_points = manipulation_points[distances <= threshold]

    centroid = np.mean(closest_points, axis=0) #### once again, get the centroid of the closest 90% points

    centroid = get_rgb_index(env, rgb, centroid)

    # Project points and find bbox coordinates
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    for point in closest_points:
        pixel_x, pixel_y = get_rgb_index(env, rgb, point)
        if pixel_x is not None and pixel_y is not None:
            min_x = min(min_x, pixel_x)
            max_x = max(max_x, pixel_x)
            min_y = min(min_y, pixel_y)
            max_y = max(max_y, pixel_y)

    # Add the overall bounding box
    bbox = [(min_x, min_y), (max_x, max_y)]

    return centroid, bbox

def get_rgb_index(env, rgb, point):
    # Get camera parameters
    view_matrix, projection_matrix = env.garment_camera.get_camera_matrices()
    
    # Get RGB image and its dimensions
    height, width, _ = rgb.shape

    # Project points and draw on image
    point_world = np.append(point, 1.0)
    
    # Transform point from world to camera view
    point_camera_view = point_world @ view_matrix

    # Project point
    point_clip = point_camera_view @ projection_matrix
    
    if point_clip[3] > 0: # check if the point is in front of the camera
        point_ndc = point_clip[:3] / point_clip[3]
        
        # Check if point is within NDC space [-1, 1] for x and y
        if -1 <= point_ndc[0] <= 1 and -1 <= point_ndc[1] <= 1:
            pixel_x = int((point_ndc[0] + 1) * width / 2)
            pixel_y = int((1 - point_ndc[1]) * height / 2) # y is inverted in image coordinates

            return pixel_x, pixel_y
    return None, None


if __name__=="__main__":
    args=parse_args_record()
    
    if not os.path.exists("Preprocess/data"):
        os.makedirs("Preprocess/data")

    if not os.path.exists("Preprocess/data/image"):
        os.makedirs("Preprocess/data/image")

    if not os.path.exists("Preprocess/data/pcd"):
        os.makedirs("Preprocess/data/pcd")

    # initial setting
    pos = np.array([0.0, 0.8, 0.2])
    ori = np.array([0.0, 0.0, 0.0])
    usd_path = None
    
    Base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_lists = os.path.join(Base_dir,"Model_HALO/GAM/checkpoints/Tops_LongSleeve/assets_list.txt")
    assets_list = []
    with open(assets_lists,"r",encoding='utf-8') as f:
        for line in f:
            clean_line = line.rstrip('\n')
            assets_list.append(clean_line)

    # usd_path=np.random.choice(assets_list)

    floors_lists = os.path.join(Base_dir,"Preprocess/floors_list.txt")
    floors_list = []
    with open(floors_lists,"r",encoding='utf-8') as f:
        for line in f:
            clean_line = line.rstrip('\n')
            floors_list.append(clean_line)

    ground_material_usd = np.random.choice(floors_list)

    env = FoldTops_Env()
    
    for usd_path in assets_list:
        for ground_material_usd in floors_list:
            np.random.seed(int(time.time()))
            x = np.random.uniform(-0.1, 0.1) # changeable
            y = np.random.uniform(0.7, 0.9) # changeable
            pos = np.array([x,y,0.0])
            ori = np.array([0.0, 0.0, 0.0])

            env.apply(pos, ori, ground_material_usd, usd_path)
            FoldTops(env)

    simulation_app.close()