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

        # step world to make it ready
        for i in range(100):
            self.step()
        


def FoldTops(env):
    for i in range(50):
        env.step()
    
    rgb = env.garment_camera.get_rgb_graph(save_or_not=False
                                           ,save_path=get_unique_filename("data", extension=".png"))
    # print(rgb.shape) # (480, 640, 3)
    
    pcd, color = env.garment_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".ply"),
        real_time_watch=False,
    )

    env.garment_pcd=pcd
    
    for i in range(50):
        env.step()
    
    manipulation_points, indices, points_similarity = env.model.get_manipulation_points(input_pcd=pcd, index_list=[957, 501, 1902, 448, 1196, 422, 1228, 824, 1364]) # [左袖口，右领口，右袖口，左领口，左下，右下]

    points_to_draw = pcd[indices]

    # Get camera parameters
    view_matrix, projection_matrix = env.garment_camera.get_camera_matrices()
    
    # Get RGB image and its dimensions
    rgb_image = rgb.copy().astype(np.uint8)
    height, width, _ = rgb_image.shape

    points = []
    bboxes = []

    # Project points and draw on image
    for point in points_to_draw:
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

                points.append((pixel_x, pixel_y))
                
                # Draw a circle on the image
                cv2.circle(rgb_image, (pixel_x, pixel_y), radius=5, color=(0, 255, 0), thickness=-1)
                
                # Draw a bounding box around the point
                box_size = 10
                top_left = (pixel_x - box_size, pixel_y - box_size)
                bottom_right = (pixel_x + box_size, pixel_y + box_size)
                cv2.rectangle(rgb_image, top_left, bottom_right, color=(255, 0, 0), thickness=2)
                bboxes.append([top_left, bottom_right])
    # Save or display the result
    output_image_path = get_unique_filename("data_vis", extension=".png")
    cv2.imwrite(output_image_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

    output_image_path = get_unique_filename("Preprocess/data/images/image", extension=".png")
    cv2.imwrite(output_image_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    return {
        "rgb": output_image_path.split("/")[-1],
        "points": points,
        "bbox": bboxes,
    }
    
    
   
if __name__=="__main__":
    args=parse_args_record()
    
    if not os.path.exists("Preprocess/data"):
        os.makedirs("Preprocess/data")

    if not os.path.exists("Preprocess/data/images"):
        os.makedirs("Preprocess/data/images")

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
            data = FoldTops(env)

            # Save to jsonl file
            with open("Preprocess/data/tile.jsonl", "a") as f:
                f.write(json.dumps(data) + "\n")

    simulation_app.close()