from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

# load external package
import os
import sys
import time
import numpy as np
import open3d as o3d
from termcolor import cprint
import threading
import json
import base64
import re
import cv2
from typing import Dict, List, Tuple, Optional
from openai import OpenAI

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
from Env_Config.Utils_Project.Point_Cloud_Manip import compute_similarity

# ------------------------- #
#   VLM Helper Functions    #
# ------------------------- #

VLM_FOLD_PROMPT = (
    "Given an image of a potentially deformable garment, identify and localize the following key regions: "
    "left_cuff, right_cuff, left_collar, right_collar, center_collar, left_hem, right_hem, center_hem, "
    "left_armpit, right_armpit, left_shoulder, right_shoulder, left_waist, right_waist. "
    "For each region, provide a tight 2D bounding box in the format: [x_min, y_min, x_max, y_max]. "
    "Return the results as a JSON array where each entry contains a \"label\" and a \"bbox_2d\" field. "
    "Example format: [{\"label\": \"left_cuff\", \"bbox_2d\": [x1, y1, x2, y2]}, {\"label\": \"right_cuff\", \"bbox_2d\": [x1, y1, x2, y2]}]. "
    "Region definitions: left_collar: left collar tip; right_collar: right collar tip; center_collar: lowest point of the V-neck or collar center; "
    "left_cuff: center of left sleeve opening; right_cuff: center of right sleeve opening; "
    "left_hem: bottom-left corner of the hem; right_hem: bottom-right corner of the hem; center_hem: midpoint of the bottom hem; "
    "left_armpit: under left armpit area; right_armpit: under right armpit area; "
    "left_shoulder: left shoulder point where sleeve attaches; right_shoulder: right shoulder point; "
    "left_waist: left waist point; right_waist: right waist point. Ensure all regions are included in the output."
)

def _get_vlm_client():
    api_key = os.environ.get("VLM_API_KEY", "EMPTY")
    base_url = os.environ.get("VLM_BASE_URL", "http://localhost:8001/v1") # Default local
    model_name = os.environ.get("VLM_MODEL_NAME", "/share_data/yanruilin/qwen3vl_full_sft_cloth_sim") # Default model
    client = OpenAI(base_url=base_url, api_key=api_key)
    return client, model_name

def _encode_rgb_to_data_url(rgb: np.ndarray) -> str:
    rgb_uint8 = rgb.astype("uint8")
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(".png", bgr)
    if not success:
        raise RuntimeError("Failed to encode RGB image to PNG.")
    img_bytes = buf.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"

def _parse_vlm_output(raw_text: str) -> List[Dict[str, object]]:
    try:
        data = json.loads(raw_text)
    except Exception:
        match = re.search(r"\[.*\]", raw_text, re.S)
        if not match:
            raise ValueError(f"Cannot find JSON array in VLM output:\n{raw_text}")
        data = json.loads(match.group(0))
    
    if not isinstance(data, list):
         raise ValueError(f"VLM output JSON is not a list: {data}")
    return data

def _ask_vlm_points(rgb: np.ndarray, client, model_name: str) -> List[Dict[str, object]]:
    image_data_url = _encode_rgb_to_data_url(rgb)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VLM_FOLD_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                ],
            }
        ],
        max_tokens=1024,
    )
    raw = response.choices[0].message.content
    return _parse_vlm_output(raw)

def get_rgb_index(env, rgb: np.ndarray, point: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
    view_matrix, projection_matrix = env.garment_camera.get_camera_matrices()
    height, width, _ = rgb.shape
    point_world = np.append(point, 1.0)
    point_camera_view = point_world @ view_matrix
    point_clip = point_camera_view @ projection_matrix
    if point_clip[3] > 0:
        point_ndc = point_clip[:3] / point_clip[3]
        if -1 <= point_ndc[0] <= 1 and -1 <= point_ndc[1] <= 1:
            pixel_x = int((point_ndc[0] + 1) * width / 2)
            pixel_y = int((1 - point_ndc[1]) * height / 2)
            return pixel_x, pixel_y
    return None, None

def _project_pcd_to_pixels(env, rgb: np.ndarray, pcd: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = pcd.shape[0]
    us = np.zeros(n, dtype=np.float32)
    vs = np.zeros(n, dtype=np.float32)
    mask = np.zeros(n, dtype=bool)
    for i, pt in enumerate(pcd):
        u, v = get_rgb_index(env, rgb, pt)
        if u is not None and v is not None:
            us[i] = u
            vs[i] = v
            mask[i] = True
    return us, vs, mask

def _center_3d_for_bbox(bbox: List[float], us: np.ndarray, vs: np.ndarray, mask: np.ndarray, pcd: np.ndarray) -> Optional[np.ndarray]:
    x_min, y_min, x_max, y_max = bbox
    inside_2d = (
        (us >= x_min)
        & (us <= x_max)
        & (vs >= y_min)
        & (vs <= y_max)
        & mask
    )
    if not np.any(inside_2d):
        return None
    return pcd[inside_2d].mean(axis=0)

def _save_stage_image(env, stage_name: str):
    if not os.path.exists(f"Data/Fold_Tops/{stage_name}"):
        os.makedirs(f"Data/Fold_Tops/{stage_name}")
    
    rgb = env.garment_camera.get_rgb_graph(save_or_not=False)
    # Convert RGB to BGR for OpenCV
    bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    filename = get_unique_filename(f"Data/Fold_Tops/{stage_name}/image", extension=".png")
    cv2.imwrite(filename, bgr)


class FoldTops_Env(BaseEnv):
    def __init__(
        self, 
        pos:np.ndarray=None, 
        ori:np.ndarray=None, 
        usd_path:str=None, 
        ground_material_usd:str=None,
        record_video_flag:bool=False, 
    ):
        # load BaseEnv
        super().__init__()

        # ------------------------------------ #
        # ---        Add Env Assets        --- #
        # ------------------------------------ #
        if ground_material_usd is None:
             Base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
             floors_lists = os.path.join(Base_dir,"Preprocess/floors_list.txt")
             floors_list = []
             with open(floors_lists,"r",encoding='utf-8') as f:
                 for line in f:
                     clean_line = line.rstrip('\n')
                     floors_list.append(clean_line)
             ground_material_usd = np.random.choice(floors_list)

        self.ground = Real_Ground(
            self.scene, 
            visual_material_usd = ground_material_usd,
            # you can use materials in 'Assets/Material/Floor' to change the texture of ground.
        )

        # 随机选择衣服材质颜色
        garment_material_list = [
            "Assets/Material/Garment/linen_Pumpkin.usd",
            "Assets/Material/Garment/linen_Blue.usd"
        ]
        selected_material = np.random.choice(garment_material_list)

        # load garment
        self.garment = Particle_Garment(
            self.world, 
            pos=np.array([0, 3.0, 0.6]),
            ori=np.array([0.0, 0.0, 0.0]),
            usd_path=os.getcwd() + "/" + "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_018/TCLC_018_obj.usd" if usd_path is None else usd_path,
            visual_material_usd=selected_material,  # 使用随机选择的材质
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

        # load bimanual_dex
        self.bimanual_dex = Bimanual_Ur10e(
            self.world,
            dexleft_pos=np.array([-0.8, 0.0, 0.5]),
            dexleft_ori=np.array([0.0, 0.0, 0.0]),
            dexright_pos=np.array([0.8, 0.0, 0.5]),
            dexright_ori=np.array([0.0, 0.0, 0.0]),
        )

        # load camera
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

        self.garment_pcd = None
        self.points_affordance_feature = None

        # load GAM Model
        self.model = GAM_Encapsulation(catogory="Tops_LongSleeve")   

        # ------------------------------------ #
        # --- Initialize World to be Ready --- #
        # ------------------------------------ #
        # initialize world
        self.reset()

        # move garment to the target position
        self.garment.set_pose(pos=np.array([pos[0], pos[1], 0.2]), ori=ori)
        self.position = [pos[0], pos[1], 0.2]
        self.orientation = ori

        self.garment_camera.initialize(
            segment_pc_enable=True, 
            segment_prim_path_list=[
                "/World/Garment/garment"
            ],
            camera_params_enable=True # Enable camera params for projection
        )

        self.env_camera.initialize(depth_enable=True)

        # add thread and record gif Asynchronously(use to collect rgb data for generating gif)
        if record_video_flag:
            self.thread_record = threading.Thread(target=self.env_camera.collect_rgb_graph_for_video)
            self.thread_record.daemon = True

        # open hand to be initial state
        self.bimanual_dex.set_both_hand_state("open", "open")

        # step world to make it ready
        for i in range(100):
            self.step()

        cprint("----------- World Configuration -----------", color="magenta", attrs=["bold"])
        cprint(f"usd_path: {usd_path}", "magenta")
        cprint(f"pos_x: {pos[0]}", "magenta")
        cprint(f"pos_y: {pos[1]}", "magenta")
        cprint("----------- World Configuration -----------", color="magenta", attrs=["bold"])

        cprint("World Ready!", "green", "on_green")

    def record_callback(self, step_size):

        if self.step_num % 5 == 0:

            joint_pos_L = self.bimanual_dex.dexleft.get_joint_positions()

            joint_pos_R = self.bimanual_dex.dexright.get_joint_positions()

            joint_state = np.array([*joint_pos_L, *joint_pos_R])

            rgb = self.env_camera.get_rgb_graph(save_or_not=False)

            point_cloud = self.env_camera.get_pointcloud_from_depth(
                show_original_pc_online=False,
                show_downsample_pc_online=False,
            )

            self.saving_data.append({ 
                "joint_state": joint_state,
                "image": rgb,
                "env_point_cloud": point_cloud,
                "garment_point_cloud":self.garment_pcd,
                "points_affordance_feature": self.points_affordance_feature,
            })

        self.step_num += 1


def FoldTops(pos, ori, usd_path, ground_material_usd, data_collection_flag, record_video_flag):

    env = FoldTops_Env(pos, ori, usd_path, ground_material_usd, record_video_flag)

    if record_video_flag:
        env.thread_record.start()

    # hide prim to get garment point cloud
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight"],
        visible=False,
    )
    for i in range(50):
        env.step()

    # Capture initial stage image (Stage 0)
    _save_stage_image(env, "stage_0")

    pcd, color = env.garment_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".ply"),
        real_time_watch=False,
    )
    rgb = env.garment_camera.get_rgb_graph(save_or_not=False)
    env.garment_pcd=pcd

    # ------------------ VLM Recognition ------------------ #
    client, model_name = _get_vlm_client()
    vlm_points_data = _ask_vlm_points(rgb, client, model_name)
    
    # Map labels to bboxes
    label_to_bbox = {}
    height, width, _ = rgb.shape
    for item in vlm_points_data:
        if item.get("label") and item.get("bbox_2d"):
             bbox_rel = item["bbox_2d"]
             # Convert 0-1000 relative to absolute pixels
             bbox_abs = [
                 bbox_rel[0] / 1000.0 * width,
                 bbox_rel[1] / 1000.0 * height,
                 bbox_rel[2] / 1000.0 * width,
                 bbox_rel[3] / 1000.0 * height
             ]
             label_to_bbox[item["label"]] = bbox_abs

    # Convert to 3D points
    us, vs, mask = _project_pcd_to_pixels(env, rgb, pcd)
    
    def get_3d_point(label):
        if label not in label_to_bbox:
            cprint(f"[WARNING] Label {label} not found in VLM output.", "yellow")
            return np.array([0, 0, 0]) # Default fallback
        bbox = label_to_bbox[label]
        # VLM output might be in [x1, y1, x2, y2]
        center_3d = _center_3d_for_bbox(bbox, us, vs, mask, pcd)
        if center_3d is None:
             cprint(f"[WARNING] No 3D point found for {label} bbox.", "yellow")
             return np.array([0, 0, 0])
        return center_3d

    # manipulation_points mapping:
    # [0]: left_cuff
    # [1]: right_shoulder
    # [2]: right_cuff
    # [3]: left_shoulder
    # [4]: left_hem
    # [5]: right_hem
    
    mp_labels = ["left_cuff", "right_shoulder", "right_cuff", "left_shoulder", "left_hem", "right_hem"]
    manipulation_points = np.zeros((6, 3))
    for i, label in enumerate(mp_labels):
        manipulation_points[i] = get_3d_point(label)

    # Compute points_similarity (affordance features)
    # We need to compute similarity for each of the 6 keypoints against the whole pcd
    # points_similarity shape should be (6, N)
    sim_list = []
    for i in range(6):
        # compute_similarity returns (N, 1)
        sim = compute_similarity(pcd, manipulation_points[i], sigma=0.1)
        sim_list.append(sim.reshape(1, -1)) # (1, N)
    
    points_similarity = np.vstack(sim_list) # (6, N)
    
    # ----------------------------------------------------- #

    # unhide
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight"],
        visible=True,
    )
    for i in range(50):
        env.step()

    # Original code replacement ends here, we use our own manipulation_points and points_similarity
    
    manipulation_points[0:4, 2] = 0.02
    manipulation_points[4:, 2] = 0.0

    # ---------------------- left hand ---------------------- #

    env.points_affordance_feature = normalize_columns(np.concatenate([points_similarity[0:1], points_similarity[0:1]], axis=0).T)

    env.bimanual_dex.dexleft.dense_step_action(target_pos=manipulation_points[0], target_ori=np.array([0.579, -0.579, -0.406, 0.406]), angular_type="quat")

    # if data_collection_flag:
    #     for i in range(20):
    #         env.step()
    #     env.record(task_name="Fold_Tops", stage_index=1)

    env.bimanual_dex.set_both_hand_state(left_hand_state="close", right_hand_state="None")

    left_sleeve_height = min(np.linalg.norm(manipulation_points[0][:2] - manipulation_points[3][:2]), 0.3)

    # print("left_sleeve_height: ", left_sleeve_height)

    lift_point_1 = np.array([manipulation_points[0][0], manipulation_points[0][1], left_sleeve_height])

    env.bimanual_dex.dexleft.dense_step_action(target_pos=lift_point_1, target_ori=np.array([0.579, -0.579, -0.406, 0.406]), angular_type="quat")

    lift_point_2 = np.array([manipulation_points[1][0], manipulation_points[1][1], left_sleeve_height])

    env.bimanual_dex.dexleft.dense_step_action(target_pos=lift_point_2, target_ori=np.array([0.579, -0.579, -0.406, 0.406]), angular_type="quat")

    env.bimanual_dex.set_both_hand_state(left_hand_state="open", right_hand_state="None")

    # if data_collection_flag:
    #     env.stop_record()

    env.garment.particle_material.set_gravity_scale(10.0)
    for i in range(200):
        env.step()
    env.garment.particle_material.set_gravity_scale(1.0) 

    # Capture Stage 1 Image
    # Hide hands
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight"],
        visible=False,
    )
    for i in range(50):
        env.step()
    
    _save_stage_image(env, "stage_1")
    
    # Show hands
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight"],
        visible=True,
    )

    env.bimanual_dex.dexleft.dense_step_action(target_pos=np.array([-0.6, 0.8, 0.5]), target_ori=np.array([0.579, -0.579, -0.406, 0.406]), angular_type="quat")


    # --------------------- right hand --------------------- #

    env.points_affordance_feature = normalize_columns(np.concatenate([points_similarity[2:3], points_similarity[2:3]], axis=0).T)

    env.bimanual_dex.dexright.dense_step_action(target_pos=manipulation_points[2], target_ori=np.array([0.406, -0.406, -0.579, 0.579]), angular_type="quat")

    # if data_collection_flag:
    #     for i in range(20):
    #         env.step()
    #     env.record(task_name="Fold_Tops", stage_index=2)

    env.bimanual_dex.set_both_hand_state(left_hand_state="None", right_hand_state="close")

    right_sleeve_height = min(np.linalg.norm(manipulation_points[2][:2] - manipulation_points[1][:2]), 0.3)

    # print("right_sleeve_height: ", right_sleeve_height)

    lift_point_1 = np.array([manipulation_points[2][0], manipulation_points[2][1], right_sleeve_height])

    env.bimanual_dex.dexright.dense_step_action(target_pos=lift_point_1, target_ori=np.array([0.406, -0.406, -0.579, 0.579]), angular_type="quat")

    lift_point_2 = np.array([manipulation_points[3][0], manipulation_points[3][1], right_sleeve_height])

    env.bimanual_dex.dexright.dense_step_action(target_pos=lift_point_2, target_ori=np.array([0.406, -0.406, -0.579, 0.579]), angular_type="quat")

    env.bimanual_dex.set_both_hand_state(left_hand_state="None", right_hand_state="open")

    # if data_collection_flag:
    #     env.stop_record()

    env.garment.particle_material.set_gravity_scale(10.0)
    for i in range(200):
        env.step()
    env.garment.particle_material.set_gravity_scale(1.0) 

    # Capture Stage 2 Image
    # Hide hands
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight"],
        visible=False,
    )
    for i in range(50):
        env.step()
        
    _save_stage_image(env, "stage_2")
    
    # Show hands
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight"],
        visible=True,
    )

    env.bimanual_dex.dexright.dense_step_action(target_pos=np.array([0.6, 0.8, 0.5]), target_ori=np.array([0.406, -0.406, -0.579, 0.579]), angular_type="quat")

    # --------------------- bottom-top --------------------- #    

    env.points_affordance_feature = normalize_columns(points_similarity[4:6].T)   

    env.bimanual_dex.dense_move_both_ik(
        left_pos=manipulation_points[4], 
        left_ori=np.array([0.579, -0.579, -0.406, 0.406]),
        right_pos=manipulation_points[5],
        right_ori=np.array([0.406, -0.406, -0.579, 0.579]),
    )

    # if data_collection_flag:
    #     for i in range(20):
    #         env.step()
    #     env.record(task_name="Fold_Tops", stage_index=3)

    env.bimanual_dex.set_both_hand_state(left_hand_state="close", right_hand_state="close")

    lift_height = manipulation_points[3][1] - manipulation_points[4][1]

    # print("lift_height: ", lift_height)

    lift_point_1 = np.array([manipulation_points[4][0], manipulation_points[4][1], lift_height/2])
    lift_point_2 = np.array([manipulation_points[5][0], manipulation_points[5][1], lift_height/2])

    env.bimanual_dex.dense_move_both_ik(
        left_pos=lift_point_1,
        left_ori=np.array([0.579, -0.579, -0.406, 0.406]),
        right_pos=lift_point_2,
        right_ori=np.array([0.406, -0.406, -0.579, 0.579]),
    )

    push_point_1 = np.array([manipulation_points[3][0], manipulation_points[3][1]+0.1, min(lift_height/2, 0.2)])
    push_point_2 = np.array([manipulation_points[1][0], manipulation_points[1][1]+0.1, min(lift_height/2, 0.2)])

    env.bimanual_dex.dense_move_both_ik(
        left_pos=push_point_1,
        left_ori=np.array([0.579, -0.579, -0.406, 0.406]),
        right_pos=push_point_2,
        right_ori=np.array([0.406, -0.406, -0.579, 0.579]),
    )

    env.bimanual_dex.set_both_hand_state(left_hand_state="open", right_hand_state="open")

    # if data_collection_flag:
    #     env.stop_record()

    env.garment.particle_material.set_gravity_scale(10.0)
    for i in range(100):
        env.step()
    env.garment.particle_material.set_gravity_scale(1.0)

    # Capture Stage 3 Image
    # Hide hands
    set_prim_visible_group(
        prim_path_list=["/World/DexLeft", "/World/DexRight"],
        visible=False,
    )
    for i in range(50):
        env.step()
        
    _save_stage_image(env, "stage_3")
    
    # Show hands (though they are hidden again shortly after, keeping logic consistent or just letting next lines handle it)
    # The next lines are:
    # dexleft_prim = prims_utils.get_prim_at_path("/World/DexLeft")
    # ... set_prim_visibility(dexleft_prim, False)
    # So we don't strictly need to show them, but for correctness of "capture image without hands" logic block:
    
    # Actually, the original code immediately hides them again:
    # dexleft_prim = prims_utils.get_prim_at_path("/World/DexLeft")
    # ... set_prim_visibility(dexleft_prim, False)
    
    # So I can just leave them hidden.
    
    dexleft_prim = prims_utils.get_prim_at_path("/World/DexLeft")
    dexright_prim = prims_utils.get_prim_at_path("/World/DexRight")
    set_prim_visibility(dexleft_prim, False)
    set_prim_visibility(dexright_prim, False)

    for i in range(50):
        env.step()   

    success=True
    points,*_=env.model.get_manipulation_points(pcd,[554,1540,1014,1385])
    boundary=[points[0][0]-0.05,points[1][0]+0.05,points[3][1]-0.1,points[2][1]+0.1]
    pcd_end,_=env.garment_camera.get_point_cloud_data_from_segment(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".ply"),
        real_time_watch=False,
    )
    success=judge_pcd(pcd_end,boundary,threshold=0.12)
    cprint(f"final result: {success}", color="green", on_color="on_green")

    # if you wanna create gif, use this code. Need Cooperation with thread.
    # if record_video_flag and success:
    #     if not os.path.exists("Data/Fold_Tops/video"):
    #         os.makedirs("Data/Fold_Tops/video")
    #     env.env_camera.create_mp4(get_unique_filename("Data/Fold_Tops/video/video", ".mp4"))

    if data_collection_flag:
        # write into .log file
        with open("Data/Fold_Tops/data_collection_log.txt", "a") as f:
            f.write(f"result:{success}  usd_path:{env.garment.usd_path}  pos_x:{pos[0]}  pos_y:{pos[1]}\n")
        if success:
            #env.record_to_npz()
            if not os.path.exists("Data/Fold_Tops/final_state_pic"):
                os.makedirs("Data/Fold_Tops/final_state_pic")
            env.env_camera.get_rgb_graph(save_or_not=True,save_path=get_unique_filename("Data/Fold_Tops/final_state_pic/img",".png"))


if __name__=="__main__":

    args=parse_args_record()

    # initial setting
    pos = np.array([0.0, 0.8, 0.2])
    ori = np.array([0.0, 0.0, 0.0])
    usd_path = args.usd_path if hasattr(args, 'usd_path') and args.usd_path else None

    # Check for env_random_flag (used by Collect.sh) or garment_random_flag (legacy)
    should_randomize = (hasattr(args, 'env_random_flag') and args.env_random_flag) or \
                      (hasattr(args, 'garment_random_flag') and args.garment_random_flag)

    if should_randomize:
        np.random.seed(int(time.time()))
        x = np.random.uniform(-0.1, 0.1) # changeable
        y = np.random.uniform(0.7, 0.9) # changeable
        pos = np.array([x,y,0.0])
        ori = np.array([0.0, 0.0, 0.0])
        
        # Only randomly select garment if usd_path was not provided
        if usd_path is None:
            Base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            assets_lists = os.path.join(Base_dir,"Model_HALO/GAM/checkpoints/Tops_LongSleeve/assets_training_list.txt")
            assets_list = []
            with open(assets_lists,"r",encoding='utf-8') as f:
                for line in f:
                    clean_line = line.rstrip('\n')
                    assets_list.append(clean_line)
            usd_path=os.getcwd() + "/" + np.random.choice(assets_list)

    FoldTops(pos, ori, usd_path, args.ground_material_usd, args.data_collection_flag, args.record_video_flag)

    if args.data_collection_flag:
        simulation_app.close()
    else:
        while simulation_app.is_running():
            simulation_app.update()

simulation_app.close()
