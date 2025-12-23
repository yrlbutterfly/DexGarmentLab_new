from isaacsim import SimulationApp
# 创建 Isaac Sim 仿真应用，headless=True 表示使用无界面模式（适合批量验证 / 集群）
# - 若需要在本地可视化调试，可以改为 headless=False；但服务器/集群上建议保持为 True
simulation_app = SimulationApp({"headless": False})

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
import atexit
import signal
import json
import base64
import re
from typing import Dict, List, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # 仅用于触发 3D 投影视图注册
from openai import OpenAI

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
from Env_Config.Utils_Project.Point_Cloud_Manip import compute_similarity
from Env_Config.Utils_Project.Parse import parse_args_val
from Env_Config.Utils_Project.Position_Judge import judge_pcd
from Env_Config.Room.Object_Tools import set_prim_visible_group, delete_prim_group
from Model_HALO.GAM.GAM_Encapsulation import GAM_Encapsulation

# ------------------------- #
#   VLM & 2D-3D 工具函数    #
# ------------------------- #

# # 通过 VLM 生成折叠 plan + keypoint bbox 的提示词
# VLM_FOLD_PROMPT = (
#     "<image>\n"
#     "The image contains a piece of clothing. Please infer how it should be folded.\n\n"
#     "Your output must be a single JSON array containing two objects: one with a \"plan\" field and one with a \"points\" field.\n\n"
#     "------------------------------------------------ OUTPUT FORMAT (STRICT) ------------------------------------------------\n\n"
#     "Your final output must always be in the following structure:\n\n"
#     "[\n\n"
#     "  { \"plan\": <plan_output> },\n\n"
#     "  { \"points\": <points_output> }\n\n"
#     "]\n\n"
#     "Where:\n\n"
#     "- <plan_output> is either:\n\n"
#     "  1) A list of folding steps, OR\n\n"
#     "  2) The string \"already finish folding\" if no folding is needed.\n\n"
#     "- <points_output> is a list of keypoint bounding box entries mentioned in the plan.\n\n"
#     "------------------------------------------------ FOLDING PLAN RULES ------------------------------------------------\n\n"
#     "If folding is required, the plan must be a list. The i-th element represents the i-th folding action:\n\n"
#     "{\n\n"
#     "  \"left\": {\"from\": <keypoint_name>, \"to\": <keypoint_name>},\n\n"
#     "  \"right\": {\"from\": <keypoint_name>, \"to\": <keypoint_name>}\n\n"
#     "}\n\n"
#     "- If a robotic arm does not need to operate during a step, assign null to both \"from\" and \"to\" for that arm.\n\n"
#     "- The folding plan must be practical for robotic manipulation:\n\n"
#     "  actions should avoid arm collisions, use clear step-by-step motions rather than\n\n"
#     "  merging multiple operations, and follow common-sense garment-folding practices.\n\n"
#     "Valid keypoint names include:\n\n"
#     "left_cuff, right_cuff\n\n"
#     "left_collar, right_collar, center_collar\n\n"
#     "left_hem, right_hem, center_hem\n\n"
#     "left_armpit, right_armpit\n\n"
#     "left_shoulder, right_shoulder\n\n"
#     "left_waist, right_waist\n\n"
#     "------------------------------------------------ STANDARD SHIRT FOLDING PROCEDURE ------------------------------------------------\n\n"
#     "The complete folding plan contains three standard steps:\n\n"
#     "Step 1 — Fold the left sleeve inward:\n\n"
#     "    Move: left_cuff  -> right_shoulder\n\n"
#     "    Right arm stays idle (null -> null)\n\n"
#     "Step 2 — Fold the right sleeve inward:\n\n"
#     "    Move: right_cuff -> left_shoulder\n\n"
#     "    Left arm stays idle (null -> null)\n\n"
#     "Step 3 — Fold the bottom hem upward:\n\n"
#     "    Move left_hem  -> left_shoulder\n\n"
#     "    Move right_hem -> right_shoulder\n\n"
#     "    (Both arms operate simultaneously)\n\n"
#     "------------------------------------------------ FOLDING COMPLETION CONDITION ------------------------------------------------\n\n"
#     "Return \"already finish folding\" if the garment is already folded.\n\n"
#     "A garment is considered \"already folded\" when:\n\n"
#     "- Most garment pixels or keypoints fall inside a compact rectangular region,\n\n"
#     "- Both sleeves have already been folded inward,\n\n"
#     "- The hem is lifted so the shirt forms a clean rectangle.\n\n"
#     "If these conditions are met, no folding plan is needed and the output should be:\n\n"
#     "[\n\n"
#     "  { \"plan\": \"already finish folding\" },\n\n"
#     "  { \"points\": [] }\n\n"
#     "]\n\n"
#     "------------------------------------------------ KEYPOINT BOUNDING BOX RULES ------------------------------------------------\n\n"
#     "Each keypoint entry must follow:\n\n"
#     "{\n\n"
#     "  \"label\": \"<keypoint_name>\",\n\n"
#     "  \"bbox\": [x_min, y_min, x_max, y_max]\n\n"
#     "}\n\n"
#     "Where:\n\n"
#     "- x_min, y_min = top-left corner of bounding box\n\n"
#     "- x_max, y_max = bottom-right corner of bounding box\n\n"
#     "------------------------------------------------ COMBINED EXAMPLE OUTPUT ------------------------------------------------\n\n"
#     "Example folding plan(the numbers in bboxes are random):\n\n"
#     "[\n\n"
#     "  {\n\n"
#     "    \"plan\": [\n\n"
#     "      {\n\n"
#     "        \"left\": {\"from\": \"left_cuff\", \"to\": \"right_shoulder\"},\n\n"
#     "        \"right\": {\"from\": null, \"to\": null}\n\n"
#     "      },\n\n"
#     "      {\n\n"
#     "        \"left\": {\"from\": null, \"to\": null},\n\n"
#     "        \"right\": {\"from\": \"right_cuff\", \"to\": \"left_shoulder\"}\n\n"
#     "      },\n\n"
#     "      {\n\n"
#     "        \"left\": {\"from\": \"left_hem\", \"to\": \"left_shoulder\"},\n\n"
#     "        \"right\": {\"from\": \"right_hem\", \"to\": \"right_shoulder\"}\n\n"
#     "      }\n\n"
#     "    ]\n\n"
#     "  },\n\n"
#     "  {\n\n"
#     "    \"points\": [\n\n"
#     "      {\"label\": \"left_cuff\", \"bbox\": [90, 180, 140, 230]},\n\n"
#     "      {\"label\": \"right_cuff\", \"bbox\": [290, 190, 340, 240]},\n\n"
#     "      {\"label\": \"left_hem\", \"bbox\": [100, 250, 150, 300]},\n\n"
#     "      {\"label\": \"right_hem\", \"bbox\": [250, 250, 300, 300]},\n\n"
#     "      {\"label\": \"left_shoulder\", \"bbox\": [100, 100, 150, 150]},\n\n"
#     "      {\"label\": \"right_shoulder\", \"bbox\": [250, 100, 300, 150]}\n\n"
#     "    ]\n\n"
#     "  }\n\n"
#     "]\n\n"
#     "Example when no folding is needed:\n\n"
#     "[\n\n"
#     "  { \"plan\": \"already finish folding\" },\n\n"
#     "  { \"points\": [] }\n\n"
#     "]\n\n"
#     "------------------------------------------------\n\n"
#     "Do not output anything except the final JSON array.\n"
# )

VLM_FOLD_PROMPT = "The image contains a piece of clothing. Infer the NEXT folding action.\n\nOutput a single JSON array with two objects:\n[\n  { \"plan\": <plan_output> },\n  { \"points\": <points_output> }\n]\n\nPLAN OUTPUT:\n- Either a list containing EXACTLY ONE action (the next step), OR the string \"already finish folding\".\n- If already folded, output:\n[\n  { \"plan\": \"already finish folding\" },\n  { \"points\": [] }\n]\n\nACTION FORMAT (single step):\n{\n  \"left\":  { \"from\": <keypoint>, \"to\": <keypoint> },\n  \"right\": { \"from\": <keypoint>, \"to\": <keypoint> }\n}\n- If an arm is idle: {\"from\": null, \"to\": null}\n\nVALID KEYPOINT NAMES (ONLY THESE 6):\nleft_cuff, right_cuff, left_hem, right_hem, left_shoulder, right_shoulder\n\nPOINTS OUTPUT:\n- \"points\" MUST include ONLY the keypoints referenced by the action in \"plan\".\n- Each entry:\n{ \"label\": \"<keypoint>\", \"bbox\": [x_min, y_min, x_max, y_max] }\n\nSTANDARD SHIRT FOLDING REFERENCE (FULL PLAN FROM A COMPLETELY FLAT, UNFOLDED SHIRT):\nStep 1: left_cuff  -> right_shoulder (right arm idle)\nStep 2: right_cuff -> left_shoulder  (left arm idle)\nStep 3: left_hem -> left_shoulder AND right_hem -> right_shoulder\nIMPORTANT: Use this reference ONLY to decide what the NEXT step should be. Do NOT output all steps.\n\nDo not output anything except the final JSON array.\n"

def _get_vlm_client(base_url: str, model_name: str):
    """
    根据传入的 base_url / model_name 构造 OpenAI 兼容的 VLM client。
    """
    api_key = os.environ.get("VLM_API_KEY", "EMPTY")
    client = OpenAI(base_url=base_url, api_key=api_key)
    return client, model_name


def _encode_rgb_to_data_url(rgb: np.ndarray) -> str:
    """
    将 RGB 图像编码为 base64 data URL，供 OpenAI 接口中的 image_url 使用。
    """
    rgb_uint8 = rgb.astype("uint8")
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(".png", bgr)
    if not success:
        raise RuntimeError("无法将 RGB 图像编码为 PNG。")
    img_bytes = buf.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"


def _parse_vlm_output(raw_text: str) -> Dict[str, object]:
    """
    解析 VLM 的原始文本输出，抽取 JSON 数组，并返回 plan 与 points 两个字段。
    """
    try:
        data = json.loads(raw_text)
    except Exception:
        match = re.search(r"\[.*\]", raw_text, re.S)
        if not match:
            raise ValueError(f"无法在 VLM 输出中找到 JSON 数组，原始输出:\n{raw_text}")
        data = json.loads(match.group(0))

    if not isinstance(data, list) or len(data) != 2:
        raise ValueError(f"VLM 输出 JSON 结构不符合预期，应为长度为 2 的列表，但得到: {data}")

    plan_obj = data[0] if isinstance(data[0], dict) else {}
    points_obj = data[1] if isinstance(data[1], dict) else {}
    plan = plan_obj.get("plan", None)
    points = points_obj.get("points", None)
    return {"plan": plan, "points": points}


def _convert_vlm_points_to_absolute(
    points: Optional[List[Dict[str, object]]],
    img_w: int,
    img_h: int
) -> List[Dict[str, object]]:
    """
    将 VLM 输出的 points (0~1000 相对坐标) 转换为绝对像素坐标。
    """
    if not points:
        return []
    
    abs_points = []
    for item in points:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        bbox = item.get("bbox")
        if bbox is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
            
        x_min_rel, y_min_rel, x_max_rel, y_max_rel = [float(v) for v in bbox]
        
        x_min = int(x_min_rel / 1000.0 * img_w)
        x_max = int(x_max_rel / 1000.0 * img_w)
        y_min = int(y_min_rel / 1000.0 * img_h)
        y_max = int(y_max_rel / 1000.0 * img_h)
        
        abs_points.append({
            "label": label,
            "bbox": [x_min, y_min, x_max, y_max]
        })
    return abs_points


def _ask_vlm_plan_and_points(rgb: np.ndarray, client, model_name: str) -> Dict[str, object]:
    """
    给定当前衣物 RGB 图像，调用本地多模态 VLM，返回解析后的 plan 与 points。
    """
    image_data_url = _encode_rgb_to_data_url(rgb)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                    {"type": "text", "text": "\n" + VLM_FOLD_PROMPT},
                ],
            }
        ],
        max_tokens=1024,
    )
    raw = response.choices[0].message.content
    result = _parse_vlm_output(raw)

    # 转换坐标为绝对像素
    points = result.get("points")
    if points:
        h, w = rgb.shape[:2]
        cprint(f"[DEBUG] Converting VLM points. Image size: {w}x{h}", "yellow")
        cprint(f"[DEBUG] Points before conversion: {points}", "yellow")
        result["points"] = _convert_vlm_points_to_absolute(points, w, h)
        cprint(f"[DEBUG] Points after conversion: {result['points']}", "yellow")
        
    return result


def _debug_save_vlm_output(
    debug_dir: str,
    step_idx: int,
    vlm_result: Dict[str, object],
) -> None:
    """
    在 debug 模式下，将每轮 VLM 的原始解析结果追加写入一个文本文件。
    """
    os.makedirs(debug_dir, exist_ok=True)
    log_path = os.path.join(debug_dir, "vlm_results.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        # 使用 ensure_ascii=False，方便直接阅读中文内容
        json_str = json.dumps(vlm_result, ensure_ascii=False)
        f.write(f"step {step_idx}: {json_str}\n")


def _debug_save_vlm_rgb_with_bbox(
    rgb: np.ndarray,
    vlm_points: Optional[List[Dict[str, object]]],
    save_path: str,
) -> None:
    """
    在 RGB 图像上可视化 VLM 输出的 label + bbox，并保存到指定路径。
    """
    if vlm_points is None:
        vlm_points = []

    # 转成 BGR，方便用 OpenCV 可视化
    img = rgb.astype("uint8").copy()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_h, img_w = img_bgr.shape[:2]

    for item in vlm_points:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        bbox = item.get("bbox")
        if (
            label is None
            or bbox is None
            or not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
        ):
            continue
        # VLM 输出的 bbox 已被转换为绝对像素坐标
        x_min, y_min, x_max, y_max = [int(v) for v in bbox]
        # 画 bbox
        cv2.rectangle(
            img_bgr,
            (x_min, y_min),
            (x_max, y_max),
            color=(0, 255, 0),
            thickness=2,
        )
        # 在框的左上角写上 label
        cv2.putText(
            img_bgr,
            str(label),
            (x_min, max(y_min - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img_bgr)


def _debug_save_vlm_rgb_raw(
    rgb: np.ndarray,
    save_path: str,
) -> None:
    """
    保存原始 RGB 图像（不带 bbox），用于对比 VLM 输入与可视化结果。
    """
    img = rgb.astype("uint8").copy()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img_bgr)


def get_rgb_index(env, rgb: np.ndarray, point: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
    """
    将 3D 点（世界坐标）投影到 garment_camera 图像平面，返回像素坐标 (u, v)。
    """
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


def _project_pcd_to_pixels(
    env,
    rgb: np.ndarray,
    pcd: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将当前衣物点云中的每个 3D 点投影到 RGB 图像上，得到像素坐标 (u, v) 以及可见性 mask。
    """
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


def _label_to_bbox_px_map(
    rgb: np.ndarray,
    points_list: Optional[List[Dict[str, object]]],
) -> Dict[str, List[float]]:
    """
    将 VLM 输出的 points_list（bbox 范围为 [0,1000] 的相对坐标）转换为像素坐标 bbox 映射。
    """
    img_h, img_w = rgb.shape[:2]
    label_to_bbox: Dict[str, List[float]] = {}
    if not isinstance(points_list, list):
        return label_to_bbox
    for item in points_list:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        bbox = item.get("bbox")
        if (
            label is None
            or bbox is None
            or not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
        ):
            continue
        x_min, y_min, x_max, y_max = [float(b) for b in bbox]
        label_to_bbox[str(label)] = [x_min, y_min, x_max, y_max]
    return label_to_bbox


def _center_3d_for_bbox(
    bbox: List[float],
    us: np.ndarray,
    vs: np.ndarray,
    mask: np.ndarray,
    pcd: np.ndarray,
) -> Optional[np.ndarray]:
    """
    给定像素坐标系下的 bbox，在点云里找出投影落在 bbox 内且可见(mask=True)的点，并返回这些点的 3D 质心。
    """
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


def _center_3d_for_label_from_bbox(
    label_name: Optional[str],
    label_to_bbox: Dict[str, List[float]],
    us: np.ndarray,
    vs: np.ndarray,
    mask: np.ndarray,
    pcd: np.ndarray,
) -> Optional[np.ndarray]:
    """
    根据 label 对应的 2D bbox，在点云中找出投影落入 bbox 内的点，并返回它们的 3D 质心。
    """
    if label_name is None:
        return None
    bbox = label_to_bbox.get(str(label_name))
    if bbox is None:
        return None
    return _center_3d_for_bbox(bbox=bbox, us=us, vs=vs, mask=mask, pcd=pcd)


def motion_plan_reset_hands_home(env, settle_steps: int = 20) -> None:
    """
    将双手 reset 回固定 home pose (同时移动)。
    """
    left_quat = np.array([0.579, -0.579, -0.406, 0.406])
    right_quat = np.array([0.406, -0.406, -0.579, 0.579])

    env.bimanual_dex.dense_move_both_ik(
        left_pos=np.array([-0.6, 0.8, 0.5]),
        left_ori=left_quat,
        right_pos=np.array([0.6, 0.8, 0.5]),
        right_ori=right_quat,
    )

    for _ in range(int(settle_steps)):
        env.step()


# ----------------------------------- #
#   Motion Planning Primitives (Hardcoded)
# ----------------------------------- #

def mp_fold_left_sleeve(env, start_pos: np.ndarray, target_pos: np.ndarray, shoulder_pos: Optional[np.ndarray] = None):
    """
    左袖折叠逻辑 (参考 Fold_Tops_origin_Env.py)
    """
    cprint("[MP] Executing Left Sleeve Fold", "cyan")
    
    # 固定的左手抓取姿态 quaternion
    left_quat = np.array([0.579, -0.579, -0.406, 0.406])
    
    # 1. 移动到起始点 (Cuff)
    env.bimanual_dex.dexleft.dense_step_action(
        target_pos=start_pos, 
        target_ori=left_quat, 
        angular_type="quat"
    )
    
    # 2. 闭合左手 (Grasp)
    env.bimanual_dex.set_both_hand_state(left_hand_state="close", right_hand_state="None")
    for _ in range(10): env.step()

    # 3. 计算抬起高度
    # 参考 origin: left_sleeve_height = min(dist(cuff, shoulder), 0.3)
    # 如果 shoulder_pos 未知，默认 0.3
    if shoulder_pos is not None:
        dist = np.linalg.norm(start_pos[:2] - shoulder_pos[:2])
        lift_height = min(dist, 0.3)
    else:
        lift_height = 0.3
    
    # 4. 抬起 (Lift)
    lift_point_1 = np.array([start_pos[0], start_pos[1], lift_height])
    env.bimanual_dex.dexleft.dense_step_action(
        target_pos=lift_point_1, 
        target_ori=left_quat, 
        angular_type="quat"
    )

    # 5. 移动到目标点 (Move to target/Shoulder)
    lift_point_2 = np.array([target_pos[0], target_pos[1], lift_height])
    env.bimanual_dex.dexleft.dense_step_action(
        target_pos=lift_point_2, 
        target_ori=left_quat, 
        angular_type="quat"
    )

    # 6. 松开左手 (Release)
    env.bimanual_dex.set_both_hand_state(left_hand_state="open", right_hand_state="None")
    for _ in range(10): env.step()

    # 7. 稳定布料 (Stabilize)
    env.garment.particle_material.set_gravity_scale(10.0)
    for _ in range(200): env.step()
    env.garment.particle_material.set_gravity_scale(1.0) 

    # 8. Reset Left Hand
    env.bimanual_dex.dexleft.dense_step_action(
        target_pos=np.array([-0.6, 0.8, 0.5]), 
        target_ori=left_quat, 
        angular_type="quat"
    )


def mp_fold_right_sleeve(env, start_pos: np.ndarray, target_pos: np.ndarray, shoulder_pos: Optional[np.ndarray] = None):
    """
    右袖折叠逻辑 (参考 Fold_Tops_origin_Env.py)
    """
    cprint("[MP] Executing Right Sleeve Fold", "cyan")

    # 固定的右手抓取姿态 quaternion
    right_quat = np.array([0.406, -0.406, -0.579, 0.579])

    # 1. 移动到起始点 (Cuff)
    env.bimanual_dex.dexright.dense_step_action(
        target_pos=start_pos, 
        target_ori=right_quat, 
        angular_type="quat"
    )

    # 2. 闭合右手 (Grasp)
    env.bimanual_dex.set_both_hand_state(left_hand_state="None", right_hand_state="close")
    for _ in range(10): env.step()

    # 3. 计算抬起高度
    if shoulder_pos is not None:
        dist = np.linalg.norm(start_pos[:2] - shoulder_pos[:2])
        lift_height = min(dist, 0.3)
    else:
        lift_height = 0.3

    # 4. 抬起 (Lift)
    lift_point_1 = np.array([start_pos[0], start_pos[1], lift_height])
    env.bimanual_dex.dexright.dense_step_action(
        target_pos=lift_point_1, 
        target_ori=right_quat, 
        angular_type="quat"
    )

    # 5. 移动到目标点 (Move to target/Shoulder)
    lift_point_2 = np.array([target_pos[0], target_pos[1], lift_height])
    env.bimanual_dex.dexright.dense_step_action(
        target_pos=lift_point_2, 
        target_ori=right_quat, 
        angular_type="quat"
    )

    # 6. 松开右手 (Release)
    env.bimanual_dex.set_both_hand_state(left_hand_state="None", right_hand_state="open")
    for _ in range(10): env.step()

    # 7. 稳定布料 (Stabilize)
    env.garment.particle_material.set_gravity_scale(10.0)
    for _ in range(200): env.step()
    env.garment.particle_material.set_gravity_scale(1.0) 

    # 8. Reset Right Hand
    env.bimanual_dex.dexright.dense_step_action(
        target_pos=np.array([0.6, 0.8, 0.5]), 
        target_ori=right_quat, 
        angular_type="quat"
    )


def mp_fold_bottom_up(
    env, 
    left_start: np.ndarray, 
    right_start: np.ndarray, 
    left_target: np.ndarray, 
    right_target: np.ndarray
):
    """
    底部向上折叠逻辑 (双臂操作, 参考 Fold_Tops_origin_Env.py)
    """
    cprint("[MP] Executing Bottom-Up Fold", "cyan")

    left_quat = np.array([0.579, -0.579, -0.406, 0.406])
    right_quat = np.array([0.406, -0.406, -0.579, 0.579])

    # 1. 双手移动到起始点 (Hems)
    env.bimanual_dex.dense_move_both_ik(
        left_pos=left_start,
        left_ori=left_quat,
        right_pos=right_start,
        right_ori=right_quat,
    )

    # 2. 双手闭合
    env.bimanual_dex.set_both_hand_state(left_hand_state="close", right_hand_state="close")
    for _ in range(10): env.step()

    # 3. 计算抬起高度
    # origin: lift_height = left_shoulder.y - left_hem.y
    # 这里用 target.y - start.y 近似，或者直接计算两点距离
    dist_y = abs(left_target[1] - left_start[1])
    lift_height = dist_y

    # 4. 抬起 (Lift to mid-air)
    # origin logic: lift_height/2
    mid_z = max(lift_height / 2.0, 0.1) # 避免太低
    lift_point_L = np.array([left_start[0], left_start[1], mid_z])
    lift_point_R = np.array([right_start[0], right_start[1], mid_z])

    env.bimanual_dex.dense_move_both_ik(
        left_pos=lift_point_L,
        left_ori=left_quat,
        right_pos=lift_point_R,
        right_ori=right_quat,
    )

    # 5. 推送到目标位置 (Push/Fold)
    # origin logic: target_y + 0.1, z = min(lift_height/2, 0.2)
    # 我们直接使用 target 坐标，稍微加一点 y 偏移以模拟覆盖
    final_z = min(mid_z, 0.2)
    push_point_L = np.array([left_target[0], left_target[1] + 0.1, final_z])
    push_point_R = np.array([right_target[0], right_target[1] + 0.1, final_z])

    env.bimanual_dex.dense_move_both_ik(
        left_pos=push_point_L,
        left_ori=left_quat,
        right_pos=push_point_R,
        right_ori=right_quat,
    )

    # 6. 松开双手
    env.bimanual_dex.set_both_hand_state(left_hand_state="open", right_hand_state="open")
    #for _ in range(10): env.step()

    # 7. 稳定布料 (Stabilize)
    env.garment.particle_material.set_gravity_scale(10.0)
    for _ in range(200): env.step()
    env.garment.particle_material.set_gravity_scale(1.0)

    # 8. Reset Both Hands (reuse existing function logic or call it)
    motion_plan_reset_hands_home(env, 20)


def execute_vlm_motion_plan(
    env,
    rgb: np.ndarray,
    pcd: np.ndarray,
    plan_step: Dict[str, object],
    points_list: Optional[List[Dict[str, object]]],
) -> bool:
    """
    根据 VLM 的 plan_step 解析意图，并调用对应的 Motion Planning 动作。
    """
    # 1. 建立 label -> 3D Point 的映射
    label_to_bbox = _label_to_bbox_px_map(rgb, points_list)
    us, vs, mask = _project_pcd_to_pixels(env, rgb, pcd)

    def _get_point(label_name):
        return _center_3d_for_label_from_bbox(label_name, label_to_bbox, us, vs, mask, pcd)

    left_cfg = plan_step.get("left", {}) or {}
    right_cfg = plan_step.get("right", {}) or {}

    left_from_label = left_cfg.get("from")
    left_to_label = left_cfg.get("to")
    right_from_label = right_cfg.get("from")
    right_to_label = right_cfg.get("to")

    # 2. 决策逻辑
    # 如果左右手都有 from/to，则认为是双臂动作 (Bottom-Up)
    if left_from_label and left_to_label and right_from_label and right_to_label:
        l_start = _get_point(left_from_label)
        l_end = _get_point(left_to_label)
        r_start = _get_point(right_from_label)
        r_end = _get_point(right_to_label)

        if l_start is None or l_end is None or r_start is None or r_end is None:
            cprint("[MP] Missing keypoints for dual fold. Skipping.", "yellow")
            return False
        
        mp_fold_bottom_up(env, l_start, r_start, l_end, r_end)
        return True

    # 如果只有左手有动作
    elif left_from_label and left_to_label:
        start = _get_point(left_from_label)
        end = _get_point(left_to_label)
        # 尝试获取 shoulder 参考点（用于计算高度），通常目标点就是 shoulder，或者显式查找 left_shoulder
        # 若 VLM plan 中 target 是 right_shoulder (Step 1)，则参考点可以是 left_shoulder
        shoulder_ref = _get_point("left_shoulder") 

        if start is None or end is None:
            cprint("[MP] Missing keypoints for left fold. Skipping.", "yellow")
            return False
        
        mp_fold_left_sleeve(env, start, end, shoulder_ref)
        return True

    # 如果只有右手有动作
    elif right_from_label and right_to_label:
        start = _get_point(right_from_label)
        end = _get_point(right_to_label)
        shoulder_ref = _get_point("right_shoulder")

        if start is None or end is None:
            cprint("[MP] Missing keypoints for right fold. Skipping.", "yellow")
            return False
        
        mp_fold_right_sleeve(env, start, end, shoulder_ref)
        return True

    else:
        cprint(f"[MP] Unrecognized plan step: {plan_step}", "yellow")
        return False


class FoldTops_Env(BaseEnv):
    """
    Motion Planning 版本的上衣折叠环境。
    区别于 dp 版本，这里不加载 SADP_G 策略模型，而是直接使用硬编码的运动原语。
    """
    def __init__(
        self, 
        pos:np.ndarray=None, 
        ori:np.ndarray=None, 
        usd_path:str=None, 
        ground_material_usd:str=None,
        record_video_flag:bool=False, 
    ):
        super().__init__()
        
        # Add Assets
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
        )
        
        self.garment = Particle_Garment(
            self.world, 
            pos=np.array([0, 3.0, 0.6]),
            ori=np.array([0.0, 0.0, 0.0]),
            usd_path="Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_Top608/TCLC_Top608_obj.usd" if usd_path is None else usd_path,
            contact_offset=0.012,             
            rest_offset=0.010,                
            particle_contact_offset=0.012,    
            fluid_rest_offset=0.010,
            solid_rest_offset=0.010,
        )

        self.bimanual_dex = Bimanual_Ur10e(
            self.world,
            dexleft_pos=np.array([-0.8, 0.0, 0.5]),
            dexleft_ori=np.array([0.0, 0.0, 0.0]),
            dexright_pos=np.array([0.8, 0.0, 0.5]),
            dexright_ori=np.array([0.0, 0.0, 0.0]),
        )

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
        
        # 评估用模型
        self.model = GAM_Encapsulation(catogory="Tops_LongSleeve") 
        
        self.reset()
        
        self.garment.set_pose(pos=np.array([pos[0], pos[1], 0.2]), ori=ori)
        self.position = [pos[0], pos[1], 0.2]
        self.orientation = ori
        
        self.garment_camera.initialize(
            segment_pc_enable=True, 
            segment_prim_path_list=["/World/Garment/garment"],
            camera_params_enable=True,
        )
        
        self.env_camera.initialize(depth_enable=True)
        
        if record_video_flag:
            self.thread_record = threading.Thread(target=self.env_camera.collect_rgb_graph_for_vedio)
            self.thread_record.daemon = True
        
        self.bimanual_dex.set_both_hand_state("open", "open")

        for i in range(100):
            self.step()
            
        cprint("----------- World Configuration (MP) -----------", color="magenta", attrs=["bold"])
        cprint(f"usd_path: {usd_path}", "magenta")
        cprint("----------- World Configuration (MP) -----------", color="magenta", attrs=["bold"])
        cprint("World Ready!", "green", "on_green")


def FoldTops(
    pos,
    ori,
    usd_path,
    ground_material_usd,
    validation_flag,
    record_video_flag,
    # 保留参数以兼容 dp 版本的调用签名，但 MP 模式下可能不使用
    training_data_num,
    stage_1_checkpoint_num,
    stage_2_checkpoint_num,
    stage_3_checkpoint_num,
    vlm_base_url,
    vlm_model_name,
    debug_flag,
    debug_dir,
):
    """
    MP 版本的任务入口。
    """
    env = FoldTops_Env(
        pos,
        ori,
        usd_path,
        ground_material_usd,
        record_video_flag,
    )

    # 视频录制逻辑
    enable_video_record = bool(record_video_flag or debug_flag)
    if enable_video_record:
        if not hasattr(env, "thread_record") or env.thread_record is None:
            env.thread_record = threading.Thread(
                target=env.env_camera.collect_rgb_graph_for_vedio
            )
            env.thread_record.daemon = True
        env.env_camera.capture = True
        if not env.thread_record.is_alive():
            env.thread_record.start()

    vlm_client, vlm_model_name = _get_vlm_client(vlm_base_url, vlm_model_name)

    # Debug 目录设置
    debug_rgb_dir = None
    debug_video_dir = None
    if debug_flag:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        default_debug_root = os.path.join("Data", "Fold_Tops_Validation_HALO_MP", "debug")
        debug_root = debug_dir if debug_dir not in (None, "") else default_debug_root
        base_debug_dir = os.path.join(debug_root, timestamp)
        debug_rgb_dir = os.path.join(base_debug_dir, "vlm_rgb")
        debug_video_dir = os.path.join(base_debug_dir, "video")
        os.makedirs(debug_rgb_dir, exist_ok=True)

    # 视频保存逻辑
    video_output_path = None
    if enable_video_record:
        if debug_video_dir is not None:
            out_dir = debug_video_dir
        else:
            out_dir = os.path.join("Data", "Fold_Tops_Validation_HALO_MP", "video")
        
        os.makedirs(out_dir, exist_ok=True)
        video_output_path = get_unique_filename(os.path.join(out_dir, "run"), ".mp4")

    video_saved = {"done": False}
    _old_sigint_handler = signal.getsignal(signal.SIGINT)
    _old_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def _finalize_video(reason: str) -> None:
        if not enable_video_record: return
        if video_saved["done"]: return
        if video_output_path is None: return
        
        old = signal.getsignal(signal.SIGINT)
        try: signal.signal(signal.SIGINT, signal.SIG_IGN)
        except: old = None
        
        try:
            env.env_camera.capture = False
            if hasattr(env, "thread_record") and env.thread_record.is_alive():
                env.thread_record.join(timeout=10.0)
            env.env_camera.create_mp4(video_output_path)
            video_saved["done"] = True
            cprint(f"[VIDEO] finalized ({reason}) -> {video_output_path}", "green")
        except Exception as e:
            cprint(f"[WARNING] Save mp4 failed: {e}", "yellow")
        finally:
            if old is not None: signal.signal(signal.SIGINT, old)

    def _request_stop(signum, frame):
        try: env.env_camera.capture = False
        except: pass
        raise KeyboardInterrupt

    try: signal.signal(signal.SIGINT, _request_stop)
    except: pass
    try: signal.signal(signal.SIGTERM, _request_stop)
    except: pass
    atexit.register(lambda: _finalize_video("exit"))

    initial_pcd = None
    finished_by_vlm = False
    
    # ------------------------------- #
    #   任务循环 (VLM + MP)           #
    # ------------------------------- #
    max_subtasks = 6 
    try:
        for subtask_idx in range(max_subtasks):
            cprint(f"=========== Subtask {subtask_idx} : VLM & Motion Planning ===========", color="cyan")

            # 1. 观测
            set_prim_visible_group(["/World/DexLeft", "/World/DexRight"], False)
            for _ in range(50): env.step()
            
            rgb = env.garment_camera.get_rgb_graph(save_or_not=False)
            pcd, _ = env.garment_camera.get_point_cloud_data_from_segment(save_or_not=False, real_time_watch=False)
            
            if pcd is None or len(pcd) == 0:
                cprint("[WARNING] PCD empty, abort.", "red")
                break
            
            if initial_pcd is None: initial_pcd = pcd.copy()
            env.garment_pcd = pcd

            # 2. VLM 推理
            vlm_result = _ask_vlm_plan_and_points(rgb, vlm_client, vlm_model_name)
            plan = vlm_result.get("plan", None)
            points = vlm_result.get("points", None)

            if debug_flag:
                _debug_save_vlm_output(os.path.dirname(debug_rgb_dir), subtask_idx, vlm_result)
                _debug_save_vlm_rgb_with_bbox(rgb, points, os.path.join(debug_rgb_dir, f"vlm_rgb_{subtask_idx:03d}.png"))
                _debug_save_vlm_rgb_raw(rgb, os.path.join(debug_rgb_dir, f"vlm_rgb_raw_{subtask_idx:03d}.png"))

            set_prim_visible_group(["/World/DexLeft", "/World/DexRight"], True)
            for _ in range(50): env.step()

            # 3. 解析 Plan
            if isinstance(plan, str) and "finish" in plan.lower():
                cprint("VLM: Finished folding.", "green")
                finished_by_vlm = True
                break
            
            if not isinstance(plan, list) or len(plan) == 0:
                cprint(f"[WARNING] Invalid plan: {plan}", "yellow")
                break

            current_step = plan[0]

            # 4. 执行 Motion Planning (Actions + Stabilize + Reset combined)
            success = execute_vlm_motion_plan(env, rgb, pcd, current_step, points)
            
            if not success:
                cprint("Motion Planning execution failed or skipped step.", "red")
                # 可以选择 break 或 continue，这里继续尝试下一步或者等待

        # ----------------------------- #
        #   评估                        #
        # ----------------------------- #
        set_prim_visible_group(["/World/DexLeft", "/World/DexRight"], False)
        for _ in range(50): env.step()
        _finalize_video("normal")

        eval_pcd = initial_pcd if initial_pcd is not None else env.garment_pcd
        # 关键点采样，用于生成 boundary
        points, *_ = env.model.get_manipulation_points(eval_pcd, [554, 1540, 1014, 1385])
        boundary = [points[0][0] - 0.05, points[1][0] + 0.05, points[3][1] - 0.1, points[2][1] + 0.1]
        
        pcd_end, _ = env.garment_camera.get_point_cloud_data_from_segment(save_or_not=False, real_time_watch=False)
        success = judge_pcd(pcd_end, boundary, threshold=0.12)
        cprint(f"final result: {success}", color="green", on_color="on_green")

        if validation_flag:
            log_dir = "Data/Fold_Tops_Validation_HALO_MP"
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "validation_log.txt"), "a") as f:
                f.write(f"result:{success}  usd_path:{env.garment.usd_path}  pos_x:{pos[0]}  pos_y:{pos[1]}\n")
            pic_dir = os.path.join(log_dir, "final_state_pic")
            os.makedirs(pic_dir, exist_ok=True)
            env.env_camera.get_rgb_graph(save_or_not=True, save_path=get_unique_filename(os.path.join(pic_dir, "img"), ".png"))

    except KeyboardInterrupt:
        cprint("[INFO] KeyboardInterrupt, finalizing...", "yellow")
        _finalize_video("keyboard_interrupt")
        raise
    finally:
        try: signal.signal(signal.SIGINT, _old_sigint_handler)
        except: pass
        try: signal.signal(signal.SIGTERM, _old_sigterm_handler)
        except: pass


if __name__=="__main__":
    args = parse_args_val()
    
    pos = np.array([0.0, 0.8, 0.2])
    ori = np.array([0.0, 0.0, 0.0])
    usd_path = None
    
    if args.garment_random_flag:
        np.random.seed(int(time.time()))
        x = np.random.uniform(-0.1, 0.1)
        y = np.random.uniform(0.7, 0.9)
        pos = np.array([x,y,0.0])
        ori = np.array([0.0, 0.0, 0.0])
        usd_path = "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_model2_014/TCLC_model2_014_obj.usd"
    
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
        args.vlm_base_url,
        args.vlm_model_name,
        args.debug,
        args.debug_dir,
    )
    
    if args.validation_flag:
        simulation_app.close()
    else:
        while simulation_app.is_running():
            simulation_app.update()

simulation_app.close()

