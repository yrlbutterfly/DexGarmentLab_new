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
from Model_HALO.SADP_G.SADP_G import SADP_G


# ------------------------- #
#   VLM & 2D-3D 工具函数    #
# ------------------------- #

# 通过 VLM 生成折叠 plan + keypoint bbox 的提示词
VLM_FOLD_PROMPT = (
    "<image>\n"
    "The image contains a piece of clothing. Please infer how it should be folded.\n\n"
    "Your output must be a single JSON array containing two objects: one with a \"plan\" field and one with a \"points\" field.\n\n"
    "------------------------------------------------ OUTPUT FORMAT (STRICT) ------------------------------------------------\n\n"
    "Your final output must always be in the following structure:\n\n"
    "[\n\n"
    "  { \"plan\": <plan_output> },\n\n"
    "  { \"points\": <points_output> }\n\n"
    "]\n\n"
    "Where:\n\n"
    "- <plan_output> is either:\n\n"
    "  1) A list of folding steps, OR\n\n"
    "  2) The string \"already finish folding\" if no folding is needed.\n\n"
    "- <points_output> is a list of keypoint bounding box entries mentioned in the plan.\n\n"
    "------------------------------------------------ FOLDING PLAN RULES ------------------------------------------------\n\n"
    "If folding is required, the plan must be a list. The i-th element represents the i-th folding action:\n\n"
    "{\n\n"
    "  \"left\": {\"from\": <keypoint_name>, \"to\": <keypoint_name>},\n\n"
    "  \"right\": {\"from\": <keypoint_name>, \"to\": <keypoint_name>}\n\n"
    "}\n\n"
    "- If a robotic arm does not need to operate during a step, assign null to both \"from\" and \"to\" for that arm.\n\n"
    "- The folding plan must be practical for robotic manipulation:\n\n"
    "  actions should avoid arm collisions, use clear step-by-step motions rather than\n\n"
    "  merging multiple operations, and follow common-sense garment-folding practices.\n\n"
    "Valid keypoint names include:\n\n"
    "left_cuff, right_cuff\n\n"
    "left_collar, right_collar, center_collar\n\n"
    "left_hem, right_hem, center_hem\n\n"
    "left_armpit, right_armpit\n\n"
    "left_shoulder, right_shoulder\n\n"
    "left_waist, right_waist\n\n"
    "------------------------------------------------ STANDARD SHIRT FOLDING PROCEDURE ------------------------------------------------\n\n"
    "The complete folding plan contains three standard steps:\n\n"
    "Step 1 — Fold the left sleeve inward:\n\n"
    "    Move: left_cuff  -> right_shoulder\n\n"
    "    Right arm stays idle (null -> null)\n\n"
    "Step 2 — Fold the right sleeve inward:\n\n"
    "    Move: right_cuff -> left_shoulder\n\n"
    "    Left arm stays idle (null -> null)\n\n"
    "Step 3 — Fold the bottom hem upward:\n\n"
    "    Move left_hem  -> left_shoulder\n\n"
    "    Move right_hem -> right_shoulder\n\n"
    "    (Both arms operate simultaneously)\n\n"
    "------------------------------------------------ FOLDING COMPLETION CONDITION ------------------------------------------------\n\n"
    "Return \"already finish folding\" if the garment is already folded.\n\n"
    "A garment is considered \"already folded\" when:\n\n"
    "- Most garment pixels or keypoints fall inside a compact rectangular region,\n\n"
    "- Both sleeves have already been folded inward,\n\n"
    "- The hem is lifted so the shirt forms a clean rectangle.\n\n"
    "If these conditions are met, no folding plan is needed and the output should be:\n\n"
    "[\n\n"
    "  { \"plan\": \"already finish folding\" },\n\n"
    "  { \"points\": [] }\n\n"
    "]\n\n"
    "------------------------------------------------ KEYPOINT BOUNDING BOX RULES ------------------------------------------------\n\n"
    "Each keypoint entry must follow:\n\n"
    "{\n\n"
    "  \"label\": \"<keypoint_name>\",\n\n"
    "  \"bbox\": [x_min, y_min, x_max, y_max]\n\n"
    "}\n\n"
    "Where:\n\n"
    "- x_min, y_min = top-left corner of bounding box\n\n"
    "- x_max, y_max = bottom-right corner of bounding box\n\n"
    "------------------------------------------------ COMBINED EXAMPLE OUTPUT ------------------------------------------------\n\n"
    "Example folding plan(the numbers in bboxes are random):\n\n"
    "[\n\n"
    "  {\n\n"
    "    \"plan\": [\n\n"
    "      {\n\n"
    "        \"left\": {\"from\": \"left_cuff\", \"to\": \"right_shoulder\"},\n\n"
    "        \"right\": {\"from\": null, \"to\": null}\n\n"
    "      },\n\n"
    "      {\n\n"
    "        \"left\": {\"from\": null, \"to\": null},\n\n"
    "        \"right\": {\"from\": \"right_cuff\", \"to\": \"left_shoulder\"}\n\n"
    "      },\n\n"
    "      {\n\n"
    "        \"left\": {\"from\": \"left_hem\", \"to\": \"left_shoulder\"},\n\n"
    "        \"right\": {\"from\": \"right_hem\", \"to\": \"right_shoulder\"}\n\n"
    "      }\n\n"
    "    ]\n\n"
    "  },\n\n"
    "  {\n\n"
    "    \"points\": [\n\n"
    "      {\"label\": \"left_cuff\", \"bbox\": [90, 180, 140, 230]},\n\n"
    "      {\"label\": \"right_cuff\", \"bbox\": [290, 190, 340, 240]},\n\n"
    "      {\"label\": \"left_hem\", \"bbox\": [100, 250, 150, 300]},\n\n"
    "      {\"label\": \"right_hem\", \"bbox\": [250, 250, 300, 300]},\n\n"
    "      {\"label\": \"left_shoulder\", \"bbox\": [100, 100, 150, 150]},\n\n"
    "      {\"label\": \"right_shoulder\", \"bbox\": [250, 100, 300, 150]}\n\n"
    "    ]\n\n"
    "  }\n\n"
    "]\n\n"
    "Example when no folding is needed:\n\n"
    "[\n\n"
    "  { \"plan\": \"already finish folding\" },\n\n"
    "  { \"points\": [] }\n\n"
    "]\n\n"
    "------------------------------------------------\n\n"
    "Do not output anything except the final JSON array.\n"
)


def _get_vlm_client(base_url: str, model_name: str):
    """
    根据传入的 base_url / model_name 构造 OpenAI 兼容的 VLM client。

    仍然从环境变量中读取 VLM_API_KEY（若后端未做鉴权，可任意字符串，默认 "EMPTY"）。
    """
    api_key = os.environ.get("VLM_API_KEY", "EMPTY")
    client = OpenAI(base_url=base_url, api_key=api_key)
    return client, model_name


def _encode_rgb_to_data_url(rgb: np.ndarray) -> str:
    """
    将 RGB 图像编码为 base64 data URL，供 OpenAI 接口中的 image_url 使用。
    """
    # Isaac 返回的是 RGB，需要先转成 BGR 再编码
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
    期望格式（严格）：
        [
          { \"plan\": ... },
          { \"points\": [...] }
        ]
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
        # VLM 输出的 bbox 为 0~1000 的相对坐标，这里同样需要根据图像尺寸转换为像素坐标
        x_min_rel, y_min_rel, x_max_rel, y_max_rel = [float(v) for v in bbox]
        x_min = int(x_min_rel / 1000.0 * img_w)
        x_max = int(x_max_rel / 1000.0 * img_w)
        y_min = int(y_min_rel / 1000.0 * img_h)
        y_max = int(y_max_rel / 1000.0 * img_h)
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


def _debug_save_affordance_3d(
    pcd: np.ndarray,
    feat: np.ndarray,
    save_path: str,
) -> None:
    """
    在 3D 点云上可视化当前的 points_affordance_feature（4 个通道），
    参考 utils.ipynb 中的画法。
    """
    if pcd is None or feat is None:
        return
    if feat.shape[1] < 4:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(figsize=(12, 8))

    channels = [
        ("left affordance", 0),
        ("left goal", 2),
        ("right affordance", 1),
        ("right goal", 3),
    ]

    for j, (title, ch_idx) in enumerate(channels):
        ax = fig.add_subplot(2, 2, j + 1, projection="3d")
        score = feat[:, ch_idx]
        p = ax.scatter(
            pcd[:, 0],
            pcd[:, 1],
            pcd[:, 2],
            c=score,
            cmap="viridis",
            s=3,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)
        fig.colorbar(p, ax=ax, label=title)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def _debug_save_pcd_only(
    pcd: np.ndarray,
    save_path: str,
) -> None:
    """
    仅可视化当前的衣物点云（不带权重），用于直观查看 point cloud 的几何形状。
    """
    if pcd is None:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        pcd[:, 0],
        pcd[:, 1],
        pcd[:, 2],
        c="b",
        s=3,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("garment_point_cloud")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def get_rgb_index(env, rgb: np.ndarray, point: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
    """
    将 3D 点（世界坐标）投影到 garment_camera 图像平面，返回像素坐标 (u, v)：
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
    if point_clip[3] > 0:
        point_ndc = point_clip[:3] / point_clip[3]

        # 检查点是否在 NDC 范围内（-1~1），否则说明超出相机视锥
        if -1 <= point_ndc[0] <= 1 and -1 <= point_ndc[1] <= 1:
            pixel_x = int((point_ndc[0] + 1) * width / 2)
            pixel_y = int((1 - point_ndc[1]) * height / 2)  # y 轴在图像坐标系中是反的
            return pixel_x, pixel_y

    return None, None


def _project_pcd_to_pixels(
    env,
    rgb: np.ndarray,
    pcd: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将当前衣物点云中的每个 3D 点投影到 RGB 图像上，得到像素坐标 (u, v) 以及可见性 mask。
    返回：
        us: (N,) 所有点的像素 x 坐标（若不可见则为 0）
        vs: (N,) 所有点的像素 y 坐标（若不可见则为 0）
        mask: (N,) bool，True 表示该点在相机视野内
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


def _gaussian_field_for_bbox(
    us: np.ndarray,
    vs: np.ndarray,
    mask: np.ndarray,
    pcd: np.ndarray,
    bbox: List[float],
    sigma: float = 0.1,
    outside_decay: float = 0.7,
) -> np.ndarray:
    """
    以 bbox 对应区域在点云上的 3D 质心为高斯核中心，在 3D 空间中对所有点生成一维权重：
    - bbox 内部对应的 3D 点距离质心越近，权重越高；
    - bbox 外部对应的 3D 点在高斯衰减基础上再乘一个更小的系数（outside_decay）。
    """
    x_min, y_min, x_max, y_max = bbox
    # 先根据 2D bbox 选出在像素空间内的点，作为 3D 高斯中心与尺度的参考
    inside_2d = (
        (us >= x_min)
        & (us <= x_max)
        & (vs >= y_min)
        & (vs <= y_max)
        & mask
    )

    if not np.any(inside_2d):
        # 若没有任何点落在该 bbox 内，则认为本次 keypoint 无法在点云上可靠定位，返回全 0
        return np.zeros(pcd.shape[0], dtype=np.float32)

    # 以 bbox 内所有 3D 点的质心作为该 keypoint 在点云上的“语义中心”
    center_3d = pcd[inside_2d].mean(axis=0)

    # 使用与 Fold_Tops_Env 中 compute_similarity 完全一致的高斯相似度形式：
    # similarity = exp(- (dist^2) / (2 * sigma^2))
    sim_col = compute_similarity(pcd, center_3d, sigma=sigma)  # (N, 1)
    field = sim_col.reshape(-1).astype(np.float32)

    # 为了在 VLM 场景下仍然利用 bbox 约束范围，
    # 对于没有有效投影（mask=False）的点，做一次额外衰减；其余保持与原 compute_similarity 一致
    field[~mask] *= outside_decay

    return field


def build_points_affordance_feature_from_vlm(
    env,
    rgb: np.ndarray,
    pcd: np.ndarray,
    plan_step: Dict[str, object],
    points_list: Optional[List[Dict[str, object]]],
) -> np.ndarray:
    """
    根据 VLM 输出的某一个 plan step + 对应 keypoint bbox，为当前点云构造
    [左手 affordance, 右手 affordance, 左手 goal, 右手 goal] 共 4 通道的特征，
    返回形状为 (N_points, 4) 的列归一化矩阵。
    """
    # 1) 建立 label -> bbox 的映射
    # 注意：当前 VLM 输出的 bbox 为 0~1000 的相对坐标，需要先根据图像尺寸转换为像素坐标
    img_h, img_w = rgb.shape[:2]
    label_to_bbox: Dict[str, List[float]] = {}
    if isinstance(points_list, list):
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
            # VLM 返回的 bbox 坐标范围为 [0, 1000]，在水平方向和竖直方向上分别线性映射到 [0, img_w] / [0, img_h]
            # [x_min, y_min, x_max, y_max]（相对坐标） -> 像素坐标
            x_min_rel, y_min_rel, x_max_rel, y_max_rel = [float(b) for b in bbox]
            x_min = x_min_rel / 1000.0 * img_w
            x_max = x_max_rel / 1000.0 * img_w
            y_min = y_min_rel / 1000.0 * img_h
            y_max = y_max_rel / 1000.0 * img_h
            label_to_bbox[str(label)] = [x_min, y_min, x_max, y_max]

    # 2) 将点云投影到像素平面
    us, vs, mask = _project_pcd_to_pixels(env, rgb, pcd)

    def _field_for_label(label_name: Optional[str]) -> np.ndarray:
        if label_name is None:
            return np.zeros(pcd.shape[0], dtype=np.float32)
        bbox = label_to_bbox.get(str(label_name))
        if bbox is None:
            return np.zeros(pcd.shape[0], dtype=np.float32)
        # 在点云的 3D 空间上构造高斯权重，而不是仅在 2D 像素平面内
        return _gaussian_field_for_bbox(us, vs, mask, pcd, bbox)

    left_cfg = plan_step.get("left", {}) if isinstance(plan_step, dict) else {}
    right_cfg = plan_step.get("right", {}) if isinstance(plan_step, dict) else {}

    left_from = left_cfg.get("from")
    left_to = left_cfg.get("to")
    right_from = right_cfg.get("from")
    right_to = right_cfg.get("to")

    feat_left_aff = _field_for_label(left_from)
    feat_right_aff = _field_for_label(right_from)
    feat_left_goal = _field_for_label(left_to)
    feat_right_goal = _field_for_label(right_to)

    features = np.stack(
        [feat_left_aff, feat_right_aff, feat_left_goal, feat_right_goal],
        axis=1,
    )
    # 列归一化到 [0, 1]，与原始 points_affordance_feature 的数值范围保持一致
    return normalize_columns(features)

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
        #   加载统一策略模型 SADP_G     #
        # ----------------------------- #
        # 这里与当前提供的 checkpoint 目录保持一致：
        #   Model_HALO/SADP_G/checkpoints/Fold_Tops_stage_1_2_3_${training_data_num}/${stage_1_checkpoint_num}.ckpt
        # 如需使用真正的统一策略模型，可将 task_name 改回 "Fold_Tops_unified"
        self.sadp_g = SADP_G(
            task_name="Fold_Tops",
            data_num=training_data_num,
            checkpoint_num=stage_1_checkpoint_num,
        )
        
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
            ],
            # 开启相机参数输出，便于后续做 3D->2D 投影（get_camera_matrices）
            camera_params_enable=True,
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


def FoldTops(
    pos,
    ori,
    usd_path,
    ground_material_usd,
    validation_flag,
    record_video_flag,
    training_data_num,
    stage_1_checkpoint_num,
    stage_2_checkpoint_num,
    stage_3_checkpoint_num,
    vlm_base_url,
    vlm_model_name,
    debug_flag,
):
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
    env = FoldTops_Env(
        pos,
        ori,
        usd_path,
        ground_material_usd,
        record_video_flag,
        training_data_num,
        stage_1_checkpoint_num,
        stage_2_checkpoint_num,
        stage_3_checkpoint_num,
    )

    # 若需要录制视频，则启动采集线程（异步）
    if record_video_flag:
        env.thread_record.start()

    # 初始化 VLM client（单例），用于整个任务周期
    vlm_client, vlm_model_name = _get_vlm_client(vlm_base_url, vlm_model_name)

    # 若开启 debug 模式，则为本次任务创建独立的调试输出目录
    debug_rgb_dir = None
    debug_feat_dir = None
    if debug_flag:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_debug_dir = os.path.join("Data", "Fold_Tops_Validation_HALO", "debug", timestamp)
        debug_rgb_dir = os.path.join(base_debug_dir, "vlm_rgb")
        debug_feat_dir = os.path.join(base_debug_dir, "pcd_feature")

    # 记录最初一次的衣物点云，用于后续基于 GAM 的评估
    initial_pcd = None

    # ------------------------------- #
    #   VLM 驱动的子任务循环          #
    # ------------------------------- #
    max_subtasks = 6
    finished_by_vlm = False

    for subtask_idx in range(max_subtasks):
        cprint(
            f"=========== Subtask {subtask_idx} : VLM 规划 ===========",
            color="cyan",
        )

        # ------------------------------- #
        #   1. 获取当前衣物点云 & RGB     #
        # ------------------------------- #
        # 为了获得“只有衣物”的观测，这里仅在渲染上隐藏双臂，不影响物理仿真
        set_prim_visible_group(
            prim_path_list=["/World/DexLeft", "/World/DexRight"],
            visible=False,
        )
        for _ in range(50):
            env.step()

        rgb = env.garment_camera.get_rgb_graph(
            save_or_not=False,
            save_path=None,
        )

        pcd, color = env.garment_camera.get_point_cloud_data_from_segment(
            save_or_not=False,
            save_path=get_unique_filename("data", extension=".ply"),
            real_time_watch=False,
        )

        if pcd is None or len(pcd) == 0:
            cprint("[WARNING] 子任务开始时衣物点云为空，提前结束折叠流程。", "red")
            break

        if initial_pcd is None:
            initial_pcd = pcd.copy()

        # 在当前子任务内保持 garment_point_cloud 不变
        env.garment_pcd = pcd

        # 调用 VLM，得到当前整件衣物的折叠 plan 与 keypoint bboxes
        vlm_result = _ask_vlm_plan_and_points(rgb, vlm_client, vlm_model_name)
        plan = vlm_result.get("plan", None)
        points = vlm_result.get("points", None)

        # 若处于 debug 模式，则记录本轮 VLM 输出与可视化结果
        if debug_flag:
            _debug_save_vlm_output(
                debug_dir=os.path.dirname(debug_rgb_dir)
                if debug_rgb_dir is not None
                else os.path.join("Data", "Fold_Tops_Validation_HALO", "debug"),
                step_idx=subtask_idx,
                vlm_result=vlm_result,
            )
            if debug_rgb_dir is not None:
                rgb_path = os.path.join(
                    debug_rgb_dir,
                    f"vlm_rgb_{subtask_idx:03d}.png",
                )
                _debug_save_vlm_rgb_with_bbox(rgb, points, rgb_path)

        # 恢复双臂可见，后续执行具体动作
        set_prim_visible_group(
            prim_path_list=["/World/DexLeft", "/World/DexRight"],
            visible=True,
        )
        for _ in range(50):
            env.step()

        # ------------------------------- #
        #   2. 解析 plan，构造特征         #
        # ------------------------------- #
        if isinstance(plan, str):
            # VLM 认为已经完成折叠
            if plan.strip().lower() == "already finish folding":
                cprint("VLM 判断已完成折叠，停止子任务循环。", "green")
                finished_by_vlm = True
                break
            else:
                cprint(
                    f"[WARNING] VLM 返回的 plan 为字符串且非 'already finish folding'：{plan}",
                    "yellow",
                )
                break

        if not isinstance(plan, list) or len(plan) == 0:
            cprint(f"[WARNING] VLM 返回的 plan 为空或格式错误：{plan}", "yellow")
            break

        # 只取第一个 step 作为当前需要执行的高层动作
        current_step = plan[0]

        env.points_affordance_feature = build_points_affordance_feature_from_vlm(
            env, rgb, pcd, current_step, points
        )

        # 在构造好 points_affordance_feature 之后，若处于 debug 模式，则在 3D 上进行可视化
        if debug_flag and debug_feat_dir is not None:
            # 1) 保存仅包含衣物点云几何形状的可视化
            pcd_path = os.path.join(
                debug_feat_dir,
                f"garment_pcd_{subtask_idx:03d}.png",
            )
            _debug_save_pcd_only(pcd, pcd_path)

            # 2) 保存在点云上叠加 points_affordance_feature 的 3D 可视化
            feat_path = os.path.join(
                debug_feat_dir,
                f"pcd_feature_{subtask_idx:03d}.png",
            )
            _debug_save_affordance_3d(pcd, env.points_affordance_feature, feat_path)

        # ------------------------------- #
        #   3. 在当前子任务下滚动策略      #
        # ------------------------------- #
        # 为了与原脚本大致保持时间长度，前两个子任务执行 8 次 rollout，之后执行 12 次
        num_outer_iters = 8 if subtask_idx < 2 else 12

        for i in range(num_outer_iters):
            print(f"Subtask_{subtask_idx}_Step: {i}")

            joint_pos_L = env.bimanual_dex.dexleft.get_joint_positions()
            joint_pos_R = env.bimanual_dex.dexright.get_joint_positions()
            joint_state = np.concatenate([joint_pos_L, joint_pos_R])

            obs = dict()
            obs["agent_pos"] = joint_state
            obs["environment_point_cloud"] = env.env_camera.get_pointcloud_from_depth()
            obs["garment_point_cloud"] = env.garment_pcd
            obs["points_affordance_feature"] = env.points_affordance_feature

            # 统一策略模型输出 [4, 60] 的关节角序列
            action = env.sadp_g.get_action(obs)
            print("action_shape:", action.shape)

            for j in range(4):
                # 将每帧动作拆成左右臂 30 维关节角
                action_L = ArticulationAction(joint_positions=action[j][:30])
                action_R = ArticulationAction(joint_positions=action[j][30:])

                env.bimanual_dex.dexleft.apply_action(action_L)
                env.bimanual_dex.dexright.apply_action(action_R)

                for _ in range(5):
                    env.step()

                # 更新观测并回传给策略（例如维护内部轨迹状态）
                joint_pos_L = env.bimanual_dex.dexleft.get_joint_positions()
                joint_pos_R = env.bimanual_dex.dexright.get_joint_positions()
                joint_state = np.concatenate([joint_pos_L, joint_pos_R])

                obs = dict()
                obs["agent_pos"] = joint_state
                obs["environment_point_cloud"] = env.env_camera.get_pointcloud_from_depth()
                obs["garment_point_cloud"] = env.garment_pcd
                obs["points_affordance_feature"] = env.points_affordance_feature

                env.sadp_g.update_obs(obs)

        # ------------------------------- #
        #   4. 子任务结束后加速布料稳定    #
        # ------------------------------- #
        env.garment.particle_material.set_gravity_scale(10.0)
        settle_steps = 200 if subtask_idx < 2 else 100
        for _ in range(settle_steps):
            env.step()
        env.garment.particle_material.set_gravity_scale(1.0)

    # 子任务循环结束（可能由 max_subtasks 或 VLM 完成信号触发）
    
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
    # 使用最初一次的衣物点云（若存在）或当前点云，调用 GAM 采样四个关键点，
    # 用于构造评估区域 boundary：
    # boundary = [min_x, max_x, min_y, max_y]
    # 直观理解：用 4 个点确定一个“理想折叠区域”的矩形包围盒
    eval_pcd = initial_pcd if initial_pcd is not None else env.garment_pcd
    points, *_ = env.model.get_manipulation_points(eval_pcd, [554, 1540, 1014, 1385])
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
        #usd_path = np.random.choice(assets_list)
        usd_path = "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_model2_014/TCLC_model2_014_obj.usd"
    
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
        args.vlm_base_url,
        args.vlm_model_name,
        args.debug,
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

