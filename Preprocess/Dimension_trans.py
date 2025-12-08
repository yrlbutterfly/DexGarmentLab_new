import os
import json
import base64
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import open3d as o3d
import cv2


# 注意：不要在这里直接导入 Env_Config.Room.Object_Tools，
# 否则在 SimulationApp 尚未初始化时会因为找不到 isaacsim.core 而报错。
# 仅导入会自动创建 SimulationApp 的环境封装。
from Deformable_Tops_Collect import FoldTops_Env, get_rgb_index


# ==============================
# 1. 文本提示 & vLLM 调用封装
# ==============================

question_grounding_point = (
    "Given an image of a potentially deformable garment, identify and localize the following key regions: "
    "left_cuff, right_cuff, left_collar, right_collar, center_collar, left_hem, right_hem, center_hem, "
    "left_armpit, right_armpit, left_shoulder, right_shoulder, left_waist, right_waist. "
    "For each region, provide a 2D point coordinate in the format [x, y] that represents the keypoint "
    "location of that region. Return the results as a JSON array where each entry contains a \"label\" and "
    "a \"point_2d\" field. Example format: [{\"label\": \"left_cuff\", \"point_2d\": [x, y]}, "
    "{\"label\": \"right_cuff\", \"point_2d\": [x, y]}]. "
    "Region definitions: left_collar: left collar tip; right_collar: right collar tip; center_collar: lowest "
    "point of the V-neck or collar center; left_cuff: center of left sleeve opening; right_cuff: center of "
    "right sleeve opening; left_hem: bottom-left corner of the hem; right_hem: bottom-right corner of the hem; "
    "center_hem: midpoint of the bottom hem; left_armpit: under left armpit area; right_armpit: under right "
    "armpit area; left_shoulder: left shoulder point where sleeve attaches; right_shoulder: right shoulder "
    "point; left_waist: left waist point; right_waist: right waist point. Ensure all regions are included in the output."
)


def ask_image_question(
    image_path: str,
    question: str,
    model_name: str,
    client,
) -> str:
    """
    使用本地 vLLM (Qwen3-VL 等) 对指定图片进行多模态问答。

    :param image_path: 本地图片路径
    :param question: 提示词（例如 question_grounding_point）
    :param model_name: vLLM 暴露的模型名称
    :param client: 已初始化好的 OpenAI 兼容 client（由调用方传入）
    """
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    image_data_url = f"data:image/jpeg;base64,{img_base64}"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                ],
            }
        ],
        max_tokens=1024,
    )

    return response.choices[0].message.content


def parse_grounding_output(raw_text: str) -> List[Dict]:
    """
    从模型输出中解析出 JSON 数组:
    期望格式: [{"label": "...", "point_2d": [x, y]}, ...]
    """
    # 直接尝试整体解析
    try:
        return json.loads(raw_text)
    except Exception:
        pass

    # 如果模型外面包了说明文字，尝试用正则抽出第一个 JSON 数组
    match = re.search(r"\[.*\]", raw_text, re.S)
    if not match:
        raise ValueError(f"无法在模型输出中找到 JSON 数组，原始输出:\n{raw_text}")

    json_str = match.group(0)
    return json.loads(json_str)


# ==============================
# 2. 2D -> 3D 最近邻映射
# ==============================

def find_nearest_3d_point(
    u_ann: float,
    v_ann: float,
    garment_vertices: np.ndarray,
    pixel_coords: List[Tuple[Optional[float], Optional[float]]],
    max_pix_dist: float = 10.0,
) -> Optional[np.ndarray]:
    """
    在所有顶点的投影中，找到距离标注像素 (u_ann, v_ann) 最近的 3D 顶点。

    参数说明：
    - u_ann: 标注点在 RGB 图像上的 x 像素坐标（横轴），通常来自模型输出的 point_2d[0]
    - v_ann: 标注点在 RGB 图像上的 y 像素坐标（纵轴），通常来自模型输出的 point_2d[1]
    - garment_vertices: 形状为 (N, 3) 的数组，当前衣物网格的所有 3D 顶点（世界坐标）
    - pixel_coords: 长度为 N 的列表，第 i 个元素是顶点 i 在图像上的投影像素坐标 (u_i, v_i)，
      若该点不在视野内则为 (None, None)
    - max_pix_dist: 允许的最大像素距离阈值；若标注点与最近顶点投影的像素距离大于该值，则认为匹配失败

    返回：
    - 若找到合适的顶点，返回对应的 3D 坐标 (np.ndarray, 形状 (3,))
    - 若未找到或最近距离超过阈值，返回 None
    """
    best_idx = None
    best_dist2 = float("inf")

    for i, (u_i, v_i) in enumerate(pixel_coords):
        if u_i is None or v_i is None:
            continue
        du = u_i - u_ann
        dv = v_i - v_ann
        d2 = du * du + dv * dv
        if d2 < best_dist2:
            best_dist2 = d2
            best_idx = i

    if best_idx is None:
        return None

    if (best_dist2 ** 0.5) > max_pix_dist:
        return None

    return garment_vertices[best_idx]


# ==============================
# 3. 主 Pipeline
# ==============================

def run_grounding_pipeline(
    client,
    model_name: str,
    garment_usd_rel: str = "Assets/Garment/Tops/Collar_Lsleeve_FrontClose/TCLC_model2_014/TCLC_model2_014_obj.usd",
):
    """
    完整流程:
    1) 初始化仿真环境并加载指定衣物
    2) 获取同一时刻的 RGB 图像和衣物网格点云
    3) 调用 vLLM 多模态模型，得到各语义区域的 2D 像素坐标
    4) 将 2D 坐标映射回衣物点云上的 3D 顶点
    5) 使用 Open3D 在 3D 上可视化这些关键点
    """
    # ---------- 1. 初始化环境 ----------
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    env = FoldTops_Env()

    # 随便给一个初始位姿 (可以按需改)
    pos = np.array([0.0, 0.8, 0.2])
    ori = np.array([0.0, 0.0, 0.0])

    # 地面材质: 如无特定需求，可以置为 None 使用默认
    ground_material_usd = None

    garment_usd_abs = os.path.join(base_dir, garment_usd_rel)

    env.apply(
        pos=pos,
        ori=ori,
        ground_material_usd=ground_material_usd,
        usd_path=garment_usd_abs,
    )

    # 在采集图像前，把左右机械臂在渲染中隐藏掉，只保留衣物和地面
    try:
        # 这里再局部导入一次，保证此时 SimulationApp 已在 Deformable_Tops_Collect 中初始化完成
        from Env_Config.Room.Object_Tools import set_prim_visible_group

        set_prim_visible_group(
            prim_path_list=["/World/DexLeft", "/World/DexRight"],
            visible=False,
        )
    except Exception as e:
        print(f"[Warning] 隐藏机械臂失败，但不影响后续流程: {e}")

    # 让衣物稳定若干步
    for _ in range(50):
        env.step()

    # ---------- 2. 获取同一状态下的 RGB 和 3D 顶点 ----------
    # 中间没有再调用任何 env.step() 或改变布料的动作
    rgb = env.garment_camera.get_rgb_graph(
        save_or_not=False,
        save_path=None,
    )
    height, width, _ = rgb.shape

    # 保存一份 RGB 图像到硬盘，供 vLLM 读取
    output_dir = os.path.join(base_dir, "Preprocess/data/2dto3d_vis")
    os.makedirs(output_dir, exist_ok=True)
    rgb_path = os.path.join(output_dir, "grounding_input.png")
    cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # 获取衣物网格当前帧 3D 顶点 (与 Deformable_Tops_Collect 中保持一致)
    scale = np.array([0.0085, 0.0085, 0.0085])
    garment_vertices = env.garment.get_vertice_positions()
    garment_vertices = garment_vertices * scale
    garment_vertices += env.garment.get_garment_center_pos()  # (N, 3)

    # 对所有顶点做 3D -> 2D 投影 (使用已有的 get_rgb_index)
    pixel_coords: List[Tuple[Optional[float], Optional[float]]] = []
    for v in garment_vertices:
        u_pix, v_pix = get_rgb_index(env, rgb, v)
        pixel_coords.append((u_pix, v_pix))

    # ---------- 3. 调用 vLLM 模型做关键点 grounding ----------
    raw_answer = ask_image_question(
        image_path=rgb_path,
        question=question_grounding_point,
        model_name=model_name,
        client=client,
    )
    grounding_list = parse_grounding_output(raw_answer)

    # grounding_list: [{"label": "...", "point_2d": [x, y]}, ...]
    # 现在模型输出的是 0-1000 范围的相对坐标，这里先还原为像素坐标
    label_to_2d: Dict[str, Tuple[float, float]] = {}
    for item in grounding_list:
        label = item.get("label")
        pt = item.get("point_2d")
        if (
            label is None
            or pt is None
            or not isinstance(pt, (list, tuple))
            or len(pt) != 2
        ):
            continue
        # pt[0], pt[1] ∈ [0, 1000]，对应图像宽、高方向的归一化坐标
        x_norm, y_norm = float(pt[0]), float(pt[1])
        x_norm = float(np.clip(x_norm, 0.0, 1000.0))
        y_norm = float(np.clip(y_norm, 0.0, 1000.0))
        x_pix = x_norm / 1000.0 * width
        y_pix = y_norm / 1000.0 * height
        label_to_2d[label] = (x_pix, y_pix)

    # ---------- 4. 2D -> 3D 映射 ----------
    keypoint_3d_list = []
    label_to_3d: Dict[str, Optional[np.ndarray]] = {}

    for label, (u_ann, v_ann) in label_to_2d.items():
        p3d = find_nearest_3d_point(
            u_ann=u_ann,
            v_ann=v_ann,
            garment_vertices=garment_vertices,
            pixel_coords=pixel_coords,
            max_pix_dist=15.0,
        )
        label_to_3d[label] = p3d
        if p3d is not None:
            keypoint_3d_list.append(p3d)
            print(f"{label}: 2D ({u_ann:.1f}, {v_ann:.1f}) -> 3D {p3d}")
        else:
            print(f"{label}: 未找到足够接近的 3D 顶点")

    # ---------- 5. 在 3D 上可视化 ----------
    # 使用 Open3D 构建衣物点云：灰色显示整件衣物的所有网格顶点
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(garment_vertices)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])

    geometries = [pcd]

    # 如果存在有效的 3D 关键点，则用红色小球高亮显示（比点云更明显）
    if len(keypoint_3d_list) > 0:
        for p in keypoint_3d_list:
            # 半径可以根据衣物整体尺度适当调节
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 纯红色
            sphere.translate(p)
            geometries.append(sphere)

    # 弹出 Open3D 可视化窗口，展示衣物点云及预测关键点
    o3d.visualization.draw_geometries(geometries)

    # 同时返回 RGB 图片路径、2D 预测结果与 3D 映射结果，便于后续使用或保存
    return {
        "rgb_path": rgb_path,
        "label_to_2d": label_to_2d,
        "label_to_3d": label_to_3d,
    }


__all__ = [
    "question_grounding_point",
    "ask_image_question",
    "parse_grounding_output",
    "find_nearest_3d_point",
    "run_grounding_pipeline",
]

if __name__ == "__main__":
    # 这里示例用 OpenAI 兼容的 vLLM 客户端，你按自己的地址和模型名改
    from openai import OpenAI

    # vLLM 一般是 OpenAI 兼容接口，示例：
    client = OpenAI(
        base_url="http://127.0.0.1:8001/v1",  # 按你启动 vLLM 的地址改
        api_key="EMPTY",                      # 如果没鉴权，随便填一个即可
    )

    model_name_ft = "/share_data/yanruilin/qwen3vl_full_sft_cloth_point/checkpoint-500"       # 比如 "Qwen/Qwen2.5-VL-3B-Instruct"

    # 直接跑完整 pipeline：初始化环境 -> RGB+点云 -> 调模型 -> 2D->3D -> 可视化
    result = run_grounding_pipeline(
        client=client,
        model_name=model_name_ft,
    )

    # 可选：打印一下关键结果
    print("RGB image saved at:", result["rgb_path"])
    print("2D points:", result["label_to_2d"])
    print("3D points:", result["label_to_3d"])
