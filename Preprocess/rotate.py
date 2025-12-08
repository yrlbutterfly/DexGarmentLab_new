import os
import json
from typing import Dict, List, Tuple

import cv2
import numpy as np


def get_deterministic_angle(name: str, max_abs_deg: float = 15.0) -> float:
    """
    根据文件名生成一个可复现的随机角度，范围 [-max_abs_deg, max_abs_deg]。
    """
    # 使用简单的 hash 作为种子，保证每次运行对同一张图角度一致
    seed = hash(name) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-max_abs_deg, max_abs_deg)
    return float(angle)


def rotate_image(img: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    绕图像中心旋转，保持尺寸不变。
    返回：旋转后的图像、用于坐标变换的 2x3 仿射矩阵 M。
    """
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return rotated, M


def transform_point(pt: List[int], M: np.ndarray) -> List[int]:
    """对点应用 2x3 仿射矩阵 M。"""
    x, y = pt
    vec = np.array([x, y, 1.0], dtype=np.float32)
    x_new, y_new = M @ vec
    return [float(x_new), float(y_new)]


def clamp_point(pt: List[float], w: int, h: int) -> List[int]:
    x, y = pt
    x = max(0.0, min(float(w - 1), x))
    y = max(0.0, min(float(h - 1), y))
    return [int(round(x)), int(round(y))]


def transform_bbox(bbox: List[int], M: np.ndarray, w: int, h: int) -> List[int]:
    """
    旋转 bbox：先旋转四个角点，再取外接 axis-aligned bbox，并 clamp 到图像范围。
    """
    x_min, y_min, x_max, y_max = bbox
    corners = [
        [x_min, y_min],
        [x_min, y_max],
        [x_max, y_min],
        [x_max, y_max],
    ]
    transformed = [transform_point(c, M) for c in corners]

    xs = [p[0] for p in transformed]
    ys = [p[1] for p in transformed]

    x_min_new = min(xs)
    x_max_new = max(xs)
    y_min_new = min(ys)
    y_max_new = max(ys)

    x_min_i, y_min_i = clamp_point([x_min_new, y_min_new], w, h)
    x_max_i, y_max_i = clamp_point([x_max_new, y_max_new], w, h)

    # 确保有非空区域
    x_min_i = min(x_min_i, x_max_i)
    x_max_i = max(x_min_i + 1, x_max_i)
    y_min_i = min(y_min_i, y_max_i)
    y_max_i = max(y_min_i + 1, y_max_i)

    x_max_i = min(x_max_i, w - 1)
    y_max_i = min(y_max_i, h - 1)

    return [int(x_min_i), int(y_min_i), int(x_max_i), int(y_max_i)]


def transform_part_dict(
    d: Dict, M: np.ndarray, w: int, h: int
) -> Dict:
    """对一个部位字典（含 point/bbox）应用旋转变换。"""
    if d is None:
        return None
    out = dict(d)
    if "point" in out:
        pt_rot = transform_point(out["point"], M)
        out["point"] = clamp_point(pt_rot, w, h)
    if "bbox" in out:
        out["bbox"] = transform_bbox(out["bbox"], M, w, h)
    return out


def process_images(
    src_dir: str, dst_dir: str, subfolder: str, max_abs_deg: float = 15.0
) -> Dict[str, Tuple[np.ndarray, int, int]]:
    """
    旋转指定子目录下所有 png 图片，返回 {文件名: (M, w, h)}。
    """
    src_img_dir = os.path.join(src_dir, subfolder)
    dst_img_dir = os.path.join(dst_dir, subfolder)
    os.makedirs(dst_img_dir, exist_ok=True)

    img_names = sorted(
        [f for f in os.listdir(src_img_dir) if f.lower().endswith(".png")]
    )

    transform_cache: Dict[str, Tuple[np.ndarray, int, int]] = {}

    for name in img_names:
        src_path = os.path.join(src_img_dir, name)
        dst_path = os.path.join(dst_img_dir, name)

        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Failed to read image: {src_path}")
            continue

        h, w = img.shape[:2]
        angle = get_deterministic_angle(name, max_abs_deg=max_abs_deg)
        rotated, M = rotate_image(img, angle)
        cv2.imwrite(dst_path, rotated)

        transform_cache[name] = (M, w, h)

    return transform_cache


def process_jsonl(
    jsonl_in: str,
    jsonl_out: str,
    transform_cache: Dict[str, Tuple[np.ndarray, int, int]],
):
    """
    用和图片相同的旋转矩阵，对 garments_box.jsonl 中的点和框做旋转。
    """
    with open(jsonl_in, "r", encoding="utf-8") as f_in, open(
        jsonl_out, "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            img_name = data.get("rgb")
            if img_name is None or img_name not in transform_cache:
                # 没有对应图像/矩阵信息，直接原样写出
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            M, w, h = transform_cache[img_name]

            new_data: Dict = {"rgb": img_name}

            for key, value in data.items():
                if key == "rgb":
                    continue
                if not isinstance(value, dict):
                    new_data[key] = value
                    continue
                new_data[key] = transform_part_dict(value, M, w, h)

            f_out.write(json.dumps(new_data, ensure_ascii=False) + "\n")


def main():
    src_dir = "/home/psibot/DexGarmentLab/Preprocess/data/stage2_1206"
    dst_dir = "/home/psibot/DexGarmentLab/Preprocess/data/stage2_1206_rot"

    os.makedirs(dst_dir, exist_ok=True)

    print("Rotating images in 'images/' ...")
    transform_cache = process_images(src_dir, dst_dir, subfolder="images", max_abs_deg=15.0)

    # 可选：对 data_vis 也做旋转，仅用于可视化检查
    if os.path.isdir(os.path.join(src_dir, "data_vis")):
        print("Rotating images in 'data_vis/' ...")
        # 对可视化图也使用同样的角度（基于文件名），但不需要记录矩阵
        process_images(src_dir, dst_dir, subfolder="data_vis", max_abs_deg=15.0)

    jsonl_in = os.path.join(src_dir, "garments_box.jsonl")
    jsonl_out = os.path.join(dst_dir, "garments_box.jsonl")
    print("Rotating annotations in garments_box.jsonl ...")
    process_jsonl(jsonl_in, jsonl_out, transform_cache)

    print("Done. Rotated dataset saved to:", dst_dir)


if __name__ == "__main__":
    main()


