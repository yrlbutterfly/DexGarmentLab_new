import os
import shutil
from typing import Tuple

import cv2
import numpy as np


def get_rng_for_name(name: str) -> np.random.Generator:
    """根据文件名得到一个可复现的随机数生成器。"""
    seed = hash(name) & 0xFFFFFFFF
    return np.random.default_rng(seed)


def adjust_brightness_contrast(
    img: np.ndarray, brightness_factor: float, contrast_factor: float
) -> np.ndarray:
    """调整亮度和对比度，img 为 uint8 BGR 图像。"""
    img_f = img.astype(np.float32) / 255.0

    # 亮度：整体乘一个系数
    img_f = img_f * brightness_factor

    # 对比度：围绕全图均值缩放
    mean = img_f.mean(axis=(0, 1), keepdims=True)
    img_f = (img_f - mean) * contrast_factor + mean

    img_f = np.clip(img_f, 0.0, 1.0)
    return (img_f * 255.0).astype(np.uint8)


def adjust_saturation(img: np.ndarray, saturation_factor: float) -> np.ndarray:
    """调整饱和度，img 为 uint8 BGR 图像。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    s = s * saturation_factor
    s = np.clip(s, 0, 255)

    hsv = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def color_jitter(
    img: np.ndarray,
    name: str,
    brightness_range: Tuple[float, float] = (0.8, 1.2),
    contrast_range: Tuple[float, float] = (0.8, 1.2),
    saturation_range: Tuple[float, float] = (0.8, 1.2),
) -> np.ndarray:
    """对图像做轻微颜色扰动：亮度、对比度、饱和度。"""
    rng = get_rng_for_name(name)

    b = rng.uniform(*brightness_range)
    c = rng.uniform(*contrast_range)
    s = rng.uniform(*saturation_range)

    out = adjust_brightness_contrast(img, b, c)
    out = adjust_saturation(out, s)
    return out


def strong_color_shift(img: np.ndarray, name: str) -> np.ndarray:
    """
    更强的颜色变换：大范围改变整张图的色调/饱和度/明度，
    使衣物整体看起来像“换了一种颜色”，但结构不变。
    """
    rng = get_rng_for_name(name)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    # 大幅随机平移色相（0~179），保证明显变色
    dh = int(rng.integers(30, 180))  # 避免太小的偏移
    h = (h + dh) % 180

    # 饱和度和亮度只做小范围变化，主要依靠色相平移来改变颜色
    s_scale = rng.uniform(0.9, 1.1)
    v_scale = rng.uniform(0.9, 1.1)
    s = np.clip(s * s_scale, 0, 255)
    v = np.clip(v * v_scale, 0, 255)

    hsv = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def process_images(src_dir: str, dst_dir: str, subfolder: str, use_strong: bool = False):
    src_img_dir = os.path.join(src_dir, subfolder)
    if not os.path.isdir(src_img_dir):
        return

    dst_img_dir = os.path.join(dst_dir, subfolder)
    os.makedirs(dst_img_dir, exist_ok=True)

    img_names = sorted(
        [f for f in os.listdir(src_img_dir) if f.lower().endswith(".png")]
    )

    for name in img_names:
        src_path = os.path.join(src_img_dir, name)
        dst_path = os.path.join(dst_img_dir, name)

        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Failed to read image: {src_path}")
            continue

        if use_strong:
            aug = strong_color_shift(img, name)
        else:
            aug = color_jitter(img, name)
        cv2.imwrite(dst_path, aug)


def main():
    src_dir = "/home/psibot/DexGarmentLab/Preprocess/data/stage2_1206"
    # 使用更强的颜色扰动，输出到单独目录
    dst_dir = "/home/psibot/DexGarmentLab/Preprocess/data/stage2_1206_color"

    os.makedirs(dst_dir, exist_ok=True)

    print("Applying STRONG color shift to 'images/' ...")
    process_images(src_dir, dst_dir, subfolder="images", use_strong=True)

    # 可选：对 data_vis 也做颜色增强，方便可视化检查
    print("Applying STRONG color shift to 'data_vis/' ...")
    process_images(src_dir, dst_dir, subfolder="data_vis", use_strong=True)

    # 直接复制标注文件（不涉及坐标变换）
    src_jsonl = os.path.join(src_dir, "garments_box.jsonl")
    dst_jsonl = os.path.join(dst_dir, "garments_box.jsonl")
    if os.path.isfile(src_jsonl):
        shutil.copy2(src_jsonl, dst_jsonl)
        print("Copied garments_box.jsonl to strong color-augmented folder.")

    print("Done. STRONG color-augmented dataset saved to:", dst_dir)


if __name__ == "__main__":
    main()


