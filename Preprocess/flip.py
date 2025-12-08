import os
import json
from typing import Dict, List, Tuple

import cv2


def flip_image_horiz(img):
    """水平翻转图像。"""
    return cv2.flip(img, 1)


def transform_point_horiz(pt: List[int], width: int) -> List[int]:
    """在给定宽度下对点做水平翻转（x -> w-1-x）。"""
    x, y = pt
    x_new = width - 1 - x
    return [int(x_new), int(y)]


def transform_bbox_horiz(bbox: List[int], width: int) -> List[int]:
    """在给定宽度下对 bbox 做水平翻转。"""
    x_min, y_min, x_max, y_max = bbox
    # 先翻两端，再重新排序
    x_min_new = width - 1 - x_max
    x_max_new = width - 1 - x_min
    if x_min_new > x_max_new:
        x_min_new, x_max_new = x_max_new, x_min_new
    return [int(x_min_new), int(y_min), int(x_max_new), int(y_max)]


def transform_part_dict(d: Dict, width: int) -> Dict:
    """对一个部位的字典（包含 point / bbox）做水平翻转。"""
    if d is None:
        return None
    d_new = dict(d)
    if "point" in d_new:
        d_new["point"] = transform_point_horiz(d_new["point"], width)
    if "bbox" in d_new:
        d_new["bbox"] = transform_bbox_horiz(d_new["bbox"], width)
    return d_new


LEFT_RIGHT_PAIRS = [
    ("left_cuff", "right_cuff"),
    ("left_collar", "right_collar"),
    ("left_hem", "right_hem"),
    ("left_armpit", "right_armpit"),
    ("left_shoulder", "right_shoulder"),
    ("left_waist", "right_waist"),
]


def process_images(src_dir: str, dst_dir: str, subfolder: str) -> Dict[str, int]:
    """
    对指定子目录下的所有 png 图片做水平翻转，保存到目标目录。
    返回: {文件名: 图像宽度}
    """
    src_img_dir = os.path.join(src_dir, subfolder)
    dst_img_dir = os.path.join(dst_dir, subfolder)
    os.makedirs(dst_img_dir, exist_ok=True)

    img_names = sorted(
        [f for f in os.listdir(src_img_dir) if f.lower().endswith(".png")]
    )

    width_cache: Dict[str, int] = {}

    for name in img_names:
        src_path = os.path.join(src_img_dir, name)
        dst_path = os.path.join(dst_img_dir, name)

        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Failed to read image: {src_path}")
            continue

        h, w = img.shape[:2]
        flipped = flip_image_horiz(img)
        cv2.imwrite(dst_path, flipped)

        width_cache[name] = w

    return width_cache


def process_jsonl(
    jsonl_in: str,
    jsonl_out: str,
    width_cache: Dict[str, int],
):
    """
    对 garments_box.jsonl 做水平翻转 + 左右关键点互换。
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
            if img_name is None or img_name not in width_cache:
                # 如果没有对应宽度信息，就原样写出
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            width = width_cache[img_name]

            # 先对所有部位做几何上的水平翻转
            transformed_parts: Dict[str, Dict] = {}
            for key, value in data.items():
                if key == "rgb":
                    continue
                if not isinstance(value, dict):
                    continue
                transformed_parts[key] = transform_part_dict(value, width)

            # 再做左右语义互换
            new_data: Dict = {"rgb": img_name}

            # 已处理的 key 集合，便于后面补充其他 key（如 center_*）
            handled_keys = set()

            for left_key, right_key in LEFT_RIGHT_PAIRS:
                left_part = transformed_parts.get(left_key)
                right_part = transformed_parts.get(right_key)

                if right_part is not None:
                    new_data[left_key] = right_part
                if left_part is not None:
                    new_data[right_key] = left_part

                handled_keys.add(left_key)
                handled_keys.add(right_key)

            # 其他未在左右对列表中的键（例如 center_collar, center_hem），保持名字不变
            for key, part in transformed_parts.items():
                if key in handled_keys:
                    continue
                if part is not None:
                    new_data[key] = part

            f_out.write(json.dumps(new_data, ensure_ascii=False) + "\n")


def main():
    src_dir = "/home/psibot/DexGarmentLab/Preprocess/data/stage2_1206"
    dst_dir = "/home/psibot/DexGarmentLab/Preprocess/data/stage2_1206_flip"

    os.makedirs(dst_dir, exist_ok=True)

    print("Flipping images in 'images/' ...")
    width_cache = process_images(src_dir, dst_dir, subfolder="images")

    # 可选：对 data_vis 也做同样的翻转（只改变图片外观，不影响标注）
    if os.path.isdir(os.path.join(src_dir, "data_vis")):
        print("Flipping images in 'data_vis/' ...")
        process_images(src_dir, dst_dir, subfolder="data_vis")

    jsonl_in = os.path.join(src_dir, "garments_box.jsonl")
    jsonl_out = os.path.join(dst_dir, "garments_box.jsonl")
    print("Flipping and swapping annotations in garments_box.jsonl ...")
    process_jsonl(jsonl_in, jsonl_out, width_cache)

    print("Done. Flipped dataset saved to:", dst_dir)


if __name__ == "__main__":
    main()


