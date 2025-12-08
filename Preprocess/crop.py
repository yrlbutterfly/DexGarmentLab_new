import os
import json
from typing import List, Tuple

import cv2


def center_crop(img, crop_ratio: float = 0.5) -> Tuple:
    """
    Center crop an image by a given ratio.

    Returns:
        cropped_img, (x_offset, y_offset, crop_w, crop_h)
    """
    h, w = img.shape[:2]
    crop_w = int(w * crop_ratio)
    crop_h = int(h * crop_ratio)

    # ensure at least 1 pixel
    crop_w = max(1, crop_w)
    crop_h = max(1, crop_h)

    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2

    cropped = img[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return cropped, (x0, y0, crop_w, crop_h)


def _clamp(v: int, low: int, high: int) -> int:
    return max(low, min(high, v))


def transform_point(
    pt: List[int], x0: int, y0: int, crop_w: int, crop_h: int
) -> List[int]:
    """Shift point by crop offset and clamp into new image."""
    x, y = pt
    x_new = x - x0
    y_new = y - y0
    x_new = _clamp(x_new, 0, crop_w - 1)
    y_new = _clamp(y_new, 0, crop_h - 1)
    return [int(x_new), int(y_new)]


def transform_bbox(
    bbox: List[int], x0: int, y0: int, crop_w: int, crop_h: int
) -> List[int]:
    """Shift bbox by crop offset and clamp into new image."""
    x_min, y_min, x_max, y_max = bbox
    x_min_new, y_min_new = transform_point([x_min, y_min], x0, y0, crop_w, crop_h)
    x_max_new, y_max_new = transform_point([x_max, y_max], x0, y0, crop_w, crop_h)

    # ensure valid box ordering
    x_min_final = min(x_min_new, x_max_new)
    x_max_final = max(x_min_new, x_max_new)
    y_min_final = min(y_min_new, y_max_new)
    y_max_final = max(y_min_new, y_max_new)

    return [x_min_final, y_min_final, x_max_final, y_max_final]


def process_images(src_dir: str, dst_dir: str, subfolder: str, crop_ratio: float = 0.5):
    src_img_dir = os.path.join(src_dir, subfolder)
    dst_img_dir = os.path.join(dst_dir, subfolder)
    os.makedirs(dst_img_dir, exist_ok=True)

    img_names = sorted(
        [f for f in os.listdir(src_img_dir) if f.lower().endswith(".png")]
    )

    cache = {}

    for name in img_names:
        src_path = os.path.join(src_img_dir, name)
        dst_path = os.path.join(dst_img_dir, name)

        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Failed to read image: {src_path}")
            continue

        cropped, (x0, y0, cw, ch) = center_crop(img, crop_ratio=crop_ratio)
        cv2.imwrite(dst_path, cropped)

        # cache crop parameters for reuse when transforming jsonl
        cache[name] = (x0, y0, cw, ch)

    return cache


def process_jsonl(
    jsonl_path: str,
    out_jsonl_path: str,
    crop_params: dict,
    image_subfolder: str = "images",
):
    """
    Apply the same crop transform to all points and bboxes in garments_box.jsonl.
    """
    with open(jsonl_path, "r", encoding="utf-8") as f_in, open(
        out_jsonl_path, "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            img_name = data.get("rgb")
            if img_name is None:
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            if img_name not in crop_params:
                # this image might have failed to read; keep original coords
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            x0, y0, cw, ch = crop_params[img_name]

            # transform all keypoints/bboxes in this record
            for key, value in data.items():
                if key == "rgb":
                    continue
                if not isinstance(value, dict):
                    continue

                if "point" in value:
                    value["point"] = transform_point(
                        value["point"], x0, y0, cw, ch
                    )
                if "bbox" in value:
                    value["bbox"] = transform_bbox(value["bbox"], x0, y0, cw, ch)

            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")


def main():
    # Source and target directories
    src_dir = "/home/psibot/DexGarmentLab/Preprocess/data/stage2_1206"
    dst_dir = "/home/psibot/DexGarmentLab/Preprocess/data/stage2_1206_crop"

    os.makedirs(dst_dir, exist_ok=True)

    # 1) process main images (these are what garments_box.jsonl refers to)
    print("Cropping images in 'images/' ...")
    crop_params = process_images(src_dir, dst_dir, subfolder="images", crop_ratio=0.5)

    # 2) (optional) also crop visualization images with the same rule
    #    This does NOT change annotations, just keeps vis consistent.
    if os.path.isdir(os.path.join(src_dir, "data_vis")):
        print("Cropping images in 'data_vis/' ...")
        process_images(src_dir, dst_dir, subfolder="data_vis", crop_ratio=0.5)

    # 3) transform jsonl annotations
    jsonl_in = os.path.join(src_dir, "garments_box.jsonl")
    jsonl_out = os.path.join(dst_dir, "garments_box.jsonl")
    print("Transforming garments_box.jsonl ...")
    process_jsonl(jsonl_in, jsonl_out, crop_params)

    print("Done. Cropped data saved to:", dst_dir)


if __name__ == "__main__":
    main()


