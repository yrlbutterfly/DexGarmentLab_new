import os
import json
import argparse
import random
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_plan_points(item: Dict[str, Any]) -> Tuple[Any, List[Dict[str, Any]]]:
    """
    从一条 stage_*.json 记录中解析出 plan 和 points。
    """
    convs = item.get("conversations", [])
    gpt_msg = None
    for c in convs:
        if c.get("from") == "gpt":
            gpt_msg = c
            break

    if gpt_msg is None:
        return None, []

    value_str = gpt_msg.get("value", "")
    try:
        parsed = json.loads(value_str)
    except json.JSONDecodeError:
        return None, []

    if not isinstance(parsed, list) or len(parsed) != 2:
        return None, []

    plan = parsed[0].get("plan", None)
    points = parsed[1].get("points", [])
    return plan, points


def scale_bbox(
    bbox: List[float], img_w: int, img_h: int, base: float = 1000.0
) -> Tuple[int, int, int, int]:
    """
    把 0-1000 相对坐标的 bbox 按图像尺寸缩放到像素坐标。
    """
    if len(bbox) != 4:
        return 0, 0, 0, 0
    x1, y1, x2, y2 = bbox
    x1 = int(np.clip(x1 / base * img_w, 0, img_w - 1))
    x2 = int(np.clip(x2 / base * img_w, 0, img_w - 1))
    y1 = int(np.clip(y1 / base * img_h, 0, img_h - 1))
    y2 = int(np.clip(y2 / base * img_h, 0, img_h - 1))
    return x1, y1, x2, y2


def draw_points(
    img: np.ndarray,
    points: List[Dict[str, Any]],
    color_map: Optional[Dict[str, Tuple[int, int, int]]] = None,
) -> Dict[str, Tuple[int, int]]:
    """
    在图像上画出 points 的 bbox 和 label。
    返回：每个 label 对应 bbox 中心点坐标，用于画 plan 的连线。
    """
    h, w = img.shape[:2]
    centers: Dict[str, Tuple[int, int]] = {}

    if color_map is None:
        # BGR
        color_map = {
            "left_cuff": (0, 0, 255),
            "right_cuff": (0, 255, 0),
            "left_hem": (255, 0, 0),
            "right_hem": (255, 255, 0),
            "left_shoulder": (255, 0, 255),
            "right_shoulder": (0, 255, 255),
        }

    for p in points:
        label = p.get("label")
        bbox = p.get("bbox")
        if label is None or bbox is None:
            continue
        x1, y1, x2, y2 = scale_bbox(bbox, w, h)
        c = color_map.get(label, (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        cv2.putText(
            img,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            c,
            1,
            cv2.LINE_AA,
        )
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        centers[label] = (cx, cy)

    return centers


def draw_plan(
    img: np.ndarray,
    plan: Any,
    centers: Dict[str, Tuple[int, int]],
) -> None:
    """
    在图像上根据 plan 画出 from -> to 的连线。
    plan 可能是字符串 "already finish folding" 或 step 列表。
    """
    h, w = img.shape[:2]

    if isinstance(plan, str):
        text = f"plan: {plan}"
        cv2.putText(
            img,
            text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return

    if not isinstance(plan, list):
        return

    # 在左上角简单打印 step 数量
    summary = f"steps: {len(plan)}"
    cv2.putText(
        img,
        summary,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # 逐 step 画箭头
    for step_idx, step in enumerate(plan):
        if not isinstance(step, dict):
            continue
        for arm_name, color in [("left", (255, 255, 255)), ("right", (0, 165, 255))]:
            arm = step.get(arm_name)
            if not isinstance(arm, dict):
                continue
            from_label = arm.get("from")
            to_label = arm.get("to")
            if from_label is None or to_label is None:
                continue
            p1 = centers.get(from_label)
            p2 = centers.get(to_label)
            if p1 is None or p2 is None:
                continue

            # 在起点/终点画明显的圆点
            cv2.circle(img, p1, 4, color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(img, p2, 4, color, thickness=-1, lineType=cv2.LINE_AA)

            # 加粗箭头，并进一步增大箭头尖长度，让方向更明显
            cv2.arrowedLine(
                img,
                p1,
                p2,
                color,
                5,
                tipLength=0.12,
            )

            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2
            label = f"{arm_name[0].upper()}{step_idx + 1}"
            cv2.putText(
                img,
                label,
                (mid_x, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )


def visualize_stage_file(
    stage_name: str,
    json_path: str,
    out_dir: str,
    num_samples: int = 5,
    seed: int = 0,
) -> None:
    print(f"Processing {stage_name} from {json_path}")
    data = load_json(json_path)
    if not isinstance(data, list) or len(data) == 0:
        print(f"  Empty or invalid json: {json_path}")
        return

    random.seed(seed)
    indices = list(range(len(data)))
    if len(indices) > num_samples:
        indices = random.sample(indices, num_samples)

    for idx in indices:
        item = data[idx]
        img_rel = item.get("image")
        if not img_rel:
            print(f"  [skip] index {idx}: no image field")
            continue

        img_path = img_rel
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.getcwd(), img_path)

        if not os.path.exists(img_path):
            print(f"  [skip] index {idx}: image not found -> {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"  [skip] index {idx}: failed to read image -> {img_path}")
            continue

        plan, points = extract_plan_points(item)
        vis_img = img.copy()

        centers = draw_points(vis_img, points)
        draw_plan(vis_img, plan, centers)

        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_name = f"{stage_name}_{idx}_{base_name}.jpg"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, vis_img)
        print(f"  saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="随机抽取 stage_*.json 中若干条样本，在图片上可视化 plan 和 points（0-1000 相对坐标缩放）。"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="Preprocess/vis_stage_labels",
        help="可视化结果输出目录",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="每个 json 文件随机抽取的样本数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子，保证复现同样的样本选择",
    )
    args = parser.parse_args()

    root = os.getcwd()
    stage_files = [
        ("stage_0", os.path.join(root, "Preprocess/data/stage_0.json")),
        ("stage_1", os.path.join(root, "Preprocess/data/stage_1.json")),
        ("stage_2", os.path.join(root, "Preprocess/data/stage_2.json")),
        ("stage_3", os.path.join(root, "Preprocess/data/stage_3.json")),
    ]

    for stage_name, json_path in stage_files:
        if not os.path.exists(json_path):
            print(f"[warn] json not found for {stage_name}: {json_path}")
            continue
        visualize_stage_file(
            stage_name=stage_name,
            json_path=json_path,
            out_dir=args.out_dir,
            num_samples=args.samples,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()


