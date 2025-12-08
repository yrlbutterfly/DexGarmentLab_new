import os
import json
import cv2


def visualize_garments_box(
    project_name: str = "neat_1125",
) -> None:
    """
    读取 Preprocess/data/{project_name}/garments_box.jsonl，
    按照 Deformable_Tops_Collect.py 里的 save_jsonl 可视化方式，
    在对应的 RGB 图像上画出所有部位的 point 和 bbox，并保存到 data_vis 目录。
    """

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(base_dir, "data", project_name)
    jsonl_path = os.path.join(data_root, "garments_box_relative.jsonl")
    image_dir = os.path.join(data_root, "images")
    vis_dir = os.path.join(data_root, "data_vis")

    os.makedirs(vis_dir, exist_ok=True)

    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"jsonl 文件不存在: {jsonl_path}")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")

    print(f"读取标注: {jsonl_path}")
    print(f"图片目录: {image_dir}")
    print(f"可视化输出目录: {vis_dir}")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[{idx}] 解析 JSON 失败: {e}")
                continue

            rgb_name = data.get("rgb")
            if not rgb_name:
                print(f"[{idx}] 缺少 'rgb' 字段，跳过。")
                continue

            img_path = os.path.join(image_dir, rgb_name)
            if not os.path.exists(img_path):
                print(f"[{idx}] 找不到图片: {img_path}，跳过。")
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"[{idx}] 无法读取图片: {img_path}，跳过。")
                continue

            # 直接在 BGR 图像上画，不再做颜色空间转换
            rgb_image = img.copy()

            # 基于相对坐标 [0, 1000] 还原到像素坐标
            h, w = rgb_image.shape[:2]
            scale = 1000.0

            # 模仿 Deformable_Tops_Collect.save_jsonl 里的可视化方式：
            # - 所有 key（除 rgb / pcd）都画 point 和 bbox
            for key, value in data.items():
                if key in ("rgb", "pcd"):
                    continue

                if not isinstance(value, dict):
                    continue

                point = value.get("point")
                bbox = value.get("bbox")

                # 画点（绿色）
                if (
                    point is not None
                    and isinstance(point, (list, tuple))
                    and len(point) == 2
                    and point[0] is not None
                    and point[1] is not None
                ):
                    # point 为相对坐标 [0, 1000]，转换为像素坐标
                    px = int(float(point[0]) / scale * w)
                    py = int(float(point[1]) / scale * h)
                    cv2.circle(
                        rgb_image,
                        (px, py),
                        radius=1,
                        color=(0, 255, 0),
                        thickness=-1,
                    )

                # 画 bbox（红框）并标注 key
                if (
                    bbox is not None
                    and isinstance(bbox, (list, tuple))
                    and len(bbox) == 4
                    and all(v is not None for v in bbox)
                ):
                    # bbox 为相对坐标 [0, 1000]，转换为像素坐标
                    bx1, by1, bx2, by2 = bbox
                    x1 = int(float(bx1) / scale * w)
                    y1 = int(float(by1) / scale * h)
                    x2 = int(float(bx2) / scale * w)
                    y2 = int(float(by2) / scale * h)
                    pt1 = (x1, y1)
                    pt2 = (x2, y2)
                    cv2.rectangle(
                        rgb_image,
                        pt1,
                        pt2,
                        color=(255, 0, 0),
                        thickness=1,
                    )

                    if (
                        point is not None
                        and isinstance(point, (list, tuple))
                        and len(point) == 2
                        and point[0] is not None
                        and point[1] is not None
                    ):
                        cv2.putText(
                            rgb_image,
                            key,
                            pt1,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3,
                            (255, 0, 0),
                            1,
                        )

            # 让可视化文件的编号和原图的编号一致：
            # image_151.png -> vis_151.png
            base_name, _ = os.path.splitext(rgb_name)
            vis_id = base_name
            if base_name.startswith("image_"):
                vis_id = base_name.split("image_")[-1]
            out_name = f"vis_{vis_id}.png"
            out_path = os.path.join(vis_dir, out_name)
            cv2.imwrite(out_path, rgb_image)
            print(f"[{idx}] 保存可视化: {out_path}")


if __name__ == "__main__":
    # 默认针对 neat_1125，如需其他项目可改成 argparse 参数
    visualize_garments_box("right_1127_rot")


