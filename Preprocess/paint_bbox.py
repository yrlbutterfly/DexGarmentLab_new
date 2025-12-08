import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt


def _draw_bboxes(image, answer):
    """
    在给定的 image 上，根据 answer 字典里的 bbox 信息画框。
    支持两种 bbox 格式：
    1) [x1, y1, x2, y2]
    2) [[x1, y1], [x2, y2]]
    """
    # 遍历 JSON 中的每个服装部位
    for label, info in answer.items():
        # 跳过 rgb 字段
        if label == "rgb":
            continue

        bbox = info["bbox"]

        # 兼容两种 bbox 格式
        if isinstance(bbox, (list, tuple)) and bbox and isinstance(bbox[0], (list, tuple)):
            # [[x1, y1], [x2, y2]]
            (x1, y1), (x2, y2) = bbox
        else:
            # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 绘制标签
        cv2.putText(
            image,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )


def paint_bbox_from_dict(image_path, answer_dict, save_path=None):
    """直接使用已经解析好的字典画 bbox。

    参数:
        image_path: 原始图片路径
        answer_dict: 一张图片的标注字典（含 rgb、各部位 bbox 等）
        save_path: 若不为 None，则把画好 bbox 的图像保存到该路径
    """
    image = cv2.imread(image_path)

    if image is None:
        print(f"cannot read image: {image_path}")
        return

    _draw_bboxes(image, answer_dict)

    # 如有需要，先以 BGR 格式保存
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def paint_bbox(image_path, answer_path):
    """
    兼容原来的接口：从一个 json 文件中读取单张图片的标注再画 bbox。
    只用于非 jsonl 的情况。
    """
    with open(answer_path, "r") as f:
        answer = json.load(f)

    paint_bbox_from_dict(image_path, answer)


def load_annotation_from_jsonl(jsonl_path, target_rgb_name):
    """
    从 jsonl 文件中按 `rgb` 字段（例如 'image_60.png'）查找对应的标注。
    找到则返回该行解析出来的 dict，找不到返回 None。
    """
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("rgb") == target_rgb_name:
                return data
    return None


if __name__ == "__main__":
    # 想要可视化的图片名
    image_name = "image_62.png"

    # 路径请根据自己的数据位置调整
    image_dir = "/home/admin01/Projects/DexGarmentLab/Preprocess/data/left_0808/images_left_0808"
    jsonl_path = "/home/admin01/Projects/DexGarmentLab/Preprocess/data/left_0808/garments_box_0808.jsonl"

    image_path = os.path.join(image_dir, image_name)
    # 保存可视化结果的路径（这里默认保存在同一目录下，文件名加上 _bbox 后缀）
    save_path = os.path.join(
        image_dir, os.path.splitext(image_name)[0] + "_bbox.png"
    )

    ann = load_annotation_from_jsonl(jsonl_path, image_name)

    if ann is None:
        print(f"在 jsonl 中没有找到 rgb == {image_name} 的标注")
    else:
        print(f"保存可视化结果到: {save_path}")
        paint_bbox_from_dict(image_path, ann, save_path=save_path)
