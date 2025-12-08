import argparse
import json
import os
from typing import Dict, Any

from PIL import Image


def convert_sample_to_relative(
    sample: Dict[str, Any],
    image_dir: str,
    scale: float = 1000.0,
    convert_points: bool = True,
) -> Dict[str, Any]:
    """
    Convert all bbox (and optionally point) coordinates in one jsonl sample
    from absolute pixel coordinates to relative [0, scale] coordinates.
    """
    rgb_name = sample.get("rgb")
    if not rgb_name:
        return sample

    image_path = os.path.join(image_dir, rgb_name)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found for sample: {image_path}")

    with Image.open(image_path) as img:
        width, height = img.size

    def to_rel_x(x: float) -> float:
        # 先映射到 [0, scale]，再四舍五入，保证结果是「好看」的整数
        return int(round(x / width * scale))

    def to_rel_y(y: float) -> float:
        return int(round(y / height * scale))

    for key, value in sample.items():
        # skip non-keypoint fields
        if key == "rgb" or not isinstance(value, dict):
            continue

        # convert point
        if convert_points and "point" in value:
            px, py = value["point"]
            value["point"] = [to_rel_x(px), to_rel_y(py)]

        # convert bbox: [x1, y1, x2, y2]
        if "bbox" in value:
            x1, y1, x2, y2 = value["bbox"]
            value["bbox"] = [
                to_rel_x(x1),
                to_rel_y(y1),
                to_rel_x(x2),
                to_rel_y(y2),
            ]

    return sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert garments_box.jsonl from absolute pixel coordinates "
            "to relative [0, 1000] coordinates for Qwen3-VL."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        default="./Preprocess/data/stage2_1206_color/garments_box.jsonl",
        help="Path to input garments_box.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default="./Preprocess/data/stage2_1206_color/garments_box_relative.jsonl",
        help="Path to output jsonl with relative coordinates",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="",
        help=(
            "Directory containing images. If not provided, will use "
            "<input_dir>/images by default."
        ),
    )
    parser.add_argument(
        "--no_convert_points",
        action="store_true",
        help="If set, only convert bbox and keep point as absolute.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1000.0,
        help="Target coordinate range upper bound (default: 1000.0).",
    )

    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    if args.image_dir:
        image_dir = os.path.abspath(args.image_dir)
    else:
        # default: sibling images directory next to jsonl file
        input_dir = os.path.dirname(input_path)
        image_dir = os.path.join(input_dir, "images")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input jsonl not found: {input_path}")

    if not os.path.isdir(image_dir):
        raise NotADirectoryError(f"Image directory not found: {image_dir}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    convert_points = not args.no_convert_points

    with open(input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            sample = convert_sample_to_relative(
                sample,
                image_dir=image_dir,
                scale=args.scale,
                convert_points=convert_points,
            )
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()


