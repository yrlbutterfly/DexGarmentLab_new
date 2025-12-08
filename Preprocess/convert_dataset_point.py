import json
import os
import cv2

from convert_dataset import merge_garments_box_relative

# Qwen3-VL 默认使用 [0, 1000] 的相对坐标，不再需要 Qwen2.5-VL 的坐标缩放


def convert_tile_jsonl_to_conversations_point(input_file: str, output_file: str):
    """
    将 jsonl 文件转换为使用 point 标注的对话格式数据集

    参数:
        input_file: 合并后的 garments_box_relative.jsonl 路径
        output_file: 输出的对话数据集 json 路径
    """
    conversations_data = []

    # 读取 JSONL 文件
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                # 解析 JSON 行
                data = json.loads(line.strip())

                # 提取图片路径（相对于 data 目录，或绝对路径）
                image_filename = data["rgb"]

                # 如果是绝对路径，直接使用；否则认为是相对于 input_file 所在目录（data）
                if os.path.isabs(image_filename):
                    image_path = image_filename
                else:
                    input_dir = os.path.dirname(input_file)
                    image_path = os.path.join(input_dir, image_filename)

                answer = []
                for key, value in data.items():
                    if key != "rgb" and key != "pcd":
                        # 对于 Qwen3-VL，我们的数据已经是 [0, 1000] 的相对坐标，
                        # 可以直接作为 point_2d 使用，无需再根据图像尺寸做缩放。
                        point = value["point"]
                        answer.append(
                            {
                                "point_2d": point,
                                "label": key,
                            }
                        )

                # 构造对话内容
                conversation_item = {
                    "image": image_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": (
                                "<image>\n"
                                "Given an image of a potentially deformable garment, "
                                "identify and localize the following key regions: "
                                "left_cuff, right_cuff, left_collar, right_collar, center_collar, "
                                "left_hem, right_hem, center_hem, left_armpit, right_armpit, "
                                "left_shoulder, right_shoulder, left_waist, right_waist. "
                                "For each region, provide a 2D point coordinate in the format "
                                "[x, y] that represents the keypoint location of that region. "
                                "Return the results as a JSON array where each entry contains a "
                                "\"label\" and a \"point_2d\" field. "
                                "Example format: "
                                "[{\"label\": \"left_cuff\", \"point_2d\": [x, y]}, "
                                "{\"label\": \"right_cuff\", \"point_2d\": [x, y]}]. "
                                "Region definitions: left_collar: left collar tip; "
                                "right_collar: right collar tip; center_collar: lowest point of "
                                "the V-neck or collar center; left_cuff: center of left sleeve "
                                "opening; right_cuff: center of right sleeve opening; "
                                "left_hem: bottom-left corner of the hem; right_hem: "
                                "bottom-right corner of the hem; center_hem: midpoint of the "
                                "bottom hem; left_armpit: under left armpit area; right_armpit: "
                                "under right armpit area; left_shoulder: left shoulder point "
                                "where sleeve attaches; right_shoulder: right shoulder point; "
                                "left_waist: left waist point; right_waist: right waist point. "
                                "Ensure all regions are included in the output."
                            ),
                        },
                        {
                            "from": "gpt",
                            "value": json.dumps(answer, ensure_ascii=False),
                        },
                    ],
                }

                conversations_data.append(conversation_item)

            except json.JSONDecodeError as e:
                print(f"[point] Error parsing line {line_num}: {e}")
                continue

    # 保存转换后的数据
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(conversations_data, f, indent=2, ensure_ascii=False)

    print("[point] 转换完成！")
    print(f"[point] 输入文件: {input_file}")
    print(f"[point] 输出文件: {output_file}")
    print(f"[point] 总共转换了 {len(conversations_data)} 条数据")

    # 打印前几条数据作为示例
    print("\n[point] 前3条转换后的数据示例:")
    for i, item in enumerate(conversations_data[:3]):
        print(f"\n=== 第 {i + 1} 条 ===")
        print(json.dumps(item, indent=2, ensure_ascii=False))


def main():
    """
    入口：
    1. 复用 convert_dataset.py 里的 merge_garments_box_relative，
       分别在 cloth_origin / cloth_augment 下合并各子文件夹的 garments_box_relative.jsonl
    2. 调用本文件的转换函数，生成 origin_point.json / augment_point.json
    """
    base_data_dir = os.path.join("Preprocess", "data")

    cloth_origin_dir = os.path.join(base_data_dir, "cloth_origin")
    cloth_augment_dir = os.path.join(base_data_dir, "cloth_augment")

    # 1) 合并 jsonl，并修正 rgb 路径（与 bbox 版保持一致）
    origin_jsonl, origin_count = merge_garments_box_relative(cloth_origin_dir)
    augment_jsonl, augment_count = merge_garments_box_relative(cloth_augment_dir)

    print(f"[point-main] cloth_origin 合并样本数: {origin_count}")
    print(f"[point-main] cloth_augment 合并样本数: {augment_count}")

    # 2) 转换为最终对话数据集（使用 point_2d）
    origin_output = os.path.join(base_data_dir, "origin_point.json")
    augment_output = os.path.join(base_data_dir, "augment_point.json")

    convert_tile_jsonl_to_conversations_point(origin_jsonl, origin_output)
    convert_tile_jsonl_to_conversations_point(augment_jsonl, augment_output)


if __name__ == "__main__":
    main()


