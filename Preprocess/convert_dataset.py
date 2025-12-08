import json
import os
import cv2


def merge_garments_box_relative(root_dir: str, merged_filename: str = "garments_box_relative.jsonl"):
    """
    将 root_dir 下各子文件夹里的 garments_box_relative.jsonl 合并到 root_dir 根目录，
    同时修正 rgb 字段，使其变为相对于 root_dir 的路径：
        子文件夹名/images/image_xxx.png
    """
    subdirs = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    subdirs.sort()

    output_path = os.path.join(root_dir, merged_filename)
    os.makedirs(root_dir, exist_ok=True)

    total_count = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for subdir in subdirs:
            jsonl_path = os.path.join(root_dir, subdir, "garments_box_relative.jsonl")
            if not os.path.exists(jsonl_path):
                # 某些子文件夹可能还没标完，直接跳过
                print(f"[merge] 跳过 {jsonl_path}（文件不存在）")
                continue

            with open(jsonl_path, "r", encoding="utf-8") as in_f:
                for line_num, line in enumerate(in_f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"[merge] 解析失败 {jsonl_path} 第 {line_num} 行: {e}")
                        continue

                    rgb = data.get("rgb")
                    if rgb is None:
                        print(f"[merge] 警告: {jsonl_path} 第 {line_num} 行没有 rgb 字段, 已跳过")
                        continue

                    # 如果原来是绝对路径，就不强行改动
                    if os.path.isabs(rgb):
                        new_rgb = rgb
                    else:
                        # 原始是 "image_0.png" 这种，这里统一改成 "子文件夹/images/image_0.png"
                        # 如果已经带有 images/ 前缀，就只加子文件夹名
                        if "/" in rgb or os.path.sep in rgb:
                            new_rgb = os.path.join(subdir, rgb)
                        else:
                            new_rgb = os.path.join(subdir, "images", rgb)

                        # 统一用 '/' 作为分隔符，方便后续模型/脚本使用
                        new_rgb = new_rgb.replace(os.sep, "/")

                    data["rgb"] = new_rgb

                    out_f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    total_count += 1

    print(f"[merge] 已在 {root_dir} 合并生成 {output_path}，共 {total_count} 条样本")
    return output_path, total_count


def convert_tile_jsonl_to_conversations(input_file: str, output_file: str):
    """
    将 jsonl 文件转换为对话格式的数据集

    参数:
        input_file: 合并后的 garments_box_relative.jsonl 路径
        output_file: 输出的对话数据集 json 路径
    """
    conversations_data = []

    # 读取JSONL文件
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # 解析JSON行
                data = json.loads(line.strip())

                # 提取图片路径（相对于 input_file 所在目录，或绝对路径）
                image_filename = data['rgb']

                # 如果是绝对路径，直接使用；否则认为是相对于 input_file 所在目录
                if os.path.isabs(image_filename):
                    image_path = image_filename
                else:
                    input_dir = os.path.dirname(input_file)
                    image_path = os.path.join(input_dir, image_filename)

                answer = []
                for key, value in data.items():
                    if key != "rgb" and key != "pcd":
                        # 对于 Qwen3-VL，我们的数据已经是 [0, 1000] 的相对坐标，
                        # 可以直接作为 bbox_2d 使用，无需再根据图像尺寸做缩放。
                        bbox = value["bbox"]
                        answer.append({
                            "bbox_2d": bbox,
                            "label": key
                        })

                # 构造对话内容
                conversation_item = {
                    "image": image_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\nGiven an image of a potentially deformable garment, identify and localize the following key regions: left_cuff, right_cuff, left_collar, right_collar, center_collar, left_hem, right_hem, center_hem, left_armpit, right_armpit, left_shoulder, right_shoulder, left_waist, right_waist. For each region, provide a tight 2D bounding box in the format: [x_min, y_min, x_max, y_max]. Return the results as a JSON array where each entry contains a \"label\" and a \"bbox_2d\" field. Example format: [{\"label\": \"left_cuff\", \"bbox_2d\": [x1, y1, x2, y2]}, {\"label\": \"right_cuff\", \"bbox_2d\": [x1, y1, x2, y2]}]. Region definitions: left_collar: left collar tip; right_collar: right collar tip; center_collar: lowest point of the V-neck or collar center; left_cuff: center of left sleeve opening; right_cuff: center of right sleeve opening; left_hem: bottom-left corner of the hem; right_hem: bottom-right corner of the hem; center_hem: midpoint of the bottom hem; left_armpit: under left armpit area; right_armpit: under right armpit area; left_shoulder: left shoulder point where sleeve attaches; right_shoulder: right shoulder point; left_waist: left waist point; right_waist: right waist point. Ensure all regions are included in the output."
                        },
                        {
                            "from": "gpt",
                            "value": json.dumps(answer, ensure_ascii=False)
                        }
                    ]
                }

                conversations_data.append(conversation_item)

            except json.JSONDecodeError as e:
                print(f"[convert] Error parsing line {line_num}: {e}")
                continue

    # 保存转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversations_data, f, indent=2, ensure_ascii=False)

    print(f"[convert] 转换完成！")
    print(f"[convert] 输入文件: {input_file}")
    print(f"[convert] 输出文件: {output_file}")
    print(f"[convert] 总共转换了 {len(conversations_data)} 条数据")

    # 打印前几条数据作为示例
    print("\n[convert] 前3条转换后的数据示例:")
    for i, item in enumerate(conversations_data[:3]):
        print(f"\n=== 第 {i+1} 条 ===")
        print(json.dumps(item, indent=2, ensure_ascii=False))


def main():
    """
    入口：
    1. 分别在 cloth_origin / cloth_augment 下合并各子文件夹的 garments_box_relative.jsonl
    2. 调用转换函数，生成 origin.json / augment.json
    """
    base_data_dir = os.path.join("Preprocess", "data")

    cloth_origin_dir = os.path.join(base_data_dir, "cloth_origin")
    cloth_augment_dir = os.path.join(base_data_dir, "cloth_augment")

    # 1) 合并 jsonl，并修正 rgb 路径
    origin_jsonl, origin_count = merge_garments_box_relative(cloth_origin_dir)
    augment_jsonl, augment_count = merge_garments_box_relative(cloth_augment_dir)

    print(f"[main] cloth_origin 合并样本数: {origin_count}")
    print(f"[main] cloth_augment 合并样本数: {augment_count}")

    # 2) 转换为最终对话数据集
    origin_output = os.path.join(base_data_dir, "origin.json")
    augment_output = os.path.join(base_data_dir, "augment.json")

    convert_tile_jsonl_to_conversations(origin_jsonl, origin_output)
    convert_tile_jsonl_to_conversations(augment_jsonl, augment_output)


if __name__ == "__main__":
    main()
