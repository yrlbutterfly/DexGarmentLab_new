import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from process_bbox import convert_to_qwen25vl_format

def convert_tile_jsonl_to_conversations():
    """
    将 tile.jsonl 文件转换为对话格式的数据集
    """
    input_file = "fold_once.jsonl"
    output_file = "converted_dataset_fold_once.json"
    
    conversations_data = []
    
    # 读取JSONL文件
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # 解析JSON行
                data = json.loads(line.strip())
                
                # 提取图片文件名
                image_filename = data['rgb']
                
                # 构造图片路径（根据目标格式）
                image_path = f"/home/admin01/Projects/DexGarmentLab/Preprocess/data/image_fold_once/{image_filename}"
                
                image = cv2.imread(image_path)
                h, w = image.shape[:2]

                
                for key, value in data.items():
                    if key != "rgb" and key != "pcd":
                        point = value["point"]
                        bbox = value["bbox"]
                        x1, y1, x2, y2 = bbox
                        bbox = convert_to_qwen25vl_format([x1, y1, x2, y2], h, w)
                        answer = []
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
                                    "value": f"<image>\nThis is a garment that may deform. Please locate {key} in this image and output the bbox coordinates in JSON format."
                                },
                                {
                                    "from": "gpt",
                                    "value": json.dumps(answer, ensure_ascii=False)
                                }
                            ]
                        }
                
                        conversations_data.append(conversation_item)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    # 保存转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversations_data, f, indent=2, ensure_ascii=False)
    
    print(f"转换完成！")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"总共转换了 {len(conversations_data)} 条数据")
    
    # 打印前几条数据作为示例
    print("\n前3条转换后的数据示例:")
    for i, item in enumerate(conversations_data[:3]):
        print(f"\n=== 第 {i+1} 条 ===")
        print(json.dumps(item, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    convert_tile_jsonl_to_conversations()
