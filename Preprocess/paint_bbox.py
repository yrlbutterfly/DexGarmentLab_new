import cv2
import json
import numpy as np
import matplotlib.pyplot as plt


def paint_bbox(image_path, answer_path):
    answer = json.load(open(answer_path, "r"))

    image = cv2.imread(image_path)

    if image is None:
        print(f"cannot read image: {image_path}")
    else:
        # 遍历JSON中的每个服装部位
        for label, info in answer.items():
            # 跳过rgb字段
            if label == "rgb":
                continue
                
            # 获取边界框坐标
            bbox = info["bbox"]
            # bbox格式: [[x1, y1], [x2, y2]]
            x1, y1 = bbox[0]
            x2, y2 = bbox[1]
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 绘制标签
            cv2.putText(image, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 可选：绘制关键点
            # point = info["point"]
            # point_x, point_y = point
            # cv2.circle(image, (point_x, point_y), 3, (255, 0, 0), -1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
        plt.axis('off')
        plt.show()
        
if __name__ == "__main__":
    image_path = "/home/admin01/Projects/DexGarmentLab/Preprocess/data/image/image_31.png"
    answer_path = "/home/admin01/Projects/DexGarmentLab/Preprocess/data/answer.json"
    paint_bbox(image_path, answer_path)
