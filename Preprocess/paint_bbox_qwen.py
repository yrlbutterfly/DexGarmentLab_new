import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from process_bbox import smart_resize

def paint_bbox(image_path, answer_path, resize=True):
    answer = json.load(open(answer_path, "r"))

    image = cv2.imread(image_path)

    h, w = image.shape[:2]
    new_height, new_width = smart_resize(h, w)

    if image is None:
        print(f"cannot read image: {image_path}")
    else:
        for bbox_info in answer:
            bbox = bbox_info["bbox_2d"].copy()  # Create a copy
            label = bbox_info["label"]
            x1, y1, x2, y2 = bbox
            if resize:
                x1 = int(x1 * w / new_width)
                y1 = int(y1 * h / new_height)
                x2 = int(x2 * w / new_width)
                y2 = int(y2 * h / new_height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
        plt.axis('off')
        plt.show()
        
if __name__ == "__main__":
    image_path = "/home/admin01/Desktop/image_0.png"
    answer_path = "/home/admin01/Projects/DexGarmentLab/Preprocess/data/answer.json"
    paint_bbox(image_path, answer_path, resize=False)
