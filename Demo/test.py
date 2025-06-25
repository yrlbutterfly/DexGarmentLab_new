import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def plot_points_on_image(image_path, points):
    # Read image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Draw points
    radius = 5
    for point in points:
        x, y = point
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill='red')
    
    # Display image
    plt.imshow(img)
    plt.axis('off')
    plt.show()


image_path = "/home/admin01/Projects/DexGarmentLab/Demo/fold.png"
points = np.array([[300, 320]])
plot_points_on_image(image_path, points)


# import cv2
# import matplotlib.pyplot as plt
# # 设置中文字体支持
# plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'AR PL UMing TW MBE']  # 使用系统中可用的中文字体
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# # 定义特征点坐标 (x, y)
# points = {
#     "领口中心顶点": (159, 45),
#     "左侧领口端点": (135, 60),
#     "右侧领口端点": (183, 60),
#     "左肩峰": (100, 75),
#     "右肩峰": (218, 75),
#     "左腋下点": (115, 125),
#     "右腋下点": (203, 125),
#     "左袖口最低点": (65, 110),
#     "右袖口最低点": (253, 110),
#     "左下摆端点": (125, 175),
#     "右下摆端点": (193, 175),
#     "中心下摆点": (159, 173)
# }

# # 读取图片
# img_path = '/home/admin01/Projects/DexGarmentLab/Demo/image.png'
# img = cv2.imread(img_path)
# # 转换 BGR 到 RGB
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # 绘制
# plt.figure(figsize=(6, 4))
# plt.imshow(img_rgb)
# for label, (x, y) in points.items():
#     plt.scatter(x, y, color='red', s=50)  # 增加点的大小和颜色
#     plt.text(x + 5, y + 5, label, fontsize=8, color='blue', weight='bold')  # 改进文本显示

# plt.axis('off')
# plt.tight_layout()  # 自动调整布局
# # plt.show()

# plt.savefig('test.png', dpi=300)
