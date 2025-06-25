import numpy as np
import open3d as o3d
import os
import sys
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox

class InteractivePointCloud:
    def __init__(self, ply_file_path, threshold=0.05):
        """
        初始化交互式点云可视化器
        
        Args:
            ply_file_path (str): PLY文件路径
            threshold (float): 距离阈值（米），默认5cm
        """
        self.ply_file_path = ply_file_path
        self.threshold = threshold
        self.point_cloud = None
        self.original_colors = None
        self.vis = None
        self.points = None
        self.colors = None
        
    def load_pointcloud(self):
        """加载PLY点云文件"""
        if not os.path.exists(self.ply_file_path):
            print(f"错误：文件 {self.ply_file_path} 不存在")
            return False
            
        try:
            # 读取PLY文件
            self.point_cloud = o3d.io.read_point_cloud(self.ply_file_path)
            
            # 获取点云数据
            self.points = np.asarray(self.point_cloud.points)
            self.colors = np.asarray(self.point_cloud.colors)
            
            # 保存原始颜色
            self.original_colors = self.colors.copy()
            
            print(f"成功加载点云文件：{self.ply_file_path}")
            print(f"点云包含 {len(self.points)} 个点")
            print(f"点云范围：")
            print(f"  X: [{self.points[:, 0].min():.3f}, {self.points[:, 0].max():.3f}]")
            print(f"  Y: [{self.points[:, 1].min():.3f}, {self.points[:, 1].max():.3f}]")
            print(f"  Z: [{self.points[:, 2].min():.3f}, {self.points[:, 2].max():.3f}]")
            
            return True
            
        except Exception as e:
            print(f"加载点云文件时出错：{e}")
            return False
    
    def reset_colors(self):
        """重置点云颜色为原始颜色"""
        if self.original_colors is not None:
            self.colors = self.original_colors.copy()
            self.point_cloud.colors = o3d.utility.Vector3dVector(self.colors)
    
    def highlight_nearby_points(self, selected_point_idx):
        """
        高亮显示距离选中点小于阈值的点
        
        Args:
            selected_point_idx (int): 选中点的索引
        """
        if selected_point_idx >= len(self.points):
            print(f"错误：点索引 {selected_point_idx} 超出范围")
            return
            
        # 重置颜色
        self.reset_colors()
        
        # 获取选中点的坐标
        selected_point = self.points[selected_point_idx]
        
        # 计算所有点到选中点的距离
        distances = np.linalg.norm(self.points - selected_point, axis=1)
        
        # 找到距离小于阈值的点
        nearby_mask = distances < self.threshold
        
        # 将近距离点标记为红色
        self.colors[nearby_mask] = [1.0, 0.0, 0.0]  # 红色
        
        # 更新点云颜色
        self.point_cloud.colors = o3d.utility.Vector3dVector(self.colors)
        
        # 打印信息
        nearby_count = np.sum(nearby_mask)
        print(f"选中点 {selected_point_idx} (坐标: {selected_point})")
        print(f"距离阈值: {self.threshold:.3f} 米")
        print(f"高亮显示 {nearby_count} 个点")
    
    def visualize(self):
        """可视化点云"""
        if self.point_cloud is None:
            print("错误：请先加载点云文件")
            return
            
        print("=" * 50)
        print("交互式点云可视化")
        print("=" * 50)
        print("操作说明：")
        print("- 鼠标左键点击：选择点并高亮显示周围区域")
        print("- 鼠标滚轮：缩放")
        print("- 鼠标中键拖拽：旋转视角")
        print("- 鼠标右键拖拽：平移视角")
        print("- 按 'R' 键：重置颜色")
        print("- 按 'Q' 键：退出")
        print("=" * 50)
        
        # 创建可视化窗口
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window("交互式点云可视化", width=1200, height=800)
        
        # 添加点云
        vis.add_geometry(self.point_cloud)
        
        # 设置渲染选项
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.1, 0.1, 0.1])  # 深灰色背景
        opt.point_size = 2.0  # 点的大小
        
        # 设置视角
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        
        # 实时交互循环
        while True:
            # 更新可视化
            if not vis.poll_events():
                break
            vis.update_renderer()
            
            # 检查是否有新的选中点
            picked_points = vis.get_picked_points()
            if picked_points:
                for point_idx in picked_points:
                    if point_idx < len(self.points):
                        self.highlight_nearby_points(point_idx)
                        print(f"选中了点 {point_idx}")
                        # 更新几何体
                        vis.update_geometry(self.point_cloud)
                        vis.poll_events()
                        vis.update_renderer()
        
        vis.destroy_window()

class PointCloud2DInteraction:
    def __init__(self, point_cloud, threshold=0.05):
        """
        2D图像交互式点云选择器
        
        Args:
            point_cloud: Open3D点云对象
            threshold (float): 距离阈值（米），默认5cm
        """
        self.point_cloud = point_cloud
        self.threshold = threshold
        self.points = np.asarray(point_cloud.points)
        self.colors = np.asarray(point_cloud.colors)
        self.original_colors = self.colors.copy()
        
        # 2D投影参数
        self.projection_2d = None
        self.point_indices_2d = None
        self.camera_params = None
        
    def project_to_2d(self, width=800, height=600):
        """
        将3D点云投影到2D图像
        
        Args:
            width (int): 2D图像宽度
            height (int): 2D图像高度
            
        Returns:
            tuple: (2D投影图像, 2D点坐标, 对应的3D点索引)
        """
        # 计算点云的边界框
        min_bound = self.points.min(axis=0)
        max_bound = self.points.max(axis=0)
        center = (min_bound + max_bound) / 2
        extent = max_bound - min_bound
        
        # 设置虚拟相机参数（俯视角度）
        camera_distance = np.linalg.norm(extent) * 1.0
        camera_pos = center + np.array([0, 0, camera_distance])
        
        # 创建虚拟相机矩阵 - 使用更合适的焦距
        max_extent = max(extent[0], extent[1])
        fx = fy = min(width, height) * 0.8  # 调整焦距使点云更好地填充图像
        cx, cy = width / 2, height / 2
        
        # 投影矩阵
        K = np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])
        
        # 视图矩阵（从相机位置看向中心）
        # 使用更准确的旋转矩阵，确保正确的俯视角度
        R = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])  # 俯视角度
        t = -R @ camera_pos
        
        # 投影3D点到2D
        points_homo = np.hstack([self.points, np.ones((len(self.points), 1))])
        points_cam = (R @ points_homo[:, :3].T + t.reshape(3, 1)).T
        
        # 透视投影
        points_2d_homo = (K @ points_cam.T).T
        points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
        
        # 修复Y轴镜像问题：OpenCV坐标系Y轴向下，需要翻转Y坐标
        points_2d[:, 1] = height - points_2d[:, 1]
        
        # 顺时针旋转180度
        # 旋转中心为图像中心
        center_x, center_y = width / 2, height / 2
        # 180度旋转矩阵: [cos(180) -sin(180); sin(180) cos(180)] = [-1 0; 0 -1]
        points_2d_centered = points_2d - np.array([center_x, center_y])
        points_2d_rotated = points_2d_centered @ np.array([[-1, 0], [0, -1]])
        points_2d = points_2d_rotated + np.array([center_x, center_y])
        
        # 过滤在图像范围内的点
        valid_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) & \
                    (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height)
        
        valid_points_2d = points_2d[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        # 创建2D图像
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # 绘制点
        for i, (point_2d, idx_3d) in enumerate(zip(valid_points_2d, valid_indices)):
            x, y = int(point_2d[0]), int(point_2d[1])
            if 0 <= x < width and 0 <= y < height:
                # 根据3D点的颜色绘制
                color_3d = self.colors[idx_3d]
                color_2d = (int(color_3d[2] * 255), int(color_3d[1] * 255), int(color_3d[0] * 255))
                cv2.circle(image, (x, y), 2, color_2d, -1)
        
        self.projection_2d = image
        self.point_indices_2d = valid_indices
        self.camera_params = (K, R, t, width, height)
        
        return image, valid_points_2d, valid_indices
    
    def click_to_3d(self, click_x, click_y):
        """
        将2D点击位置映射到3D点云
        
        Args:
            click_x (int): 2D点击的x坐标
            click_y (int): 2D点击的y坐标
            
        Returns:
            int: 最接近的3D点索引，如果没有找到返回-1
        """
        if self.projection_2d is None or self.point_indices_2d is None:
            return -1
        
        # 找到距离点击位置最近的2D点
        K, R, t, width, height = self.camera_params
        
        # 重新计算2D投影（简化版本）
        points_homo = np.hstack([self.points, np.ones((len(self.points), 1))])
        points_cam = (R @ points_homo[:, :3].T + t.reshape(3, 1)).T
        points_2d_homo = (K @ points_cam.T).T
        points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
        
        # 修复Y轴镜像问题：OpenCV坐标系Y轴向下，需要翻转Y坐标
        points_2d[:, 1] = height - points_2d[:, 1]
        
        # 顺时针旋转180度
        # 旋转中心为图像中心
        center_x, center_y = width / 2, height / 2
        # 180度旋转矩阵: [cos(180) -sin(180); sin(180) cos(180)] = [-1 0; 0 -1]
        points_2d_centered = points_2d - np.array([center_x, center_y])
        points_2d_rotated = points_2d_centered @ np.array([[-1, 0], [0, -1]])
        points_2d = points_2d_rotated + np.array([center_x, center_y])
        
        # 计算点击位置到所有2D点的距离
        click_pos = np.array([click_x, click_y])
        distances = np.linalg.norm(points_2d - click_pos, axis=1)
        
        # 找到最近的点
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        
        # 如果距离太远，认为没有选中
        if min_distance > 20:  # 20像素阈值
            return -1
        
        return min_idx
    
    def highlight_nearby_points(self, selected_point_idx):
        """
        高亮显示距离选中点小于阈值的点
        
        Args:
            selected_point_idx (int): 选中点的索引
        """
        if selected_point_idx >= len(self.points):
            print(f"错误：点索引 {selected_point_idx} 超出范围")
            return
            
        # 重置颜色
        self.colors = self.original_colors.copy()
        
        # 获取选中点的坐标
        selected_point = self.points[selected_point_idx]
        
        # 计算所有点到选中点的距离
        distances = np.linalg.norm(self.points - selected_point, axis=1)
        
        # 找到距离小于阈值的点
        nearby_mask = distances < self.threshold
        
        # 将近距离点标记为红色
        self.colors[nearby_mask] = [1.0, 0.0, 0.0]  # 红色
        
        # 更新点云颜色
        self.point_cloud.colors = o3d.utility.Vector3dVector(self.colors)
        
        # 打印信息
        nearby_count = np.sum(nearby_mask)
        print(f"选中点 {selected_point_idx} (坐标: {selected_point})")
        print(f"距离阈值: {self.threshold:.3f} 米")
        print(f"高亮显示 {nearby_count} 个点")
    
    def reset_colors(self):
        """重置点云颜色为原始颜色"""
        self.colors = self.original_colors.copy()
        self.point_cloud.colors = o3d.utility.Vector3dVector(self.colors)
    
    def create_2d_interface(self):
        """
        创建2D交互界面
        
        Returns:
            int: 选中的3D点索引
        """
        # 创建2D投影
        image, points_2d, indices_2d = self.project_to_2d()
        
        # 创建Tkinter窗口
        root = tk.Tk()
        root.title("2D点云选择器")
        
        # 创建matplotlib图形
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title("Click to select point, Enter to confirm, R to reset, Q to quit")
        ax.axis('off')
        
        selected_point_idx = [-1]  # 用列表存储，以便在回调中修改
        
        def on_click(event):
            if event.inaxes == ax:
                click_x, click_y = int(event.xdata), int(event.ydata)
                point_idx = self.click_to_3d(click_x, click_y)
                if point_idx >= 0:
                    selected_point_idx[0] = point_idx
                    self.highlight_nearby_points(point_idx)
                    print(f"选中3D点: {point_idx}")
                    print(f"2D点击坐标: ({click_x}, {click_y})")
                    print(f"对应3D坐标: {self.points[point_idx]}")
                    # 在2D图像上标记选中的点
                    ax.clear()
                    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    ax.plot(click_x, click_y, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
                    ax.axis('off')
                    canvas.draw()
                else:
                    print("未选中有效点")
        
        def on_key(event):
            if event.key == 'enter':
                if selected_point_idx[0] >= 0:
                    root.quit()
                else:
                    messagebox.showwarning("警告", "请先选择一个点")
            elif event.key == 'r':
                selected_point_idx[0] = -1
                self.reset_colors()
                ax.clear()
                ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                ax.set_title("Click to select point, S to confirm, R to reset, Q to quit")
                ax.axis('off')
                canvas.draw()
                print("已重置")
            elif event.key == 'q':
                selected_point_idx[0] = -1
                root.quit()
        
        # 连接事件
        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        # 创建画布
        canvas = FigureCanvasTkAgg(fig, root)
        canvas.draw()
        canvas.get_tk_widget().pack()
        
        # 运行界面
        root.mainloop()
        root.destroy()
        
        return selected_point_idx[0]

def main():
    """主函数"""
    ply_file_path = '/home/admin01/Projects/DexGarmentLab/highlighted_pointcloud.ply'
    threshold = 0.05
    save_path = 'highlighted_pointcloud.ply'

    while True:
        # 1. 加载点云
        print("加载点云文件...")
        point_cloud = o3d.io.read_point_cloud(ply_file_path)
        if not point_cloud.has_points():
            print("无法加载点云文件，程序退出")
            return
        
        print(f"成功加载点云文件：{ply_file_path}")
        print(f"点云包含 {len(point_cloud.points)} 个点")
        
        # 2. 创建2D交互器
        print("创建2D交互界面...")
        interaction = PointCloud2DInteraction(point_cloud, threshold)
        
        # 3. 在2D界面中选择点
        print("请在2D界面中点击选择点...")
        selected_point_idx = interaction.create_2d_interface()
        
        if selected_point_idx < 0:
            print("未选中任何点，程序退出")
            return
        
        # 4. 显示3D高亮结果
        print("显示3D高亮结果...")
        print("3D窗口：按 r 重新选点，按 q 退出，按 S 保存")
        key = show_highlight_and_wait_key(point_cloud)
        
        if key == 'r':
            print("重新选点...")
            continue
        elif key == 'q':
            print("退出程序")
            return
        elif key == 'enter':
            o3d.io.write_point_cloud(save_path, point_cloud)
            print(f"已保存高亮点云到 {save_path}")
            return
        else:
            print("未知操作，程序退出")
            return

def show_highlight_and_wait_key(point_cloud):
    import open3d as o3d
    key_result = {'key': None}

    def on_key_r(vis):
        key_result['key'] = 'r'
        vis.close()
        return False

    def on_key_q(vis):
        key_result['key'] = 'q'
        vis.close()
        return False

    def on_key_enter(vis):
        key_result['key'] = 'enter'
        vis.close()
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("高亮结果", width=1200, height=800)
    vis.add_geometry(point_cloud)
    vis.register_key_callback(ord('R'), on_key_r)
    vis.register_key_callback(ord('r'), on_key_r)
    vis.register_key_callback(ord('Q'), on_key_q)
    vis.register_key_callback(ord('q'), on_key_q)
    vis.register_key_callback(ord('S'), on_key_enter)  # Enter
    vis.register_key_callback(ord('s'), on_key_enter)  # Enter

    while vis.poll_events():
        vis.update_renderer()
        if key_result['key']:
            break
    vis.destroy_window()
    return key_result['key']

if __name__ == "__main__":
    main() 