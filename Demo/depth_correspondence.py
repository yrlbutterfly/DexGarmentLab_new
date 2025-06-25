import numpy as np
import cv2
import open3d as o3d

class DepthCorrespondence:
    def __init__(self, camera_matrix):
        """
        基于深度图的2D-3D对应关系
        
        Args:
            camera_matrix: 相机内参矩阵 (3x3)
        """
        self.K = camera_matrix
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]
    
    def depth_to_pointcloud(self, depth_image, rgb_image=None):
        """
        将深度图转换为点云
        
        Args:
            depth_image: 深度图 (H, W)
            rgb_image: RGB图像 (H, W, 3)，可选
            
        Returns:
            points_3d: 3D点坐标 (N, 3)
            colors: 颜色信息 (N, 3)，如果提供RGB图像
            valid_mask: 有效深度掩码
        """
        height, width = depth_image.shape
        
        # 创建网格坐标
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # 计算3D坐标
        Z = depth_image
        X = (x - self.cx) * Z / self.fx
        Y = (y - self.cy) * Z / self.fy
        
        # 过滤无效深度值
        valid_mask = (Z > 0) & (Z < np.inf)
        
        # 提取有效点
        points_3d = np.stack([X[valid_mask], Y[valid_mask], Z[valid_mask]], axis=1)
        
        colors = None
        if rgb_image is not None:
            colors = rgb_image[valid_mask] / 255.0  # 归一化到[0,1]
        
        return points_3d, colors, valid_mask
    
    def pointcloud_to_depth(self, points_3d, image_shape):
        """
        将点云投影回深度图
        
        Args:
            points_3d: 3D点坐标 (N, 3)
            image_shape: 图像尺寸 (height, width)
            
        Returns:
            depth_image: 深度图
            color_image: 颜色图（如果点云有颜色）
        """
        height, width = image_shape
        
        # 投影到2D
        x = points_3d[:, 0] * self.fx / points_3d[:, 2] + self.cx
        y = points_3d[:, 1] * self.fy / points_3d[:, 2] + self.cy
        
        # 过滤在图像范围内的点
        valid_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        
        x = x[valid_mask].astype(int)
        y = y[valid_mask].astype(int)
        z = points_3d[valid_mask, 2]
        
        # 创建深度图
        depth_image = np.zeros((height, width), dtype=np.float32)
        depth_image[y, x] = z
        
        return depth_image
    
    def find_correspondence_from_depth(self, depth_image, rgb_image, target_points_2d):
        """
        基于深度图找到2D点对应的3D点
        
        Args:
            depth_image: 深度图
            rgb_image: RGB图像
            target_points_2d: 目标2D点坐标 (N, 2)
            
        Returns:
            correspondence: 对应关系 {2d_idx: (x, y, z)}
        """
        height, width = depth_image.shape
        correspondence = {}
        
        for i, (u, v) in enumerate(target_points_2d):
            u, v = int(u), int(v)
            
            # 检查点是否在图像范围内
            if 0 <= u < width and 0 <= v < height:
                depth = depth_image[v, u]
                
                # 检查深度值是否有效
                if depth > 0 and depth < np.inf:
                    # 计算3D坐标
                    x = (u - self.cx) * depth / self.fx
                    y = (v - self.cy) * depth / self.fy
                    z = depth
                    
                    correspondence[i] = (x, y, z)
        
        return correspondence

# 示例使用
def example_depth_correspondence():
    # 相机内参（需要根据实际相机调整）
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    # 创建对应关系计算器
    depth_calc = DepthCorrespondence(camera_matrix)
    
    # 加载深度图和RGB图像
    depth_image = cv2.imread("depth.png", cv2.IMREAD_ANYDEPTH)  # 16位深度图
    rgb_image = cv2.imread("rgb.jpg")
    
    # 将深度图转换为点云
    points_3d, colors, valid_mask = depth_calc.depth_to_pointcloud(depth_image, rgb_image)
    
    print(f"有效深度点数: {len(points_3d)}")
    print(f"深度图尺寸: {depth_image.shape}")
    
    # 示例：找到图像中某些点对应的3D坐标
    target_points_2d = np.array([[320, 240], [400, 300], [200, 180]])
    
    correspondence = depth_calc.find_correspondence_from_depth(
        depth_image, rgb_image, target_points_2d
    )
    
    print("2D到3D对应关系:")
    for idx_2d, coords_3d in correspondence.items():
        print(f"2D点 {idx_2d}: {target_points_2d[idx_2d]} -> 3D坐标: {coords_3d}")

if __name__ == "__main__":
    example_depth_correspondence() 