import numpy as np
import cv2
import open3d as o3d

class PointCorrespondence:
    def __init__(self, camera_matrix, dist_coeffs, rotation_matrix, translation_vector):
        """
        初始化点对应关系计算器
        
        Args:
            camera_matrix: 相机内参矩阵 (3x3)
            dist_coeffs: 畸变系数
            rotation_matrix: 旋转矩阵 (3x3)
            translation_vector: 平移向量 (3x1)
        """
        self.K = camera_matrix
        self.dist = dist_coeffs
        self.R = rotation_matrix
        self.t = translation_vector
        
    def project_3d_to_2d(self, points_3d):
        """
        将3D点投影到2D图像平面
        
        Args:
            points_3d: 3D点坐标 (N, 3)
            
        Returns:
            points_2d: 2D投影坐标 (N, 2)
            valid_mask: 有效点掩码
        """
        # 将3D点转换到相机坐标系
        points_cam = (self.R @ points_3d.T + self.t.reshape(3, 1)).T
        
        # 投影到2D
        points_2d_homo = (self.K @ points_cam.T).T
        points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
        
        # 应用畸变校正
        points_2d_undist = cv2.undistortPoints(
            points_2d.reshape(-1, 1, 2), 
            self.K, 
            self.dist, 
            P=self.K
        ).reshape(-1, 2)
        
        # 检查点是否在图像范围内
        valid_mask = (points_2d_undist[:, 0] >= 0) & (points_2d_undist[:, 0] < image_width) & \
                    (points_2d_undist[:, 1] >= 0) & (points_2d_undist[:, 1] < image_height)
        
        return points_2d_undist, valid_mask
    
    def find_2d_to_3d_correspondence(self, points_2d, points_3d, tolerance=2.0):
        """
        找到2D点对应的3D点
        
        Args:
            points_2d: 2D点坐标 (N, 2)
            points_3d: 3D点坐标 (M, 3)
            tolerance: 匹配容差（像素）
            
        Returns:
            correspondence: 对应关系字典 {2d_idx: 3d_idx}
        """
        # 投影所有3D点到2D
        projected_2d, valid_mask = self.project_3d_to_2d(points_3d)
        valid_3d_indices = np.where(valid_mask)[0]
        valid_projected_2d = projected_2d[valid_mask]
        
        correspondence = {}
        
        for i, point_2d in enumerate(points_2d):
            # 计算距离
            distances = np.linalg.norm(valid_projected_2d - point_2d, axis=1)
            min_idx = np.argmin(distances)
            min_distance = distances[min_idx]
            
            if min_distance <= tolerance:
                correspondence[i] = valid_3d_indices[min_idx]
        
        return correspondence

# 示例使用
def example_usage():
    # 假设的相机参数（需要根据实际情况调整）
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    dist_coeffs = np.array([k1, k2, p1, p2, k3])  # 畸变系数
    
    # 相机外参（相对于点云坐标系）
    rotation_matrix = np.eye(3)  # 需要根据实际标定结果调整
    translation_vector = np.array([0, 0, 0])  # 需要根据实际标定结果调整
    
    # 创建对应关系计算器
    correspondence_calc = PointCorrespondence(
        camera_matrix, dist_coeffs, rotation_matrix, translation_vector
    )
    
    # 加载点云
    point_cloud = o3d.io.read_point_cloud("data_0.ply")
    points_3d = np.asarray(point_cloud.points)
    
    # 加载RGB图像
    rgb_image = cv2.imread("rgb_image.jpg")
    image_height, image_width = rgb_image.shape[:2]
    
    # 示例：找到图像中某些点对应的3D点
    points_2d = np.array([[320, 240], [400, 300], [200, 180]])  # 示例2D点
    
    correspondence = correspondence_calc.find_2d_to_3d_correspondence(
        points_2d, points_3d, tolerance=5.0
    )
    
    print("2D到3D对应关系:")
    for idx_2d, idx_3d in correspondence.items():
        print(f"2D点 {idx_2d}: {points_2d[idx_2d]} -> 3D点 {idx_3d}: {points_3d[idx_3d]}")

if __name__ == "__main__":
    example_usage() 