import numpy as np
import cv2
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

class FeatureCorrespondence:
    def __init__(self, camera_matrix):
        """
        基于特征匹配的2D-3D对应关系
        
        Args:
            camera_matrix: 相机内参矩阵
        """
        self.K = camera_matrix
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]
    
    def extract_features_from_pointcloud(self, point_cloud, rgb_image):
        """
        从点云和RGB图像中提取特征点
        
        Args:
            point_cloud: Open3D点云对象
            rgb_image: RGB图像
            
        Returns:
            points_3d: 3D特征点坐标
            points_2d: 2D特征点坐标
            features: 特征描述符
        """
        # 将点云投影到2D
        points_3d = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)
        
        # 投影到2D
        x = points_3d[:, 0] * self.fx / points_3d[:, 2] + self.cx
        y = points_3d[:, 1] * self.fy / points_3d[:, 2] + self.cy
        
        # 过滤在图像范围内的点
        height, width = rgb_image.shape[:2]
        valid_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        
        valid_points_3d = points_3d[valid_mask]
        valid_points_2d = np.stack([x[valid_mask], y[valid_mask]], axis=1)
        valid_colors = colors[valid_mask]
        
        # 使用SIFT提取特征
        sift = cv2.SIFT_create()
        keypoints_2d, descriptors = sift.detectAndCompute(rgb_image, None)
        
        if keypoints_2d is None:
            return valid_points_3d, valid_points_2d, None
        
        # 将关键点坐标转换为numpy数组
        keypoints_coords = np.array([kp.pt for kp in keypoints_2d])
        
        # 找到最近的点云点
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(valid_points_2d)
        distances, indices = nn.kneighbors(keypoints_coords)
        
        # 过滤距离太远的匹配
        max_distance = 5.0  # 像素距离阈值
        good_matches = distances.flatten() < max_distance
        
        matched_points_3d = valid_points_3d[indices[good_matches].flatten()]
        matched_points_2d = keypoints_coords[good_matches]
        matched_descriptors = descriptors[good_matches]
        
        return matched_points_3d, matched_points_2d, matched_descriptors
    
    def find_correspondence_by_features(self, target_image, point_cloud, rgb_image):
        """
        通过特征匹配找到对应关系
        
        Args:
            target_image: 目标RGB图像
            point_cloud: 点云数据
            rgb_image: 点云对应的RGB图像
            
        Returns:
            correspondence: 对应关系
        """
        # 提取特征
        points_3d, points_2d, descriptors_cloud = self.extract_features_from_pointcloud(
            point_cloud, rgb_image
        )
        
        # 从目标图像提取特征
        sift = cv2.SIFT_create()
        keypoints_target, descriptors_target = sift.detectAndCompute(target_image, None)
        
        if descriptors_cloud is None or descriptors_target is None:
            return {}
        
        # 特征匹配
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors_cloud, descriptors_target, k=2)
        
        # 应用比率测试
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        # 构建对应关系
        correspondence = {}
        for match in good_matches:
            cloud_idx = match.queryIdx
            target_idx = match.trainIdx
            
            if cloud_idx < len(points_3d) and target_idx < len(keypoints_target):
                target_point = keypoints_target[target_idx].pt
                cloud_point_3d = points_3d[cloud_idx]
                cloud_point_2d = points_2d[cloud_idx]
                
                correspondence[target_idx] = {
                    'target_2d': target_point,
                    'cloud_3d': cloud_point_3d,
                    'cloud_2d': cloud_point_2d,
                    'distance': match.distance
                }
        
        return correspondence
    
    def visualize_correspondence(self, target_image, correspondence, point_cloud, rgb_image):
        """
        可视化对应关系
        
        Args:
            target_image: 目标图像
            correspondence: 对应关系
            point_cloud: 点云
            rgb_image: 点云对应的RGB图像
        """
        # 创建可视化图像
        vis_image = target_image.copy()
        
        # 绘制匹配点
        for idx, corr in correspondence.items():
            target_pt = tuple(map(int, corr['target_2d']))
            cv2.circle(vis_image, target_pt, 5, (0, 255, 0), -1)
            cv2.putText(vis_image, f"{idx}", target_pt, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # 显示图像
        cv2.imshow("Feature Correspondence", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 可视化点云中的对应点
        points_3d = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)
        
        # 高亮显示匹配的点
        for corr in correspondence.values():
            cloud_3d = corr['cloud_3d']
            # 找到最近的点云点
            distances = np.linalg.norm(points_3d - cloud_3d, axis=1)
            nearest_idx = np.argmin(distances)
            colors[nearest_idx] = [1.0, 0.0, 0.0]  # 红色
        
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        
        # 显示点云
        o3d.visualization.draw_geometries([point_cloud])

# 示例使用
def example_feature_correspondence():
    # 相机内参
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    # 创建特征对应关系计算器
    feature_calc = FeatureCorrespondence(camera_matrix)
    
    # 加载数据
    point_cloud = o3d.io.read_point_cloud("data_0.ply")
    rgb_image = cv2.imread("rgb_image.jpg")
    target_image = cv2.imread("target_image.jpg")
    
    # 找到对应关系
    correspondence = feature_calc.find_correspondence_by_features(
        target_image, point_cloud, rgb_image
    )
    
    print(f"找到 {len(correspondence)} 个对应关系")
    
    # 显示对应关系
    for idx, corr in correspondence.items():
        print(f"目标图像点 {idx}: {corr['target_2d']} -> "
              f"点云3D坐标: {corr['cloud_3d']}")
    
    # 可视化结果
    feature_calc.visualize_correspondence(
        target_image, correspondence, point_cloud, rgb_image
    )

if __name__ == "__main__":
    example_feature_correspondence() 