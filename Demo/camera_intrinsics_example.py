import numpy as np
import cv2
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_rot_matrix

def get_camera_intrinsics_example():
    """
    示例：如何使用Isaac Sim Camera的内置方法获取内参矩阵
    """
    print("=== Isaac Sim Camera 内参获取示例 ===")
    
    # 假设你已经有了一个Isaac Sim Camera对象
    # camera = env.garment_camera.camera  # 从Recording_Camera中获取Camera对象
    
    # 方法1：使用get_intrinsics_matrix()获取完整的内参矩阵
    # intrinsics_matrix = camera.get_intrinsics_matrix()
    # print(f"内参矩阵 K:\n{intrinsics_matrix}")
    
    # 方法2：获取内参的各个组成部分
    # focal_length = camera.get_focal_length()
    # horizontal_aperture = camera.get_horizontal_aperture()
    # vertical_aperture = camera.get_vertical_aperture()
    # width, height = camera.get_resolution()
    
    # fx = width * focal_length / horizontal_aperture
    # fy = height * focal_length / vertical_aperture
    # cx = width * 0.5
    # cy = height * 0.5
    
    # print(f"焦距 (像素): fx={fx:.2f}, fy={fy:.2f}")
    # print(f"主点坐标: cx={cx:.2f}, cy={cy:.2f}")
    # print(f"图像分辨率: {width}x{height}")
    
    # 方法3：获取视场角
    # horizontal_fov = camera.get_horizontal_fov()
    # vertical_fov = camera.get_vertical_fov()
    # print(f"视场角: 水平={np.degrees(horizontal_fov):.2f}°, 垂直={np.degrees(vertical_fov):.2f}°")
    
    print("注意：要使用这些方法，需要确保相机已经初始化并且投影类型设置为'pinhole'")

def project_3d_points_to_2d_with_intrinsics(points_3d, camera_position, camera_orientation, camera_object):
    """
    使用Isaac Sim Camera的内置内参矩阵进行3D到2D投影
    
    Args:
        points_3d: 3D点坐标 (N, 3)
        camera_position: 相机位置 [x, y, z]
        camera_orientation: 相机方向 [roll, pitch, yaw] (度)
        camera_object: Isaac Sim Camera对象
        
    Returns:
        points_2d: 2D投影坐标 (N, 2)
        valid_mask: 有效点掩码
    """
    # 使用Isaac Sim Camera的内置方法获取内参矩阵
    K = camera_object.get_intrinsics_matrix()
    print(f"获取到的内参矩阵:\n{K}")
    
    # 获取图像分辨率
    width, height = camera_object.get_resolution()
    print(f"图像分辨率: {width}x{height}")
    
    # 相机外参：从世界坐标系到相机坐标系的变换
    quat = euler_angles_to_quat(camera_orientation, degrees=True)
    R = quat_to_rot_matrix(quat)
    
    # 平移向量：相机位置
    t = -R @ np.array(camera_position)
    
    # 将3D点转换到相机坐标系
    points_cam = (R @ points_3d.T + t.reshape(3, 1)).T
    
    # 投影到2D
    points_2d_homo = (K @ points_cam.T).T
    points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
    
    # 检查点是否在图像范围内且在前方
    valid_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) & \
                (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height) & \
                (points_cam[:, 2] > 0)  # 在相机前方
    
    return points_2d, valid_mask

def compare_intrinsics_methods():
    """
    比较不同获取内参的方法
    """
    print("\n=== 内参获取方法比较 ===")
    
    # 假设的分辨率
    width, height = 640, 480
    
    # 方法1：基于分辨率的估算（原来的方法）
    fx_estimated = fy_estimated = min(width, height) * 0.8
    cx_estimated, cy_estimated = width / 2, height / 2
    
    K_estimated = np.array([[fx_estimated, 0, cx_estimated],
                           [0, fy_estimated, cy_estimated],
                           [0, 0, 1]])
    
    print("方法1 - 基于分辨率估算:")
    print(f"  fx = fy = {fx_estimated:.2f}")
    print(f"  cx = {cx_estimated:.2f}, cy = {cy_estimated:.2f}")
    print(f"  内参矩阵:\n{K_estimated}")
    
    # 方法2：Isaac Sim Camera的内置方法（推荐）
    print("\n方法2 - Isaac Sim Camera内置方法（推荐）:")
    print("  K = camera.get_intrinsics_matrix()")
    print("  这种方法会考虑相机的实际物理参数（焦距、光圈等）")
    print("  提供更准确的内参矩阵")
    
    print("\n推荐使用Isaac Sim Camera的内置方法，因为它:")
    print("1. 考虑了相机的实际物理参数")
    print("2. 提供了更准确的内参矩阵")
    print("3. 自动处理不同分辨率的缩放")
    print("4. 与Isaac Sim的渲染系统保持一致")

if __name__ == "__main__":
    get_camera_intrinsics_example()
    compare_intrinsics_methods() 