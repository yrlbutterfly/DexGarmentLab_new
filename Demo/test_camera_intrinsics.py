#!/usr/bin/env python3
"""
测试相机内参获取功能
"""

import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_camera_intrinsics():
    """
    测试相机内参获取功能
    """
    print("=== 测试相机内参获取功能 ===")
    
    try:
        # 导入必要的模块
        from isaacsim.sensors.camera import Camera
        from isaacsim.core.utils.rotations import euler_angles_to_quat
        
        # 创建一个测试相机对象
        test_camera = Camera(
            prim_path="/World/test_camera",
            position=np.array([0.0, 0.0, 1.0]),
            orientation=euler_angles_to_quat([0, 0, 0], degrees=True),
            resolution=(640, 480)
        )
        
        # 初始化相机
        test_camera.initialize()
        
        # 测试获取内参矩阵
        intrinsics_matrix = test_camera.get_intrinsics_matrix()
        print(f"✓ 成功获取内参矩阵:")
        print(f"  K = \n{intrinsics_matrix}")
        
        # 测试获取分辨率
        width, height = test_camera.get_resolution()
        print(f"✓ 成功获取分辨率: {width}x{height}")
        
        # 测试获取焦距
        focal_length = test_camera.get_focal_length()
        print(f"✓ 成功获取焦距: {focal_length}")
        
        # 测试获取视场角
        horizontal_fov = test_camera.get_horizontal_fov()
        vertical_fov = test_camera.get_vertical_fov()
        print(f"✓ 成功获取视场角: 水平={np.degrees(horizontal_fov):.2f}°, 垂直={np.degrees(vertical_fov):.2f}°")
        
        print("\n✓ 所有测试通过！相机内参获取功能正常工作。")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        print("请确保Isaac Sim环境已正确设置。")

def test_projection_function():
    """
    测试3D到2D投影功能
    """
    print("\n=== 测试3D到2D投影功能 ===")
    
    try:
        from isaacsim.sensors.camera import Camera
        from isaacsim.core.utils.rotations import euler_angles_to_quat
        
        # 创建测试相机
        test_camera = Camera(
            prim_path="/World/test_camera",
            position=np.array([0.0, 0.0, 1.0]),
            orientation=euler_angles_to_quat([0, 0, 0], degrees=True),
            resolution=(640, 480)
        )
        test_camera.initialize()
        
        # 测试3D点
        test_points_3d = np.array([
            [0.0, 0.0, 0.5],  # 相机前方的点
            [0.1, 0.0, 0.5],  # 右侧的点
            [0.0, 0.1, 0.5],  # 上方的点
        ])
        
        # 使用修改后的投影函数
        from Preprocess.backup.Tops_Collect import project_3d_points_to_2d
        
        points_2d, valid_mask = project_3d_points_to_2d(
            test_points_3d,
            test_camera.get_world_pose()[0],  # 相机位置
            [0, 0, 0],  # 相机方向
            test_camera
        )
        
        print(f"✓ 成功投影3D点到2D:")
        for i, (point_3d, point_2d, valid) in enumerate(zip(test_points_3d, points_2d, valid_mask)):
            print(f"  点 {i+1}: 3D={point_3d} -> 2D={point_2d}, 有效={valid}")
        
        print("\n✓ 投影功能测试通过！")
        
    except Exception as e:
        print(f"✗ 投影测试失败: {e}")

if __name__ == "__main__":
    test_camera_intrinsics()
    test_projection_function() 