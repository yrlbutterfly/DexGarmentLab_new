#!/usr/bin/env python3
"""
测试JSONL数据保存和加载功能
"""

import numpy as np
import os
from visualization_utils import HeadlessVisualizer

def create_test_data():
    """创建测试数据"""
    test_data = []
    
    for i in range(5):
        # 创建模拟数据
        data = {
            'step_num': i * 15,
            'joint_state': np.random.rand(12),  # 12个关节状态
            'image': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),  # 模拟RGB图像
            'env_point_cloud': np.random.rand(1000, 3),  # 1000个3D点
            'garment_point_cloud': np.random.rand(500, 3),  # 500个服装点
            'points_affordance_feature': np.random.rand(10, 2),  # 10个特征点
        }
        test_data.append(data)
    
    return test_data

def test_jsonl_save_load():
    """测试JSONL保存和加载功能"""
    print("=== 测试JSONL功能 ===")
    
    # 创建可视化器
    visualizer = HeadlessVisualizer("test_output")
    
    # 创建测试数据
    test_data = create_test_data()
    print(f"创建了 {len(test_data)} 帧测试数据")
    
    # 保存完整数据
    print("\n1. 保存完整数据（包含图像和点云）...")
    jsonl_path = visualizer.save_data_to_jsonl(
        test_data,
        filename="test_complete_data.jsonl",
        include_images=True,
        include_pointclouds=True,
        include_metadata=True
    )
    
    # 保存轻量级数据
    print("\n2. 保存轻量级数据（不包含图像和点云）...")
    jsonl_light_path = visualizer.save_data_to_jsonl(
        test_data,
        filename="test_light_data.jsonl",
        include_images=False,
        include_pointclouds=False,
        include_metadata=True
    )
    
    # 加载数据
    print("\n3. 加载完整数据...")
    loaded_data = visualizer.load_data_from_jsonl(jsonl_path)
    print(f"加载了 {len(loaded_data)} 帧数据")
    
    # 验证数据
    print("\n4. 验证数据完整性...")
    for i, (original, loaded) in enumerate(zip(test_data, loaded_data)):
        print(f"帧 {i}:")
        print(f"  - 关节状态形状: {original['joint_state'].shape} vs {loaded['joint_state'].shape}")
        print(f"  - 图像形状: {original['image'].shape} vs {loaded['image'].shape}")
        print(f"  - 环境点云形状: {original['env_point_cloud'].shape} vs {loaded['env_point_cloud'].shape}")
        print(f"  - 服装点云形状: {original['garment_point_cloud'].shape} vs {loaded['garment_point_cloud'].shape}")
        print(f"  - 特征形状: {original['points_affordance_feature'].shape} vs {loaded['points_affordance_feature'].shape}")
        
        # 检查数据是否一致
        joint_match = np.allclose(original['joint_state'], loaded['joint_state'])
        image_match = np.allclose(original['image'], loaded['image'])
        env_pcd_match = np.allclose(original['env_point_cloud'], loaded['env_point_cloud'])
        garment_pcd_match = np.allclose(original['garment_point_cloud'], loaded['garment_point_cloud'])
        feature_match = np.allclose(original['points_affordance_feature'], loaded['points_affordance_feature'])
        
        print(f"  - 数据一致性: 关节={joint_match}, 图像={image_match}, 环境点云={env_pcd_match}, 服装点云={garment_pcd_match}, 特征={feature_match}")
    
    print("\n=== JSONL测试完成 ===")
    print(f"完整数据文件: {jsonl_path}")
    print(f"轻量级数据文件: {jsonl_light_path}")
    
    # 显示文件大小
    complete_size = os.path.getsize(jsonl_path) / (1024 * 1024)  # MB
    light_size = os.path.getsize(jsonl_light_path) / 1024  # KB
    
    print(f"完整数据文件大小: {complete_size:.2f} MB")
    print(f"轻量级数据文件大小: {light_size:.2f} KB")

def test_jsonl_analysis():
    """测试JSONL数据分析功能"""
    print("\n=== 测试JSONL数据分析 ===")
    
    visualizer = HeadlessVisualizer("test_output")
    
    # 加载数据
    jsonl_path = "test_output/test_complete_data.jsonl"
    if os.path.exists(jsonl_path):
        data = visualizer.load_data_from_jsonl(jsonl_path)
        
        # 分析数据
        print(f"数据帧数: {len(data)}")
        
        if data:
            # 分析关节状态
            joint_states = [d['joint_state'] for d in data]
            joint_states = np.array(joint_states)
            print(f"关节状态形状: {joint_states.shape}")
            print(f"关节状态范围: [{joint_states.min():.3f}, {joint_states.max():.3f}]")
            
            # 分析图像
            if 'image' in data[0]:
                images = [d['image'] for d in data]
                image_shapes = [img.shape for img in images]
                print(f"图像形状: {image_shapes}")
            
            # 分析点云
            if 'env_point_cloud' in data[0]:
                env_pcd_sizes = [len(d['env_point_cloud']) for d in data]
                print(f"环境点云大小: {env_pcd_sizes}")
            
            if 'garment_point_cloud' in data[0]:
                garment_pcd_sizes = [len(d['garment_point_cloud']) for d in data]
                print(f"服装点云大小: {garment_pcd_sizes}")

if __name__ == "__main__":
    test_jsonl_save_load()
    test_jsonl_analysis() 