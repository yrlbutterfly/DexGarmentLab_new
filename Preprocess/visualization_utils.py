import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import open3d as o3d
import os
import json
import base64
from typing import Optional, List, Tuple, Dict, Any
import cv2

class HeadlessVisualizer:
    """在headless模式下工作的可视化工具"""
    
    def __init__(self, output_dir: str = "visualization_output"):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def save_rgb_image(self, rgb_image: np.ndarray, frame_num: int, 
                      title: str = "RGB Image", filename: Optional[str] = None) -> str:
        """
        保存RGB图像
        
        Args:
            rgb_image: RGB图像数组
            frame_num: 帧号
            title: 图像标题
            filename: 自定义文件名
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            filename = f"rgb_frame_{frame_num:04d}.png"
        
        filepath = os.path.join(self.output_dir, filename)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(rgb_image)
        plt.title(f"{title} - Frame {frame_num}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filepath, dpi=500, bbox_inches='tight')
        plt.close()
        
        print(f"Saved RGB image: {filepath}")
        return filepath
    
    def save_pointcloud(self, points: np.ndarray, frame_num: int,
                       colors: Optional[np.ndarray] = None,
                       filename: Optional[str] = None) -> str:
        """
        保存点云数据
        
        Args:
            points: 点云坐标 (N, 3)
            frame_num: 帧号
            colors: 点云颜色 (N, 3)，可选
            filename: 自定义文件名
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            filename = f"pointcloud_frame_{frame_num:04d}.ply"
        
        filepath = os.path.join(self.output_dir, filename)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_point_cloud(filepath, pcd)
        print(f"Saved point cloud: {filepath}")
        return filepath
    
    def save_pointcloud_with_highlight(self, points: np.ndarray, 
                                     highlight_indices: List[int],
                                     frame_num: int,
                                     highlight_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
                                     base_color: Tuple[float, float, float] = (0.7, 0.7, 0.7),
                                     filename: Optional[str] = None) -> str:
        """
        保存带高亮点的点云
        
        Args:
            points: 点云坐标 (N, 3)
            highlight_indices: 要高亮的点索引
            frame_num: 帧号
            highlight_color: 高亮颜色 (R, G, B)
            base_color: 基础颜色 (R, G, B)
            filename: 自定义文件名
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            filename = f"highlighted_pointcloud_frame_{frame_num:04d}.ply"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # 创建颜色数组
        colors = np.full((len(points), 3), base_color)
        colors[highlight_indices] = highlight_color
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_point_cloud(filepath, pcd)
        print(f"Saved highlighted point cloud: {filepath}")
        return filepath
    
    def save_manipulation_points_visualization(self, points: np.ndarray, 
                                            manipulation_points: np.ndarray,
                                            frame_num: int,
                                            filename: Optional[str] = None) -> str:
        """
        保存操作点可视化
        
        Args:
            points: 完整点云坐标
            manipulation_points: 操作点坐标
            frame_num: 帧号
            filename: 自定义文件名
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            filename = f"manipulation_points_frame_{frame_num:04d}.ply"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # 创建颜色数组
        colors = np.full((len(points), 3), (0.7, 0.7, 0.7))  # 灰色基础
        
        # 找到最近的操作点并高亮
        for mp in manipulation_points:
            distances = np.linalg.norm(points - mp, axis=1)
            nearby_mask = distances < 0.05  # 5cm阈值
            colors[nearby_mask] = (1.0, 0.0, 0.0)  # 红色高亮
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_point_cloud(filepath, pcd)
        print(f"Saved manipulation points visualization: {filepath}")
        return filepath
    
    def create_summary_plot(self, data_list: List[dict], save_path: Optional[str] = None) -> str:
        """
        创建数据摘要图表
        
        Args:
            data_list: 数据列表，每个元素包含step_num等信息
            save_path: 保存路径
            
        Returns:
            保存的文件路径
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "data_summary.png")
        
        # 提取数据
        step_nums = [data.get('step_num', i) for i, data in enumerate(data_list)]
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 步数统计
        axes[0, 0].hist(step_nums, bins=20, alpha=0.7, color='blue')
        axes[0, 0].set_title('Step Number Distribution')
        axes[0, 0].set_xlabel('Step Number')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. 数据收集时间线
        axes[0, 1].plot(step_nums, 'o-', alpha=0.7)
        axes[0, 1].set_title('Data Collection Timeline')
        axes[0, 1].set_xlabel('Data Index')
        axes[0, 1].set_ylabel('Step Number')
        
        # 3. 关节状态统计（如果有）
        if data_list and 'joint_state' in data_list[0]:
            joint_states = [data['joint_state'] for data in data_list]
            joint_states = np.array(joint_states)
            
            axes[1, 0].boxplot(joint_states.T)
            axes[1, 0].set_title('Joint States Distribution')
            axes[1, 0].set_xlabel('Joint Index')
            axes[1, 0].set_ylabel('Joint Position')
        
        # 4. 点云大小统计（如果有）
        if data_list and 'garment_point_cloud' in data_list[0]:
            pcd_sizes = [len(data['garment_point_cloud']) if data['garment_point_cloud'] is not None else 0 
                        for data in data_list]
            
            axes[1, 1].plot(pcd_sizes, 'o-', alpha=0.7, color='green')
            axes[1, 1].set_title('Point Cloud Size Over Time')
            axes[1, 1].set_xlabel('Data Index')
            axes[1, 1].set_ylabel('Point Cloud Size')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved summary plot: {save_path}")
        return save_path

    def save_data_to_jsonl(self, data_list: List[Dict[str, Any]], 
                          filename: Optional[str] = None,
                          include_images: bool = True,
                          include_pointclouds: bool = True,
                          include_metadata: bool = True) -> str:
        """
        将数据保存为JSONL格式
        
        Args:
            data_list: 数据列表
            filename: 输出文件名
            include_images: 是否包含图像数据
            include_pointclouds: 是否包含点云数据
            include_metadata: 是否包含元数据
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            filename = f"data_collection_{len(data_list)}_frames.jsonl"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for i, data in enumerate(data_list):
                json_line = {}
                
                # 添加元数据
                if include_metadata:
                    json_line['frame_id'] = i
                    json_line['step_num'] = data.get('step_num', i)
                    json_line['timestamp'] = data.get('timestamp', None)
                
                # 添加关节状态
                if 'joint_state' in data:
                    json_line['joint_state'] = data['joint_state'].tolist() if isinstance(data['joint_state'], np.ndarray) else data['joint_state']
                
                # 添加图像数据（base64编码）
                if include_images and 'image' in data and data['image'] is not None:
                    try:
                        # 将RGB图像转换为JPEG格式的base64字符串
                        image = data['image']
                        if isinstance(image, np.ndarray):
                            # 确保图像是uint8格式
                            if image.dtype != np.uint8:
                                image = (image * 255).astype(np.uint8)
                            
                            # 转换为BGR格式用于OpenCV
                            if len(image.shape) == 3 and image.shape[2] == 3:
                                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            else:
                                image_bgr = image
                            
                            # 编码为JPEG
                            _, buffer = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                            image_base64 = base64.b64encode(buffer).decode('utf-8')
                            json_line['image_base64'] = image_base64
                            json_line['image_shape'] = list(image.shape)
                    except Exception as e:
                        print(f"Failed to encode image for frame {i}: {e}")
                
                # 添加点云数据
                if include_pointclouds:
                    # 环境点云
                    if 'env_point_cloud' in data and data['env_point_cloud'] is not None:
                        try:
                            pcd_data = data['env_point_cloud']
                            if isinstance(pcd_data, np.ndarray):
                                json_line['env_point_cloud'] = pcd_data.tolist()
                                json_line['env_point_cloud_shape'] = list(pcd_data.shape)
                        except Exception as e:
                            print(f"Failed to serialize env point cloud for frame {i}: {e}")
                    
                    # 服装点云
                    if 'garment_point_cloud' in data and data['garment_point_cloud'] is not None:
                        try:
                            pcd_data = data['garment_point_cloud']
                            if isinstance(pcd_data, np.ndarray):
                                json_line['garment_point_cloud'] = pcd_data.tolist()
                                json_line['garment_point_cloud_shape'] = list(pcd_data.shape)
                        except Exception as e:
                            print(f"Failed to serialize garment point cloud for frame {i}: {e}")
                
                # 添加特征数据
                if 'points_affordance_feature' in data and data['points_affordance_feature'] is not None:
                    try:
                        feature_data = data['points_affordance_feature']
                        if isinstance(feature_data, np.ndarray):
                            json_line['points_affordance_feature'] = feature_data.tolist()
                            json_line['points_affordance_feature_shape'] = list(feature_data.shape)
                    except Exception as e:
                        print(f"Failed to serialize affordance features for frame {i}: {e}")
                
                # 写入JSON行
                f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(data_list)} frames to JSONL: {filepath}")
        return filepath
    
    def load_data_from_jsonl(self, filepath: str) -> List[Dict[str, Any]]:
        """
        从JSONL文件加载数据
        
        Args:
            filepath: JSONL文件路径
            
        Returns:
            数据列表
        """
        data_list = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    
                    # 转换回numpy数组
                    if 'joint_state' in data:
                        data['joint_state'] = np.array(data['joint_state'])
                    
                    if 'env_point_cloud' in data:
                        data['env_point_cloud'] = np.array(data['env_point_cloud'])
                    
                    if 'garment_point_cloud' in data:
                        data['garment_point_cloud'] = np.array(data['garment_point_cloud'])
                    
                    if 'points_affordance_feature' in data:
                        data['points_affordance_feature'] = np.array(data['points_affordance_feature'])
                    
                    # 解码图像（如果需要）
                    if 'image_base64' in data:
                        try:
                            image_base64 = data['image_base64']
                            image_bytes = base64.b64decode(image_base64)
                            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                            # 转换回RGB格式
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            data['image'] = image_rgb
                            # 删除base64数据以节省内存
                            del data['image_base64']
                        except Exception as e:
                            print(f"Failed to decode image for frame {line_num}: {e}")
                    
                    data_list.append(data)
                    
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON at line {line_num}: {e}")
                    continue
        
        print(f"Loaded {len(data_list)} frames from JSONL: {filepath}")
        return data_list

def create_visualization_report(output_dir: str = "visualization_output") -> str:
    """
    创建可视化报告
    
    Args:
        output_dir: 输出目录
        
    Returns:
        报告文件路径
    """
    report_path = os.path.join(output_dir, "visualization_report.html")
    
    # 获取所有生成的文件
    files = os.listdir(output_dir)
    rgb_files = [f for f in files if f.startswith("rgb_frame_") and f.endswith(".png")]
    pcd_files = [f for f in files if f.startswith("pointcloud_frame_") and f.endswith(".ply")]
    
    # 生成HTML报告
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Visualization Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .section {{ margin: 20px 0; }}
            .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            .image-item {{ text-align: center; }}
            .image-item img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            .file-list {{ background: #f5f5f5; padding: 10px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>Visualization Report</h1>
        
        <div class="section">
            <h2>Generated Files</h2>
            <div class="file-list">
                <h3>RGB Images ({len(rgb_files)} files):</h3>
                <ul>
                    {''.join([f'<li>{f}</li>' for f in sorted(rgb_files)])}
                </ul>
                
                <h3>Point Clouds ({len(pcd_files)} files):</h3>
                <ul>
                    {''.join([f'<li>{f}</li>' for f in sorted(pcd_files)])}
                </ul>
            </div>
        </div>
        
        <div class="section">
            <h2>RGB Images</h2>
            <div class="image-grid">
                {''.join([f'''
                <div class="image-item">
                    <img src="{f}" alt="{f}">
                    <p>{f}</p>
                </div>
                ''' for f in sorted(rgb_files)])}
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"Created visualization report: {report_path}")
    return report_path 