import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
from torchvision.transforms.functional import to_tensor

def preview_video(frames, is_rgb=True, title="Video Preview", speed_multiplier=2):
    """
    使用OpenCV显示视频预览，支持播放控制
    
    Parameters:
    frames (np.ndarray): Video frames of shape (T, H, W, C)
    title (str): Window title
    speed_multiplier (int): Playback speed multiplier
    
    Controls:
    - SPACE: Pause/Resume
    - R: Restart video
    - Q: Continue processing
    - ESC: Skip this file
    
    Returns:
    bool: True if user wants to skip this file, False to continue processing
    """
    frame_idx = 0
    playing = True
    delay = int(1000 / (30 * speed_multiplier))  # 基础延迟30fps
    
    # # 先清理可能存在的窗口
    # try:
    #     cv2.destroyAllWindows()
    #     cv2.waitKey(1)  # 给系统一点时间处理
    # except:
    #     pass

    # print(f"---{title}---")
    try:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 800, 800)
    except Exception as e:
        print(f"Error creating window: {e}")
        return False
    
    while True:

        # 检查窗口是否还存在
        try:
            # getWindowProperty 返回-1表示窗口已关闭
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                print("Window was closed by user")
                cv2.destroyWindow(title)
                return False
        except:
            print("Window was closed by user")
            return False


        if playing:
            frame = frames[frame_idx].copy()
            
            # 添加进度条和控制说明
            total_frames = len(frames)
            progress = int((frame_idx / total_frames) * 100)
            
            # 添加控制说明文字
            info_text = f"Frame: {frame_idx}/{total_frames} ({progress}%)"
            controls_text = "SPACE: Play/Pause | R: Restart | Q: Continue | ESC: Skip"
            
            # 在图像上添加文字
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, controls_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 如果是RGB格式，转换为BGR
            if is_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(title, frame)
            
            if frame_idx < len(frames) - 1:
                frame_idx += 1
            else:
                playing = False  # 播放结束后暂停
        else:
            # 暂停时显示当前帧
            cv2.imshow(title, frames[frame_idx])
        
        key = cv2.waitKey(delay if playing else 0) & 0xFF # 阻塞主线程
        
        if key == ord(' '):  # 空格键：暂停/继续
            playing = not playing
        elif key == ord('r'):  # R键：重新开始
            frame_idx = 0
            playing = True
        elif key == ord('q'):  # Q键：继续处理
            cv2.destroyWindow(title)
            return False
        elif key == 27:  # ESC键：跳过文件
            cv2.destroyWindow(title)
            return True
    
    cv2.destroyWindow(title)
    return False

def get_bounding_box(first_frame, mask_camera_name='1viewer'):
    """
    Get user prompt (bounding box) for SAM2 video tracking
    
    Parameters:
    first_frame: First frame of the video
    mask_camera_name: Name for the mask camera
    
    Returns:
    bbox: Bounding box coordinates [x1, y1, x2, y2] or None if cancelled
    """
    while True:  # 添加外层循环
        points = []
        frame_copy = first_frame.copy()
        window_name = "Select Bounding Box"

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
                points.append((x, y))
                # 在点击位置画一个绿色圆点
                cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
                # 如果已有两个点，画出矩形框
                if len(points) == 2:
                    cv2.rectangle(frame_copy, points[0], points[1], (0, 255, 0), 2)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 800)
        cv2.setMouseCallback(window_name, mouse_callback)

        while True:
            img_display = frame_copy.copy()
            # Convert RGB to BGR for display
            # img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
            cv2.putText(img_display, 
                    f"{mask_camera_name} | Points: {len(points)}/2 | R: Reset | Enter: Confirm | ESC: Skip", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(window_name, img_display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                cv2.destroyWindow(window_name)
                return None
            elif key == ord('r'):  # Reset
                points = []
                frame_copy = first_frame.copy()
            elif key == 13 and len(points) == 2:  # Enter
                cv2.destroyWindow(window_name)
                break

        if len(points) != 2:
            return None

        (x1, y1), (x2, y2) = points
        bbox = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])

        # 显示预览
        preview = first_frame.copy() 
        preview = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR) # frame is rgb
        cv2.rectangle(preview, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        window_name = "Confirm Bounding Box"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 800)

        retry = False  # 添加标志位
        while True:
            preview_copy = preview.copy()
            cv2.putText(preview_copy, "Enter: Confirm | R: Retry | ESC: Skip",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window_name, preview_copy)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyWindow(window_name)
                return None
            elif key == ord('r'):  # Retry
                cv2.destroyWindow(window_name)
                retry = True  # 设置重试标志
                break
            elif key == 13:  # Enter
                cv2.destroyWindow(window_name)
                break

        if retry:
            continue  # 重试
        break
    
    return bbox

def preview_masks(masks, title="Mask Preview", speed_multiplier=2):
    """
    Preview binary masks using OpenCV
    
    Parameters:
    masks (np.ndarray): Binary masks of shape (T, H, W) or (T, 1, H, W)
    title (str): Window title
    speed_multiplier (int): Playback speed multiplier
    
    Returns:
    bool: True if user wants to skip this file, False to continue processing
    """
    # Handle different mask shapes
    if len(masks.shape) == 4:
        masks = masks.squeeze(1)  # Remove channel dimension if present
    
    frame_idx = 0
    playing = True
    delay = int(1000 / (30 * speed_multiplier))
    
    try:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 800, 800)
    except Exception as e:
        print(f"Error creating window: {e}")
        return False
    
    while True:
        # Check if window still exists
        try:
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                print("Window was closed by user")
                cv2.destroyWindow(title)
                return False
        except:
            print("Window was closed by user")
            return False

        if playing:
            # Convert binary mask to 8-bit image
            mask = masks[frame_idx].astype(np.uint8) * 255
            
            # Convert to 3-channel for better visualization
            mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            
            # Add progress info
            total_frames = len(masks)
            progress = int((frame_idx / total_frames) * 100)
            
            info_text = f"Frame: {frame_idx}/{total_frames} ({progress}%)"
            controls_text = "SPACE: Play/Pause | R: Restart | Q: Continue | ESC: Skip"
            
            cv2.putText(mask_colored, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(mask_colored, controls_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(title, mask_colored)
            
            if frame_idx < len(masks) - 1:
                frame_idx += 1
            else:
                playing = False
        else:
            # Show current frame when paused
            mask = masks[frame_idx].astype(np.uint8) * 255
            mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            cv2.imshow(title, mask_colored)
        
        key = cv2.waitKey(delay if playing else 0) & 0xFF
        
        if key == ord(' '):  # Space: pause/resume
            playing = not playing
        elif key == ord('r'):  # R: restart
            frame_idx = 0
            playing = True
        elif key == ord('q'):  # Q: continue
            cv2.destroyWindow(title)
            return False
        elif key == 27:  # ESC: skip
            cv2.destroyWindow(title)
            return True
    
    cv2.destroyWindow(title)
    return False

def sam2_track_video(predictor, video_path, bbox, reverse_tracking=False):
    """
    使用SAM2追踪视频中的对象
    
    Parameters:
    predictor: SAM2预测器
    video_path: 视频文件路径
    bbox: 边界框坐标
    reverse_tracking: 是否使用反向追踪
    
    Returns:
    processed_masks: 处理后的mask数组
    """
    processed_masks = []
    temp_video_path = None
    
    # 创建临时视频文件用于反向追踪
    if reverse_tracking:
        print("创建反向视频用于追踪...")
        
        # 读取原视频
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        # 反转帧顺序
        reversed_frames = frames[::-1]
        
        # 创建临时反向视频文件
        temp_video_path = video_path.replace('.mp4', '_reversed_temp.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        if len(reversed_frames) > 0:
            height, width = reversed_frames[0].shape[:2]
            out = cv2.VideoWriter(temp_video_path, fourcc, 30.0, (width, height))
            
            for frame in reversed_frames:
                out.write(frame)
            out.release()
            
            # 使用反向视频进行追踪
            tracking_video_path = temp_video_path
        else:
            print("无法读取视频帧，使用原视频")
            tracking_video_path = video_path
    else:
        # 正向追踪
        tracking_video_path = video_path
    
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Initialize state with video file path
            state = predictor.init_state(tracking_video_path)
            
            # Add bounding box prompt to starting frame
            obj_id = 1  # Object ID for tracking
            tracking_start_frame_idx = 0
            
            # Add new bounding box prompt
            frame_idx, object_ids, masks = predictor.add_new_points_or_box(
                state, 
                frame_idx=tracking_start_frame_idx,
                obj_id=obj_id,
                box=bbox
            )
            
            # Get the initial mask
            initial_mask = masks[0].cpu().numpy()
            processed_masks.append(initial_mask)
            
            # Propagate through the video
            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                # Extract mask for our object
                mask = masks[0].cpu().numpy()  # First (and only) object
                processed_masks.append(mask)
        
        # 清理临时文件
        if reverse_tracking and temp_video_path:
            import os
            try:
                os.remove(temp_video_path)
                print(f"已删除临时文件: {temp_video_path}")
            except:
                print(f"无法删除临时文件: {temp_video_path}")
        
        processed_masks = np.array(processed_masks)
        
        # 如果是反向追踪，需要将mask结果也反转回来
        if reverse_tracking:
            processed_masks = processed_masks[::-1]
            print("已将mask结果反转回正确顺序")
        
        return processed_masks
        
    except Exception as e:
        print(f"追踪过程中发生错误: {e}")
        # 清理临时文件
        if reverse_tracking and temp_video_path:
            import os
            try:
                os.remove(temp_video_path)
            except:
                pass
        return None

if __name__ == '__main__':
    # Load video
    camera_rgbs = []
    video_path = '/home/admin01/Projects/DexGarmentLab/Data/Fold_Dress/vedio/vedio_0.mp4'
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB for SAM2
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        camera_rgbs.append(frame_rgb)
    cap.release()
    camera_rgbs = np.array(camera_rgbs)
    print(f"Video shape: {camera_rgbs.shape}")
    # camera_rgbs = camera_rgbs[10:]

    # 选择追踪方向
    print("选择追踪方向:")
    print("1. 从前往后追踪 (默认)")
    print("2. 从后往前追踪")
    choice = input("请输入选择 (1 或 2): ").strip()
    
    reverse_tracking = (choice == '2')
    
    if reverse_tracking:
        print("使用从后往前追踪模式")
        # 使用最后一帧作为起始帧
        reference_frame = camera_rgbs[-1]
        print(f"使用最后一帧 (帧{len(camera_rgbs)-1}) 作为起始帧")
    else:
        print("使用从前往后追踪模式")
        # 使用第一帧作为起始帧
        reference_frame = camera_rgbs[0]
        print(f"使用第一帧 (帧0) 作为起始帧")

    # Get user prompt for reference frame
    bbox = get_bounding_box(reference_frame)
    
    print(f"Bounding box: {bbox}")
    if bbox is None:
        print("No bounding box selected, exiting...")
        exit()

    checkpoint = "/home/admin01/Projects/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)

    print(f"Predictor: {predictor}")
    
    # 使用SAM2追踪视频
    processed_masks = sam2_track_video(predictor, video_path, bbox, reverse_tracking)
    
    if processed_masks is None:
        print("追踪失败，退出程序")
        exit()
    
    print(f"Processed masks shape: {processed_masks.shape}")
    print(processed_masks.dtype)
    print(np.max(processed_masks), np.min(processed_masks))

    # 创建可视化帧
    visualized_frames = []
    for frame, mask in zip(camera_rgbs, processed_masks):
        vis_frame = frame.copy()  # RGB格式
        
        # 处理mask形状 - 如果是(1, H, W)格式，去掉第一个维度
        if len(mask.shape) == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        
        # 为不同的mask区域使用不同的颜色
        mask_area = (mask > 0.5)  # 使用阈值处理浮点mask
        
        # 创建彩色叠加
        overlay = vis_frame.copy()
        overlay[mask_area] = [0, 255, 0]  # 绿色叠加
        
        # 混合原始图像和mask
        vis_frame = cv2.addWeighted(vis_frame, 0.7, overlay, 0.3, 0)
        
        # 添加到预览列表
        visualized_frames.append(vis_frame)

    # 转换为numpy数组
    visualized_frames = np.array(visualized_frames)
    
    # 如果是反向追踪，预览时也倒放显示
    if reverse_tracking:
        visualized_frames = visualized_frames[::-1]
        print("预览视频已倒放以显示反向追踪效果")
    
    # 使用preview_video函数预览叠加结果
    title = "SAM2 Backward Tracking Preview" if reverse_tracking else "SAM2 Forward Tracking Preview"
    should_skip = preview_video(
        frames=visualized_frames,
        is_rgb=False,  # 使用RGB格式
        title=title,
        speed_multiplier=2
    )
    
    if should_skip:
        print("Skipped by user")
    else:
        print("Processing completed")
        
    # 保存追踪结果
    if reverse_tracking:
        print("反向追踪完成！")
        print(f"从最后一帧开始追踪，追踪了 {len(processed_masks)} 帧")
    else:
        print("正向追踪完成！")
        print(f"从第一帧开始追踪，追踪了 {len(processed_masks)} 帧")

    # 可选：也可以单独预览mask
    # should_skip_mask = preview_masks(
    #     masks=processed_masks,
    #     title="SAM2 Processed Masks Preview",
    #     speed_multiplier=2
    # )


