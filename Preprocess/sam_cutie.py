import cv2

from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
import torch
import gc
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

def get_sam_mask(first_frame, mask_camera_name='1viewer'):
    sam_checkpoint = '/home/admin01/Models/sam/sam_vit_h_4b8939.pth' # /home/admin01/Models/sam/sam_vit_h_4b8939.pth
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    # Set the image first
    predictor.set_image(first_frame)

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

        # 生成mask
        masks, scores, logits = predictor.predict(
            box=bbox,
            multimask_output=True
        )
        best_mask = masks[np.argmax(scores)]

        # 显示预览
        preview = first_frame.copy() 
        # preview = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR) # frame is rgb
        # preview = cv2.cvtColor(preview) # BGR
        mask_overlay = preview.copy()
        mask_overlay[best_mask] = [0, 255, 0]
        preview = cv2.addWeighted(preview, 0.7, mask_overlay, 0.3, 0)

        window_name = "Confirm Mask"
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
                # continue
                retry = True  # 设置重试标志
                break
            elif key == 13:  # Enter
                cv2.destroyWindow(window_name)
                break

        if retry:
            continue  # 重试
        break
    # np.save('best_mask.npy', best_mask)
    return best_mask


if __name__ == '__main__':
    cutie = get_default_model()
    processor = InferenceCore(cutie, cfg=cutie.cfg)
    processor.max_internal_size = -1

    camera_rgbs = []
    cap = cv2.VideoCapture('/home/admin01/Projects/DexGarmentLab/Data/Fold_Dress/vedio/vedio_0.mp4')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        camera_rgbs.append(frame)
    cap.release()
    camera_rgbs = np.array(camera_rgbs)
    print(camera_rgbs.shape)
    camera_rgbs = camera_rgbs[10:]
    camera_rgbs = camera_rgbs[::-1]

    first_frame = camera_rgbs[0]
    best_mask = get_sam_mask(first_frame)

    mask_uint8 = best_mask.astype(np.uint8) * 255
    cv2.imshow('best_mask', mask_uint8)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    mask = torch.from_numpy(best_mask.astype('uint8')).cuda()
    objects = np.unique(best_mask.astype('uint8'))
    objects = objects[objects != 0].tolist()

    processed_masks = []
    with torch.no_grad():
        for ti, frame in enumerate(camera_rgbs):
            image_tensor = to_tensor(frame).cuda().float()

            if ti == 0:
                output_prob = processor.step(image_tensor, mask, objects=objects)
            else:
                output_prob = processor.step(image_tensor)

            # convert output probabilities to an object mask
            tracked_mask = processor.output_prob_to_mask(output_prob)
            tracked_mask_np = tracked_mask.cpu().numpy().astype(np.uint8)

            # 合并追踪的mask和静态mask2
            combined_mask = np.zeros_like(tracked_mask_np, dtype=np.uint8)
            combined_mask[tracked_mask_np == 1] = 1  # tracked_mask的区域设为1

            processed_masks.append(combined_mask)
    
    processed_masks = np.array(processed_masks)

    visualized_frames = []
    for frame, mask in zip(camera_rgbs, processed_masks):
        vis_frame = frame.copy()  # RGB格式
        
        # 为不同的mask区域使用不同的颜色
        mask1_area = (mask == 1)
        
        # 创建彩色叠加
        overlay = vis_frame.copy()
        overlay[mask1_area] = [0, 255, 0]  # 第一个mask用蓝色
        
        # 混合原始图像和mask
        vis_frame = cv2.addWeighted(vis_frame, 0.7, overlay, 0.3, 0)
        
        # 添加到预览列表
        visualized_frames.append(vis_frame)

    # 转换为numpy数组
    visualized_frames = np.array(visualized_frames)
    print(visualized_frames.shape)

    # 使用现有的preview_video函数预览
    should_skip = preview_video(
        frames=visualized_frames,
        is_rgb=False,  # 因为我们使用的是RGB格式
        title="Processed Masks Preview",
        speed_multiplier=2
    )
    

