import os
import cv2


def images_to_video(
    images_dir: str,
    output_path: str,
    fps: int = 1,
    font_scale: float = 1.0,
    thickness: int = 2,
) -> None:
    # 收集并排序所有 PNG 图片
    files = [f for f in os.listdir(images_dir) if f.lower().endswith(".png")]
    if not files:
        raise RuntimeError(f"目录中没有找到 PNG 图片: {images_dir}")

    files.sort()  # 按文件名排序，保证 001, 002, ... 的顺序

    first_img_path = os.path.join(images_dir, files[0])
    frame = cv2.imread(first_img_path)
    if frame is None:
        raise RuntimeError(f"无法读取图片: {first_img_path}")

    height, width = frame.shape[:2]

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    margin_x, margin_y = 10, 30

    for name in files:
        img_path = os.path.join(images_dir, name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图片，跳过: {img_path}")
            continue

        # 在左上角写上文件名
        cv2.putText(
            img,
            name,
            (margin_x, margin_y),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA,
        )

        writer.write(img)

    writer.release()
    print(f"视频已保存到: {output_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images_dir = os.path.join(
        base_dir, "Preprocess", "data", "tops_preview_once", "images"
    )

    # 播放不要太快：这里设置为 1 FPS，每张图片展示约 1 秒
    output_video = os.path.join(
        base_dir, "Preprocess", "data", "tops_preview_once", "tops_preview_once.mp4"
    )

    images_to_video(
        images_dir=images_dir,
        output_path=output_video,
        fps=1,
        font_scale=1.0,
        thickness=2,
    )



