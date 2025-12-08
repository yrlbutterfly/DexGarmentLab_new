import argparse
import json
import os
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np


def load_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """读取 jsonl 标注文件，返回每一行的 dict 列表。"""
    data_list: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data_list.append(json.loads(line))
    return data_list


def save_jsonl(jsonl_path: str, data_list: List[Dict[str, Any]]) -> None:
    """把内存中的标注列表写回 jsonl 文件。"""
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def infer_keys_for_part(example_item: Dict[str, Any], part_name: str) -> List[str]:
    """
    根据给定的 part_name 推断应当编辑的键名列表。

    规则：
    - 如果 part_name 本身就是一个键（例如 'left_cuff'），则只返回这个键；
    - 否则，尝试匹配 'left_{part_name}' / 'right_{part_name}' / '{part_name}' 这几种形式；
    - 典型用法：
        part_name = 'cuff'   -> ['left_cuff', 'right_cuff']（如果存在）
        part_name = 'collar' -> ['left_collar', 'right_collar', 'center_collar']（如果存在）
    """
    keys: List[str] = []

    # 1) 直接是完整键名
    if part_name in example_item:
        return [part_name]

    # 2) 推断左右以及 center
    for prefix in ["left_", "right_", "center_"]:
        k = prefix + part_name
        if k in example_item:
            keys.append(k)

    # 3) 退而求其次，直接用 part_name
    if not keys and part_name in example_item:
        keys.append(part_name)

    return keys


def get_box_from_item(item: Dict[str, Any], key: str) -> Optional[Tuple[int, int, int, int]]:
    """从一条标注中取出指定 key 的 bbox。"""
    info = item.get(key)
    if not info or "bbox" not in info:
        return None
    bbox = info["bbox"]
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
    else:
        return None
    return int(x1), int(y1), int(x2), int(y2)


def point_in_box(x: int, y: int, box: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """返回 bbox 的中心点坐标。"""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def get_point_from_item(item: Dict[str, Any], key: str) -> Optional[Tuple[int, int]]:
    """从一条标注中取出指定 key 的 point。"""
    info = item.get(key)
    if not info or "point" not in info:
        return None
    pt = info["point"]
    if isinstance(pt, (list, tuple)) and len(pt) == 2:
        x, y = pt
    else:
        return None
    return int(x), int(y)


def point_to_bbox(
    x: int,
    y: int,
    half_size: int,
    img_w: Optional[int] = None,
    img_h: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    """
    以给定点 (x, y) 为中心生成一个紧凑的小 bbox。

    如果提供 img_w / img_h，则会自动裁剪到图像范围内。
    """
    x1 = x - half_size
    y1 = y - half_size
    x2 = x + half_size
    y2 = y + half_size

    if img_w is not None and img_h is not None:
        x1 = max(0, min(x1, img_w - 1))
        x2 = max(0, min(x2, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        y2 = max(0, min(y2, img_h - 1))

    return int(x1), int(y1), int(x2), int(y2)


class BBoxEditor:
    def __init__(
        self,
        jsonl_path: str,
        image_dir: str,
        part_names: List[str],
        start_index: int = 0,
    ) -> None:
        self.jsonl_path = jsonl_path
        self.image_dir = image_dir
        # 支持一次指定多个部位名称
        # 例如：["cuff", "collar"] 或 ["left_cuff", "right_collar"]
        self.part_names: List[str] = part_names
        # 这里的 start_index 代表“图片 id”，例如 700 -> 对应文件名中包含 image_700.xxx 的那一条
        self.start_index = start_index

        self.data_list = load_jsonl(jsonl_path)

        if not self.data_list:
            raise RuntimeError(f"jsonl 文件为空: {jsonl_path}")

        # 用第一条数据，根据所有 part_names 推断本次要编辑的 key 列表（去重后排序）
        example_item = self.data_list[0]
        target_keys_set = set()
        for part_name in self.part_names:
            keys = infer_keys_for_part(example_item, part_name)
            target_keys_set.update(keys)

        self.target_keys = sorted(target_keys_set)
        if not self.target_keys:
            raise RuntimeError(
                f"在 jsonl 中没有找到与 {self.part_names} 对应的任何键，请检查名称是否正确。"
            )

        print(f"将只显示/编辑以下字段的 bbox: {self.target_keys}")

        # 交互状态
        self.current_img: Optional[np.ndarray] = None
        self.current_draw_img: Optional[np.ndarray] = None
        self.drawing: bool = False
        self.start_point: Tuple[int, int] = (0, 0)
        self.end_point: Tuple[int, int] = (0, 0)
        self.current_label: Optional[str] = None
        self.modified: bool = False  # 当前图片是否有修改

        # 按钮相关状态
        self.selected_button_label: Optional[str] = None  # 当前选中的 label
        self.button_height = 40
        self.button_margin = 10
        self.button_boxes: Dict[str, Tuple[int, int, int, int]] = {}

        # 以 point 为中心生成小 bbox 的一半边长（像素）
        # 实际 bbox 大小约为 (2 * box_half_size) x (2 * box_half_size)
        # 可以按需要调小/调大，这里设成 6 -> 约 12x12 的小框
        self.box_half_size: int = 6

    def _resolve_start_index(self) -> int:
        """
        根据 self.start_index 作为“图片 id”在 jsonl 中找到实际起始下标。

        约定：
        - jsonl 的 'rgb' 字段中包含形如 'image_700.png' 的文件名；
        - 用户传入 --start-index 700 时，从第一条包含 'image_700' 的记录开始；
        - 若 start_index <= 0 或未找到匹配项，则从第 0 条开始。
        """
        # 非正数，直接从头开始
        if self.start_index <= 0:
            return 0

        target_id = int(self.start_index)
        target_pattern = f"image_{target_id}"

        for idx, item in enumerate(self.data_list):
            rgb_name = str(item.get("rgb", ""))
            if target_pattern in rgb_name:
                print(
                    f"根据图片 id {target_id} 找到起始样本：index={idx}, rgb='{rgb_name}'"
                )
                return idx

        print(
            f"警告：未在 jsonl 中找到包含 '{target_pattern}' 的图片，"
            f"将从第 0 条开始。"
        )
        return 0

    # ---------------- 按钮与 label 选择逻辑 ----------------

    @staticmethod
    def _short_label(key: str) -> str:
        """
        将完整的 key 转成更简短、但可区分左右/中等信息的标记。
        例如：left_cuff -> L，right_cuff -> R，center_collar -> C。
        其他情况就直接返回原 key。
        """
        if key.startswith("left_"):
            return "L"
        if key.startswith("right_"):
            return "R"
        if key.startswith("center_"):
            return "C"
        return key

    def _build_button_boxes(self, img_width: int) -> None:
        """
        根据当前 target_keys 构造顶部按钮的矩形区域。
        简单地从左到右排布。
        """
        self.button_boxes = {}
        x = self.button_margin
        y1 = 0
        y2 = self.button_height

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        for label in self.target_keys:
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            btn_w = text_w + 2 * self.button_margin
            x2 = x + btn_w

            # 若超过图像宽度，仍然画，只是可能被截断；一般 target_keys 数量很少不成问题
            self.button_boxes[label] = (x, y1, x2, y2)
            x = x2 + self.button_margin

    def _draw_buttons(self, img: np.ndarray) -> None:
        """在图像顶部画出所有 label 的按钮。"""
        if not self.button_boxes:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        for label, (x1, y1, x2, y2) in self.button_boxes.items():
            # 选中按钮用不同颜色
            if label == self.selected_button_label:
                bg_color = (0, 165, 255)  # 橙色
                text_color = (255, 255, 255)
            else:
                bg_color = (60, 60, 60)
                text_color = (255, 255, 255)

            cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)

            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            text_x = x1 + self.button_margin
            text_y = y1 + (y2 - y1 + text_h) // 2

            cv2.putText(
                img,
                label,
                (text_x, text_y),
                font,
                font_scale,
                text_color,
                thickness,
            )

    def _choose_label_for_new_box(
        self,
        item: Dict[str, Any],
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> Optional[str]:
        """
        决定新框应该对应哪个 label：
        1. 若用户通过按钮已明确选中 label，则直接使用；
        2. 否则，在所有已有框中找到中心点距离新框中心最近的那个；
        3. 若一个都没有，则退回到 target_keys[0]。
        """
        # 情况 1：按钮已有选择
        if self.selected_button_label is not None:
            return self.selected_button_label

        # 情况 2：自动找最近的旧框
        cx_new = (x1 + x2) / 2.0
        cy_new = (y1 + y2) / 2.0

        best_label: Optional[str] = None
        best_dist2: Optional[float] = None

        for key in self.target_keys:
            box = get_box_from_item(item, key)
            if not box:
                continue
            cx_old, cy_old = box_center(box)
            dx = cx_new - cx_old
            dy = cy_new - cy_old
            dist2 = dx * dx + dy * dy
            if best_dist2 is None or dist2 < best_dist2:
                best_dist2 = dist2
                best_label = key

        # 情况 3：若没有任何旧框，退回第一个 key
        if best_label is None and self.target_keys:
            best_label = self.target_keys[0]

        return best_label

    def _mouse_callback(self, event, x, y, flags, param):
        if self.current_img is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            item = param  # 当前这条 json dict

            # 先看是否点在顶部按钮区域
            for label, (bx1, by1, bx2, by2) in self.button_boxes.items():
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    # 点击按钮：切换当前选中的 label，不开始画框
                    self.selected_button_label = label
                    if self.current_img is not None:
                        self.current_draw_img = self.current_img.copy()
                        self._draw_existing_boxes(self.current_draw_img, item)
                        cv2.imshow("bbox_editor", self.current_draw_img)
                    return

            # 否则，认为是在图像上点击，用于修改 / 新建 point
            h, w = self.current_img.shape[:2]

            # 将当前点击附近当作一个“小框”，用原有逻辑推断应当对应的 label
            half = self.box_half_size
            x1 = max(0, x - half)
            y1 = max(0, y - half)
            x2 = min(w - 1, x + half)
            y2 = min(h - 1, y + half)

            label = self._choose_label_for_new_box(item, x1, y1, x2, y2)
            if label is None:
                return

            if label not in item:
                item[label] = {}

            # 更新 point
            item[label]["point"] = [int(x), int(y)]

            # 以 point 为中心生成紧凑 bbox，并更新到 json
            bx1, by1, bx2, by2 = point_to_bbox(
                int(x), int(y), self.box_half_size, img_w=w, img_h=h
            )
            item[label]["bbox"] = [bx1, by1, bx2, by2]

            self.modified = True

            # 重新绘制整张图
            if self.current_img is not None:
                self.current_draw_img = self.current_img.copy()
                self._draw_existing_boxes(self.current_draw_img, item)
                cv2.imshow("bbox_editor", self.current_draw_img)

    def _normalize_item_boxes(self, item: Dict[str, Any], img: np.ndarray) -> bool:
        """
        将当前 item 中 target_keys 对应的 bbox 统一规范成以 point 为中心的
        固定大小小框（self.box_half_size 决定大小），并保持中心不变。

        返回值：
            是否对该 item 做了实际修改（用于决定是否需要保存）。
        """
        changed = False
        h, w = img.shape[:2]

        for key in self.target_keys:
            info = item.get(key)
            if not isinstance(info, dict):
                continue

            # 优先使用已有 point；如果没有 point，就用 bbox 中心作为 point
            pt = get_point_from_item(item, key)
            if pt is not None:
                px, py = pt
            else:
                box = get_box_from_item(item, key)
                if not box:
                    continue
                cx, cy = box_center(box)
                px, py = int(cx), int(cy)

            # 生成规范化 bbox
            bx1, by1, bx2, by2 = point_to_bbox(
                px, py, self.box_half_size, img_w=w, img_h=h
            )

            old_box = info.get("bbox")
            old_point = info.get("point")

            new_box = [int(bx1), int(by1), int(bx2), int(by2)]
            new_point = [int(px), int(py)]

            # 只有当 bbox 或 point 实际发生变化时才视为修改
            if old_box != new_box or old_point != new_point:
                info["bbox"] = new_box
                info["point"] = new_point
                item[key] = info
                changed = True

        return changed

    def _draw_existing_boxes(self, img: np.ndarray, item: Dict[str, Any]) -> None:
        """只画当前 target_keys 的 bbox，并在顶部画按钮。"""
        for key in self.target_keys:
            # 以 point 为主进行可视化，并由 point 生成一个小 bbox
            pt = get_point_from_item(item, key)
            draw_box: Optional[Tuple[int, int, int, int]] = None
            color = (0, 255, 0)

            h, w = img.shape[:2]

            if pt is not None:
                px, py = pt
                # 画点：半径更小但边界更清晰（细描边）
                cv2.circle(img, (px, py), 2, color, -1)
                cv2.circle(img, (px, py), 3, (0, 0, 0), 1)  # 黑色描边，提升对比度
                # 基于 point 生成紧凑 bbox（仅用于显示）
                draw_box = point_to_bbox(px, py, self.box_half_size, img_w=w, img_h=h)
            else:
                # 若没有 point，则退回使用已有 bbox，并在其中心画点
                box = get_box_from_item(item, key)
                if box:
                    x1, y1, x2, y2 = box
                    cx, cy = box_center(box)
                    cv2.circle(img, (int(cx), int(cy)), 2, color, -1)
                    cv2.circle(img, (int(cx), int(cy)), 3, (0, 0, 0), 1)
                    # 显示时仍用原 bbox，避免在未操作时就改变数据语义
                    draw_box = (x1, y1, x2, y2)

            if draw_box is None:
                continue

            x1, y1, x2, y2 = draw_box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            label_text = self._short_label(key)
            cv2.putText(
                img,
                label_text,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        # 顶部按钮栏（画在最上层）
        self._draw_buttons(img)

        # 画一些简单的操作提示
        h, w = img.shape[:2]
        cv2.putText(
            img,
            "Click button; click image to move point; Enter: next; Del/d: delete; q: quit",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )

    def run(self) -> None:
        cv2.namedWindow("bbox_editor", cv2.WINDOW_NORMAL)

        # 将 start_index 解释为“图片 id”，先解析成实际的列表下标
        idx = self._resolve_start_index()

        while 0 <= idx < len(self.data_list):
            num_items = len(self.data_list)
            item = self.data_list[idx]
            rgb_name = item.get("rgb")
            if not rgb_name:
                print(f"[{idx}] 缺少 'rgb' 字段，跳过。")
                idx += 1
                continue

            img_path = os.path.join(self.image_dir, rgb_name)
            if not os.path.exists(img_path):
                print(f"[{idx}] 图片不存在: {img_path}，跳过。")
                idx += 1
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"[{idx}] 无法读取图片: {img_path}，跳过。")
                idx += 1
                continue
            self.current_img = img
            self.current_draw_img = img.copy()
            self.modified = False
            # 每张图片重置按钮选择（也可以改成保留上一张的选择）
            self.selected_button_label = None

            # 根据当前图像宽度构建按钮布局
            h, w = self.current_img.shape[:2]
            self._build_button_boxes(w)

            # 进入该图片时，自动把当前 target_keys 的 bbox 规范成以 point 为中心的固定小框
            # （中心保持不变），这样即使不点击，只按 Enter 也会把 bbox 统一成 12x12。
            normalized = self._normalize_item_boxes(item, self.current_img)
            if normalized:
                self.modified = True

            # 初始显示：只画当前部位相关的 box
            self._draw_existing_boxes(self.current_draw_img, item)
            cv2.imshow("bbox_editor", self.current_draw_img)

            # 设置鼠标回调，param 传当前 item
            cv2.setMouseCallback("bbox_editor", self._mouse_callback, param=item)

            deleted = False  # 是否删除了当前这条数据

            print(
                f"[{idx}/{num_items-1}] {rgb_name} - 略过请直接按 Enter，"
                f"鼠标点击图像可修改 {self.target_keys} 的 point（并自动更新小 bbox），按 Del 或 d 删除，按 q 退出。"
            )

            while True:
                key = cv2.waitKey(20) & 0xFF

                # 回车 (Enter) -> 这张图片结束，保存修改（如果有）并进入下一张
                if key in (13, 10):  # 兼容不同系统的 Enter
                    if self.modified:
                        print(f"  已修改：{rgb_name}")
                    else:
                        print(f"  未修改：{rgb_name}")
                    break

                # q 退出整个程序
                if key in (ord("q"), ord("Q")):
                    print("检测到 q，提前退出编辑。")
                    cv2.destroyWindow("bbox_editor")
                    # 退出前保存到文件
                    print("正在保存 jsonl 修改...")
                    save_jsonl(self.jsonl_path, self.data_list)
                    print(f"已保存到: {self.jsonl_path}")
                    return

                # Delete 键或 d/D：删除当前图片及对应 jsonl 记录
                if key in (127, ord("d"), ord("D")):
                    print(f"  删除当前图片及标注：{rgb_name}")
                    # 删除图片文件（若存在）
                    if os.path.exists(img_path):
                        try:
                            os.remove(img_path)
                        except OSError as e:
                            print(f"  删除图片文件失败: {e}")

                    # 从数据列表中删除当前记录
                    del self.data_list[idx]
                    deleted = True
                    break

            if deleted:
                # 不自增 idx，因为当前 idx 位置已变成下一条数据
                continue

            # 下一张
            idx += 1

        cv2.destroyWindow("bbox_editor")

        # 全部处理结束后统一保存
        print("所有图片处理完毕，正在保存 jsonl 修改...")
        save_jsonl(self.jsonl_path, self.data_list)
        print(f"已保存到: {self.jsonl_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "交互式 bbox 可视化与修改工具："
            "只针对指定部位（例如 cuff / collar / hem / left_cuff 等）显示与编辑，"
            "若存在左右同名位置（left_xxx / right_xxx），则会一起显示在同一张图上。"
        )
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        required=False,
        default="./Preprocess/data/stage3_1207/garments_box.jsonl",
        help="garments_box.jsonl 的完整路径",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=False,
        default="./Preprocess/data/stage3_1207/images",
        help="对应 RGB 图片所在目录（例如 .../Preprocess/data/cloth_origin_new/neat_1127/images）",
    )
    parser.add_argument(
        "--part",
        type=str,
        nargs="+",
        required=True,
        help=(
            "衣服部位名称，可以一次指定一个或多个。"
            "若为 'cuff' / 'collar' / 'hem' / 'armpit' / 'shoulder' / 'waist' 等，"
            "会自动匹配 left_/right_/center_ 前缀；"
            "也可以直接写完整键名，如 'left_cuff'。"
        ),
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help=(
            "从指定图片 id 开始标注（例如 700 -> 从文件名中包含 image_700 的图片开始；"
            "若为 0 或未找到，则从第 0 条开始）"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    editor = BBoxEditor(
        jsonl_path=args.jsonl,
        image_dir=args.image_dir,
        part_names=args.part,
        start_index=args.start_index,
    )
    editor.run()


