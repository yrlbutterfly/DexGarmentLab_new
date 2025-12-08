from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import os
import sys
import time
import numpy as np
import cv2

# 加载自定义包
sys.path.append(os.getcwd())
from Env_StandAlone.BaseEnv import BaseEnv
from Env_Config.Garment.Particle_Garment import Particle_Garment
from Env_Config.Camera.Recording_Camera import Recording_Camera
from Env_Config.Room.Real_Ground import Real_Ground
from Env_Config.Utils_Project.Code_Tools import get_unique_filename


class GarmentPreviewEnv(BaseEnv):
    """用于单张预览衣服外观的简化环境"""

    def __init__(self):
        super().__init__()

        # 相机，只用于拍 RGB 图
        self.garment_camera = Recording_Camera(
            camera_position=np.array([0.0, 0.8, 6.0]),
            camera_orientation=np.array([0.0, 90.0, 90.0]),
            prim_path="/World/garment_camera",
        )

        self.garment = None
        self.ground = None

    def cleanup(self):
        """清理上一件衣服和地面"""
        from Env_Config.Room.Object_Tools import delete_prim_group

        # 删除衣服
        if hasattr(self, "garment") and self.garment is not None:
            try:
                delete_prim_group([self.garment.garment_prim_path])
                delete_prim_group([self.garment.particle_system_path])
                delete_prim_group([self.garment.particle_material_path])
            except Exception:
                pass
            self.garment = None

        # 删除地面
        if hasattr(self, "ground") and self.ground is not None:
            try:
                delete_prim_group(["/World/defaultGroundPlane"])
            except Exception:
                pass
            self.ground = None

    def apply(self, pos: np.ndarray, ori: np.ndarray, ground_material_usd: str, usd_path: str):
        """加载一件衣服到场景中"""
        self.cleanup()

        # 地面
        self.ground = Real_Ground(
            self.scene,
            visual_material_usd=ground_material_usd,
        )

        # 衣服（粒子布料）
        self.garment = Particle_Garment(
            self.world,
            pos=np.array([0.0, 3.0, 0.6]),
            ori=np.array([0.0, 0.0, 0.0]),
            usd_path=usd_path,
            contact_offset=0.012,
            rest_offset=0.010,
            particle_contact_offset=0.012,
            fluid_rest_offset=0.010,
            solid_rest_offset=0.010,
            visual_material_usd=np.random.choice(
                [
                    "Assets/Material/Garment/linen_Pumpkin.usd",
                    "Assets/Material/Garment/linen_Blue.usd",
                ]
            ),
        )

        # 重置世界
        self.reset()

        # 把衣服放到目标位置
        self.garment.set_pose(pos=np.array([pos[0], pos[1], 0.2]), ori=ori)

        # 初始化相机（开启分割是为了保持接口一致，主要还是用 RGB）
        self.garment_camera.initialize(
            segment_pc_enable=True,
            segment_prim_path_list=["/World/Garment/garment"],
            camera_params_enable=True,
        )

        # 让仿真跑一会儿，使衣服稳定
        for _ in range(150):
            self.step()


def capture_garment_preview(env: GarmentPreviewEnv, usd_path: str, save_dir: str, index: int):
    """加载一件衣服并保存一张预览图"""
    # 固定一个大致合适的摆放位置
    pos = np.array([0.0, 0.8, 0.2])
    ori = np.array([0.0, 0.0, 0.0])

    # 随机选一个地面材质（可选）
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    floors_lists = os.path.join(base_dir, "Preprocess/floors_list.txt")
    floors_list = []
    if os.path.exists(floors_lists):
        with open(floors_lists, "r", encoding="utf-8") as f:
            for line in f:
                clean_line = line.rstrip("\n")
                if clean_line:
                    floors_list.append(clean_line)
    ground_material_usd = np.random.choice(floors_list) if floors_list else None

    # 加载衣服
    env.apply(pos, ori, ground_material_usd, usd_path)

    # 再多跑几步，保证状态稳定
    for _ in range(50):
        env.step()

    # 拍一张 RGB 图
    rgb = env.garment_camera.get_rgb_graph(
        save_or_not=False,
        save_path=get_unique_filename("data", extension=".png"),
    )

    # 根据 usd 路径生成可读的图片名
    usd_name = os.path.splitext(os.path.basename(usd_path))[0]  # 例如 TCLC_Jacket057_obj
    img_name = f"{index:03d}_{usd_name}.png"
    save_path = os.path.join(save_dir, img_name)

    cv2.imwrite(save_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"Saved preview image: {save_path}")


if __name__ == "__main__":
    # ===================================================================== #
    project_name = "tops_preview_once"
    # ===================================================================== #

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 输出目录：Preprocess/data/tops_preview_once/images
    output_root = os.path.join(base_dir, "Preprocess", "data", project_name)
    images_dir = os.path.join(output_root, "images")
    os.makedirs(images_dir, exist_ok=True)

    # 读取衣服列表
    assets_lists = os.path.join(
        base_dir, "Model_HALO/GAM/checkpoints/Tops_LongSleeve/assets_list.txt"
    )
    assets_list = []
    with open(assets_lists, "r", encoding="utf-8") as f:
        for line in f:
            clean_line = line.rstrip("\n")
            if clean_line:
                assets_list.append(clean_line)

    print(f"Total garments in list: {len(assets_list)}")

    # 创建环境
    env = GarmentPreviewEnv()

    # 逐件衣服生成一张预览图
    for idx, usd_path in enumerate(assets_list):
        print(f"[{idx + 1}/{len(assets_list)}] Preview garment: {usd_path}")
        np.random.seed(int(time.time()))
        capture_garment_preview(env, usd_path, images_dir, idx + 1)

    simulation_app.close()


