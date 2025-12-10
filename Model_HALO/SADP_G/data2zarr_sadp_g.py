import argparse
import json
import os
import shutil

import numpy as np
import zarr
from PIL import Image
from tqdm import tqdm
import copy
import open3d as o3d
import pickle


def main():
    """
    主函数：将多 episode 的 npz 轨迹数据，打包转换成一个 zarr 数据集，
    方便后续 diffusion policy / RL 等模型高效随机访问和按 batch 读取。
    """

    # ---------- 命令行参数解析 ----------
    parser = argparse.ArgumentParser(
        description="Convert data to zarr format for diffusion policy",
    )
    parser.add_argument(
        "task_name",
        type=str,
        default="Fold_Dress",
        help="任务名称 (例如: Fold_Dress, Fold_Tops)",
    )
    parser.add_argument(
        "stage_index",
        type=str,
        default="1",
        help=(
            "当前流水线的阶段索引："
            "可以是单个数字 (例如: '1')，"
            "也可以是逗号分隔的多个阶段 (例如: '1,2,3' 表示合并 3 个 stage 到同一个 zarr 中)"
        ),
    )
    parser.add_argument(
        "train_data_num",
        type=int,
        default=200,
        help="要处理的 episode 数量 (例如: 200, 表示 data_0 ~ data_199.npz)",
    )
    args = parser.parse_args()

    # 从命令行参数中取值
    task_name = args.task_name
    stage_index_str = str(args.stage_index)
    # 支持 "1" 或 "1,2,3" 这种写法；统一解析成整数列表
    stage_indices = [int(s) for s in stage_index_str.split(",")]
    train_data_num = args.train_data_num

    # ---------- 路径设置 ----------
    # 当前脚本的绝对路径
    current_abs_dir = os.path.dirname(os.path.abspath(__file__))
    # 工程根目录 (假设当前脚本在 Model_HALO/SADP_G/ 之下)
    parent_dir = os.path.dirname(os.path.dirname(current_abs_dir))
    print("Project Root Dir : ", parent_dir)

    # 读取的 npz 轨迹数据目录: <工程根>/Data/<task_name>/train_data
    load_dir = parent_dir + f"/Data/{task_name}/train_data"
    print("Meta Data Load Dir : ", load_dir)

    # 输出的 zarr 文件路径 (保存在当前工作目录下的 data/ 目录中)
    # 单 stage:  例如 data/Fold_Tops_stage_1_200.zarr
    # 多 stage:  例如 data/Fold_Tops_stage_1,2,3_200.zarr
    save_dir = f"data/{task_name}_stage_{stage_index_str}_{train_data_num}.zarr"
    # 若已存在同名 zarr 目录，则先删除，确保是干净的输出
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    print("Save Dir : ", save_dir)

    # ---------- 初始化 zarr 根结点与子 group ----------
    # zarr.group(path) 会在 path 处创建/打开一个 group (目录形式)
    zarr_root = zarr.group(save_dir)
    # data group 用来存放真正的训练数据
    zarr_data = zarr_root.create_group("data")
    # meta group 用来存放元信息 (例如 episode 切分)
    zarr_meta = zarr_root.create_group("meta")

    # zarr 压缩器配置，Blosc + zstd，适中压缩等级
    # 注意：真正的 dataset 是在第一次写入 batch 时，根据数据形状动态创建的
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    # ---------- 批处理配置 ----------
    # 每处理 batch_size 个 episode，就把当前缓存的数据一次性写入 zarr
    batch_size = 100

    # 以下是用来在内存里缓存一个 batch 的数据的 list
    environment_point_cloud_arrays = []  # 环境点云 (2048, 6)
    garment_point_cloud_arrays = []      # 衣物点云 (2048, 3)
    # object_point_cloud_arrays = []     # 如未来需要物体点云可打开
    points_affordance_feature_arrays = []  # 点的 affordance 特征
    state_arrays = []                   # 机器人关节状态 (当前时刻)
    action_arrays = []                  # 机器人关节状态 (下一时刻，作为 action 标签)
    episode_ends_arrays = []            # 每个 episode 结束时，累积的全局 step 索引

    total_count = 0      # 到目前为止，所有 episode 中累积的 step 数量
    current_batch = 0    # 已经写入到 zarr 的 batch 计数

    # ---------- 逐 episode / 逐 stage 读取 npz 并展开为 step-level 数据 ----------
    for current_ep in tqdm(
        range(train_data_num),
        desc=f"Processing {train_data_num} MetaData",
    ):
        # 打开形如 data_0.npz, data_1.npz, ...
        data = np.load(load_dir + f"/data_{current_ep}.npz", allow_pickle=True)

        # 支持一个 episode 中包含多个 stage_x：
        # 对于每个 stage_i，都视为一个独立的 episode，依次展开并写入同一个 zarr，
        # 这样就可以在训练时一起使用 1/2/3 stage 的轨迹来训练一个 checkpoint。
        for stage_idx in stage_indices:
            # 取出对应阶段的数据，meta_data 通常是一个长度为 T 的列表/数组，元素为 dict
            # key 形如 'stage_1', 'stage_2', ...
            meta_key = f"stage_{stage_idx}"
            if meta_key not in data:
                # 若某个 stage 在当前 npz 中不存在，则跳过
                print(f"[Warning] {load_dir}/data_{current_ep}.npz 中不存在键 {meta_key}，跳过该 stage。")
                continue

            meta_data = data[meta_key]
            data_length = len(meta_data)  # 当前 stage 的时间步长 T

            # 这里只遍历到 data_length-1，是因为 action 使用下一时刻的 joint_state
            for i in range(data_length - 1):
                # ---------- 环境点云 ----------
                # env_point_cloud: (2048, 6) = (x, y, z, r, g, b) 或类似结构
                assert meta_data[i]["env_point_cloud"].shape == (2048, 6)
                environment_point_cloud_arrays.append(meta_data[i]["env_point_cloud"])

                # ---------- 衣物点云 ----------
                # garment_point_cloud: (2048, 3) = (x, y, z)
                assert meta_data[i]["garment_point_cloud"].shape == (2048, 3)
                garment_point_cloud_arrays.append(meta_data[i]["garment_point_cloud"])

                # ---------- 物体点云（当前脚本未使用，如有需要可解开） ----------
                # assert meta_data[i]['object_point_cloud'].shape == (2048, 3)
                # object_point_cloud_arrays.append(meta_data[i]['object_point_cloud'])

                # ---------- 点的 affordance 特征 ----------
                # points_affordance_feature 的具体维度依赖于上游特征提取
                # 这里不限制具体维度，可以是 2 维 (左右手 affordance)
                # 也可以是 4 维 (左右手 affordance + 左右手 goal) 等
                points_affordance_feature_arrays.append(
                    meta_data[i]["points_affordance_feature"]
                )

                # ---------- 状态与动作 ----------
                # state 使用当前时刻的关节状态
                state_arrays.append(meta_data[i]["joint_state"])
                # action 使用下一时刻的关节状态，可视为 "下一步状态" 或 delta 的 proxy
                action_arrays.append(meta_data[i + 1]["joint_state"])

                # 全局 step 计数 +1
                total_count += 1

            # 一个 stage episode 结束时，记录此刻的 total_count
            # 例如：第 0 个 npz 的 stage_1 有 50 步，则记录 50；
            #       同一个 npz 的 stage_2 又有 40 步，则记录 90；以此类推。
            episode_ends_arrays.append(copy.deepcopy(total_count))

        # ---------- 条件触发写入 zarr ----------
        # 当:
        #   1) 当前 episode 数达到了 batch_size 的整数倍，或者
        #   2) 已经是最后一个 episode
        # 即把当前 batch 缓存的数据一次性写入 zarr，避免频繁 IO。
        if (current_ep + 1) % batch_size == 0 or (current_ep + 1) == train_data_num:
            # 将 list 转为 numpy 数组，方便后续统一写入
            environment_point_cloud_arrays = np.array(environment_point_cloud_arrays)
            garment_point_cloud_arrays = np.array(garment_point_cloud_arrays)
            # object_point_cloud_arrays = np.array(object_point_cloud_arrays)
            points_affordance_feature_arrays = np.array(
                points_affordance_feature_arrays
            )
            action_arrays = np.array(action_arrays)
            state_arrays = np.array(state_arrays)
            episode_ends_arrays = np.array(episode_ends_arrays)

            # 打印当前 batch 的数据形状，便于调试和 sanity check
            print(
                "environment_point_cloud_arrays shape : ",
                environment_point_cloud_arrays.shape,
            )
            print(
                "garment_point_cloud_arrays shape : ",
                garment_point_cloud_arrays.shape,
            )
            # print("object_point_cloud_arrays shape : ", object_point_cloud_arrays.shape)
            print(
                "points_affordance_feature_arrays shape : ",
                points_affordance_feature_arrays.shape,
            )
            print("state_arrays shape : ", state_arrays.shape)
            print("action_arrays shape : ", action_arrays.shape)

            # ---------- 首次写入时，根据真实数据形状动态创建 zarr dataset ----------
            if current_batch == 0:
                # 环境点云数据集
                zarr_data.create_dataset(
                    "environment_point_cloud",
                    # 首维为 0，表示初始样本数为 0，后续通过 append 动态增长
                    shape=(0, *environment_point_cloud_arrays.shape[1:]),
                    # chunk 以 batch_size 为单位，便于按 batch 读取训练
                    chunks=(batch_size, *environment_point_cloud_arrays.shape[1:]),
                    dtype=environment_point_cloud_arrays.dtype,
                    compressor=compressor,
                    overwrite=True,
                )

                # 衣物点云数据集
                zarr_data.create_dataset(
                    "garment_point_cloud",
                    shape=(0, *garment_point_cloud_arrays.shape[1:]),
                    chunks=(batch_size, *garment_point_cloud_arrays.shape[1:]),
                    dtype=garment_point_cloud_arrays.dtype,
                    compressor=compressor,
                    overwrite=True,
                )

                # 如需写入 object_point_cloud，可以参考下面的注释代码
                # zarr_data.create_dataset(
                #     "object_point_cloud",
                #     shape=(0, *object_point_cloud_arrays.shape[1:]),
                #     chunks=(batch_size, *object_point_cloud_arrays.shape[1:]),
                #     dtype=object_point_cloud_arrays.dtype,
                #     compressor=compressor,
                #     overwrite=True,
                # )

                # affordance 特征数据集
                zarr_data.create_dataset(
                    "points_affordance_feature",
                    shape=(0, *points_affordance_feature_arrays.shape[1:]),
                    chunks=(batch_size, *points_affordance_feature_arrays.shape[1:]),
                    dtype=points_affordance_feature_arrays.dtype,
                    compressor=compressor,
                    overwrite=True,
                )

                # 状态数据集 (joint_state)
                zarr_data.create_dataset(
                    "state",
                    shape=(0, state_arrays.shape[1]),
                    chunks=(batch_size, state_arrays.shape[1]),
                    dtype="float32",
                    compressor=compressor,
                    overwrite=True,
                )

                # 动作数据集 (下一时刻 joint_state)
                zarr_data.create_dataset(
                    "action",
                    shape=(0, action_arrays.shape[1]),
                    chunks=(batch_size, action_arrays.shape[1]),
                    dtype="float32",
                    compressor=compressor,
                    overwrite=True,
                )

                # episode 结束索引数据集
                # 形状为 (0, )，一维数组，长度等于 episode 数
                zarr_meta.create_dataset(
                    "episode_ends",
                    shape=(0,),
                    chunks=(batch_size,),
                    dtype="int64",
                    compressor=compressor,
                    overwrite=True,
                )

            # ---------- 将当前 batch 的数据 append 进 zarr ----------
            zarr_data["environment_point_cloud"].append(environment_point_cloud_arrays)
            zarr_data["garment_point_cloud"].append(garment_point_cloud_arrays)
            # zarr_data["object_point_cloud"].append(object_point_cloud_arrays)
            zarr_data["points_affordance_feature"].append(
                points_affordance_feature_arrays
            )
            zarr_data["state"].append(state_arrays)
            zarr_data["action"].append(action_arrays)
            zarr_meta["episode_ends"].append(episode_ends_arrays)

            print(
                f"Batch {current_batch + 1} written with {len(garment_point_cloud_arrays)} samples."
            )

            # ---------- 清空当前 batch 的缓存，准备下一批 ----------
            environment_point_cloud_arrays = []
            garment_point_cloud_arrays = []
            # object_point_cloud_arrays = []
            points_affordance_feature_arrays = []
            action_arrays = []
            state_arrays = []
            episode_ends_arrays = []

            current_batch += 1  # 已写入 batch 数 +1


if __name__ == "__main__":
    # 当此脚本作为主程序运行时，执行 main()
    main()