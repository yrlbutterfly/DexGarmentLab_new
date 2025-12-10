#!/bin/bash

# --------------------------------------------- #
# Fold_Tops_Collect.sh
# 对 Tops_LongSleeve 训练集中的每件衣服各跑 5 次
# 一共 100 条轨迹，每条轨迹单独启动 / 关闭一次 Isaac Sim
# --------------------------------------------- #

# 可选参数：
#   $1: ground_material_usd（地面材质的 usd 路径，可为空）

set -e

TASK_NAME="Fold_Tops"

# 配置 Isaac Sim python.sh 路径（按需修改）
ISAAC_PATH=~/isaacsim_4.5.0/python.sh
export ISAAC_PATH

# 训练集列表文件
ASSETS_FILE="Model_HALO/GAM/checkpoints/Tops_LongSleeve/assets_train.txt"

# 每件衣服重复次数
REPEAT_PER_GARMENT=5

# 可选地面材质 usd 路径
GROUND_MATERIAL_USD="$1"

# 统计进度
TOTAL_GARMENTS=$(grep -cve '^\s*$' "$ASSETS_FILE")
TOTAL_TRAJ=$((TOTAL_GARMENTS * REPEAT_PER_GARMENT))
CURRENT_TRAJ=0

echo "==============================================="
echo " Task: $TASK_NAME data collection"
echo " Assets file: $ASSETS_FILE"
echo " Garments: $TOTAL_GARMENTS, repeat: $REPEAT_PER_GARMENT, total traj: $TOTAL_TRAJ"
if [ -n "$GROUND_MATERIAL_USD" ]; then
    echo " Ground material: $GROUND_MATERIAL_USD"
else
    echo " Ground material: <default>"
fi
echo "==============================================="

while read -r line; do
    # 跳过空行
    if [[ -z "$line" ]]; then
        continue
    fi

    GARMENT_PATH="$line"
    echo ""
    echo ">>> Garment: $GARMENT_PATH"

    for ((i=1; i<=REPEAT_PER_GARMENT; i++)); do
        CURRENT_TRAJ=$((CURRENT_TRAJ + 1))
        echo "  -> Trajectory ${CURRENT_TRAJ}/${TOTAL_TRAJ} (repeat $i/$REPEAT_PER_GARMENT)"

        # 组装命令
        CMD=( "$ISAAC_PATH" Env_StandAlone/Fold_Tops_Env.py
              --usd_path "$GARMENT_PATH"
              --env_random_flag True
              --data_collection_flag True
              --record_video_flag True )

        if [ -n "$GROUND_MATERIAL_USD" ]; then
            CMD+=( --ground_material_usd "$GROUND_MATERIAL_USD" )
        fi

        # 每次调用都会：
        # - 启动一个新的 SimulationApp
        # - 运行一条轨迹并保存
        # - 在脚本内部调用 simulation_app.close() 关闭
        "${CMD[@]}"

        # 稍微等一会，避免频繁重启导致系统过载
        sleep 3
    done
done < "$ASSETS_FILE"

echo ""
echo "All $TOTAL_TRAJ trajectories for $TASK_NAME have been collected."