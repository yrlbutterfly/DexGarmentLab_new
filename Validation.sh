# Examples:
# bash Validation.sh Hang_Coat 100 100

# 'task_name' e.g. Hang_Coat, Hang_Tops, Wear_Scarf, etc.
# 'validation_num' The episodes number you need to validate. e.g. 50, 100, etc.
# 'training_data_num' The expert data number used for training policy. e.g. 100, 200, 300, etc.

# when you run this script, you need to input the checkpoint number parameters according to task.
# for example:
# As for Fold Dress which has three stages, 
# you need to input the stage_1_checkpoint_num, stage_2_checkpoint_num, stage_3_checkpoint_num.

#!/bin/bash

###############################################################################
# 示例用法（Usage Examples）
###############################################################################
# 1. 以 Hang_Coat 任务为例，在已有 100 条专家数据上进行 100 次验证：
#    bash Validation.sh Hang_Coat 100 100
#
# 2. 参数说明（Parameters）：
#    - task_name         : 任务名称，例如：
#                          Hang_Coat, Hang_Tops, Wear_Scarf, Fold_Dress, Fold_Tops 等
#    - validation_num    : 需要跑多少条验证 episode，比如 50、100
#    - training_data_num : 训练该策略时使用的专家数据数量，例如 100、200、300
#
# 3. 关于 checkpoint 输入：
#    - 不同任务的阶段（stage）数不同，本脚本会根据任务名称来决定需要你输入几个阶段的
#      checkpoint 序号：
#        * Fold_Dress / Fold_Tops ：有 3 个阶段，需要输入 stage_1 / stage_2 / stage_3
#        * Fling_Dress / Fling_Tops / Fold_Trousers ：有 2 个阶段，需要输入 stage_1 / stage_2
#        * 其他任务：默认只需要 1 个阶段的 checkpoint（stage_1）
###############################################################################

#==============================#
# 一、获取命令行参数           #
#==============================#
# $1: 任务名称（task_name）
# $2: 验证 episode 数量（validation_num）
# $3: 训练数据数量（training_data_num）
# 获取参数
task_name=$1          # 任务名称
validation_num=$2
training_data_num=$3

#==============================#
# 二、根据任务类型决定需要的   #
#     checkpoint 阶段数量      #
#==============================#
# 这里仅决定“需要向用户询问哪些 checkpoint 序号”，
#   实际在仿真中会通过传给 Python 的参数来加载对应 ckpt。
if [[ "$task_name" == "Fold_Dress" || "$task_name" == "Fold_Tops" ]]; then
    # 三阶段任务：需要手动输入 1/2/3 阶段的 checkpoint 序号
    # 这里需要修改成统一的checkpoint目录
    read -p "Please Input <stage_1_checkpoint_num>: " stage_1_checkpoint_num
    read -p "Please Input <stage_2_checkpoint_num>: " stage_2_checkpoint_num
    read -p "Please Input <stage_3_checkpoint_num>: " stage_3_checkpoint_num
elif [[ "$task_name" == "Fling_Dress" || "$task_name" == "Fling_Tops" || "$task_name" == "Fold_Trousers" ]]; then
    # 二阶段任务：只需要 1/2 阶段，第三阶段恒为 0（代表无）
    read -p "Please Input <stage_1_checkpoint_num>: " stage_1_checkpoint_num
    read -p "Please Input <stage_2_checkpoint_num>: " stage_2_checkpoint_num
    stage_3_checkpoint_num=0
else
    # 其他任务：只需要 1 个阶段，2/3 阶段一律置 0
    read -p "Please Input <stage_1_checkpoint_num>: " stage_1_checkpoint_num
    stage_2_checkpoint_num=0
    stage_3_checkpoint_num=0
fi

#==============================#
# 三、Isaac 仿真环境相关配置   #
#==============================#
type=HALO                          # 当前使用的策略 / 模型类别标签，这里用 HALO 表示
isaac_path=~/isaacsim_4.5.0/python.sh  # Isaac Sim 中 python.sh 的路径（按自己安装位置修改）

export ISAAC_PATH=$isaac_path      # 将 Isaac 的 python 路径导出为环境变量，方便后面直接使用

#==============================#
# 四、创建验证结果存放目录     #
#==============================#
# 目录结构示例：
#   Data/{task_name}_Validation_HALO/
#     ├── final_state_pic/   # 每次验证结束时的最终状态截图
#     ├── video/             # 每次验证过程的视频（原脚本中是 vedio，保留不改名）
#     └── validation_log.txt # 日志信息（可在 Python 端写入）
base_dir="Data/${task_name}_Validation_${type}"
mkdir -p "${base_dir}/final_state_pic"
mkdir -p "${base_dir}/video"
touch "${base_dir}/validation_log.txt"

#==============================#
# 五、统计当前已生成的验证数量 #
#==============================#
# 通过统计 final_state_pic 目录下文件数量，判断已经完成了多少条验证。
current_num=$(ls "${base_dir}/final_state_pic" | wc -l)

#==============================#
# 六、进度条打印函数           #
#==============================#
# print_progress current total task_name
# - 仅向 stderr 打印，不会污染 stdout（方便后续 stdout 做日志/管道处理）
print_progress() {
    local current=$1
    local total=$2
    local task=$3
    local width=50
    local percent=$((100 * current / total))
    local filled=$((width * current / total))   # 已完成的长度
    local empty=$((width - filled))             # 未完成的长度

    # 生成进度条字符串：
    # - 用 █ 表示已完成部分
    # - 用空格表示未完成部分
    local bar=$(printf "%0.s█" $(seq 1 $filled))
    bar+=$(printf "%0.s " $(seq 1 $empty))

    # 输出任务名和进度条，到标准错误输出（stderr）
    printf "\rTask: %-20s |%s| %3d%% (%d/%d)" "$task" "$bar" "$percent" "$current" "$total" >&2
}

#==============================#
# 七、主验证循环               #
#==============================#
# 循环条件：当前已完成数量 < 目标验证数量 validation_num
while [ "$current_num" -lt "$validation_num" ]; do

    # (1) 实时打印进度条，显示当前进度
    print_progress "$current_num" "$validation_num" "$task_name"

    # (2) 调用 Isaac 环境进行一次完整的验证 episode
    #     - 这里将 stdout 重定向到 /dev/null，并将 stderr 一并重定向（> /dev/null 2>&1），
    #       避免仿真过程中的大量输出刷屏。
    #     - 如果你希望在终端中看到 Isaac 的详细输出，可删除 `> /dev/null 2>&1`。
    $ISAAC_PATH Env_Validation/${task_name}_${type}.py \
        --env_random_flag True \
        --garment_random_flag True \
        --record_video_flag True \
        --validation_flag True \
        --training_data_num "$training_data_num" \
        --stage_1_checkpoint_num "$stage_1_checkpoint_num" \
        --stage_2_checkpoint_num "$stage_2_checkpoint_num" \
        --stage_3_checkpoint_num "$stage_3_checkpoint_num" \
        > /dev/null 2>&1

    # (3) 每跑完一条验证，就重新统计 final_state_pic 中的文件数量，
    #     以此作为“当前已完成验证数”。
    current_num=$(ls "${base_dir}/final_state_pic" | wc -l)

    # (4) 适当 sleep，避免在极端情况下高频率地启动仿真
    sleep 5
done

#==============================#
# 八、收尾工作                 #
#==============================#
# 循环结束后，再打印一次完整进度条，确保为 100%
print_progress "$current_num" "$validation_num" "$task_name"

# 在 stderr 中换行，保证提示美观
echo >&2
