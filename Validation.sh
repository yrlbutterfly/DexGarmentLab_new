#!/bin/bash

# Examples:
# bash Validation.sh Hang_Coat 100 100
#
# 'task_name' e.g. Hang_Coat, Hang_Tops, Wear_Scarf, etc.
# 'validation_num' The episodes number you need to validate. e.g. 50, 100, etc.
# 'training_data_num' The expert data number used for training policy. e.g. 100, 200, 300, etc.

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
task_name=$1          # 任务名称
validation_num=$2
training_data_num=$3

#==============================#
# 二、根据任务类型决定需要的   #
#     checkpoint 阶段数量      #
#==============================#
# if [[ "$task_name" == "Fold_Dress" || "$task_name" == "Fold_Tops" ]]; then
#     # 三阶段任务：需要手动输入 1/2/3 阶段的 checkpoint 序号
#     # 这里需要修改成统一的checkpoint目录
#     read -p "Please Input <stage_1_checkpoint_num>: " stage_1_checkpoint_num
#     read -p "Please Input <stage_2_checkpoint_num>: " stage_2_checkpoint_num
#     read -p "Please Input <stage_3_checkpoint_num>: " stage_3_checkpoint_num
# elif [[ "$task_name" == "Fling_Dress" || "$task_name" == "Fling_Tops" || "$task_name" == "Fold_Trousers" ]]; then
#     # 二阶段任务：只需要 1/2 阶段，第三阶段恒为 0（代表无）
#     read -p "Please Input <stage_1_checkpoint_num>: " stage_1_checkpoint_num
#     read -p "Please Input <stage_2_checkpoint_num>: " stage_2_checkpoint_num
#     stage_3_checkpoint_num=0
# else
#     # 其他任务：只需要 1 个阶段，2/3 阶段一律置 0
#     read -p "Please Input <stage_1_checkpoint_num>: " stage_1_checkpoint_num
#     stage_2_checkpoint_num=0
#     stage_3_checkpoint_num=0
# fi

#==============================#
# 二.1、VLM 与 debug 相关参数  #
#==============================#
# 所有使用 parse_args_val 的验证脚本现在都需要 --vlm_model_name
#read -p "Please Input <vlm_model_name> (for VLM): " vlm_model_name
# 是否开启 debug 模式（保存 VLM 输出、RGB+bbox、pcd+feature 可视化）
#read -p "Enable debug mode? (True/False, default False): " debug_flag
if [[ -z "$debug_flag" ]]; then
    debug_flag=False
fi

#==============================#
# 三、Isaac 仿真环境相关配置   #
#==============================#
type=HALO                          # 当前使用的策略 / 模型类别标签
isaac_path=~/isaacsim_4.5.0/python.sh  # 按自己安装位置修改

export ISAAC_PATH=$isaac_path

#==============================#
# 四、创建验证结果存放目录     #
#==============================#
base_dir="Data/${task_name}_Validation_${type}"
mkdir -p "${base_dir}/final_state_pic"
mkdir -p "${base_dir}/video"
touch "${base_dir}/validation_log.txt"

#==============================#
# 五、统计当前已生成的验证数量 #
#==============================#
current_num=$(ls "${base_dir}/final_state_pic" | wc -l)

#==============================#
# 六、进度条打印函数           #
#==============================#
print_progress() {
    local current=$1
    local total=$2
    local task=$3
    local width=50
    local percent=$((100 * current / total))
    local filled=$((width * current / total))
    local empty=$((width - filled))

    local bar=$(printf "%0.s█" $(seq 1 $filled))
    bar+=$(printf "%0.s " $(seq 1 $empty))

    printf "\rTask: %-20s |%s| %3d%% (%d/%d)" "$task" "$bar" "$percent" "$current" "$total" >&2
}

#==============================#
# 七、主验证循环               #
#==============================#
while [ "$current_num" -lt "$validation_num" ]; do
    print_progress "$current_num" "$validation_num" "$task_name"

    $ISAAC_PATH Env_Validation/Fold_Tops_HALO.py \
        --env_random_flag True \
        --garment_random_flag True \
        --record_video_flag True \
        --validation_flag True \
        --training_data_num "$training_data_num" \
        --stage_1_checkpoint_num 3000 \
        --stage_2_checkpoint_num 0 \
        --stage_3_checkpoint_num 0 \
        --debug True \
        --vlm_model_name "/share_data/yanruilin/qwen3vl_full_sft_cloth_sim"

    current_num=$(ls "${base_dir}/final_state_pic" | wc -l)
    sleep 5
done

#==============================#
# 八、收尾工作                 #
#==============================#
print_progress "$current_num" "$validation_num" "$task_name"
echo >&2