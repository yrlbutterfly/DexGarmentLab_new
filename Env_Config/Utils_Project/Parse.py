import argparse
import numpy as np

def parse_args_record():
    parser=argparse.ArgumentParser()

    def str_to_ndarray(value):
        value = value.strip('"').strip("\\")
        value_list=value.split(',')
        if len(value_list)!=3:
            raise argparse.ArgumentTypeError("The length of pos and ori must be 3")
        return np.array(value_list).astype(float)
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser.add_argument("--ground_material_usd", type=str, default=None, help="ground material usd path")
    parser.add_argument("--data_collection_flag",type=str2bool, default=False, help="data collection flag")
    parser.add_argument("--record_video_flag",type=str2bool, default=False, help="record vedio flag")
    parser.add_argument("--env_random_flag", type=str2bool, default=False, help="env random flag")
    # 在数据采集脚本中也支持多衣服随机采集 / 随机衣服标志
    # 与验证脚本中的定义保持一致，提供两个等价的命令行名称：
    #   --multi_garment_collection_flag  或  --garment_random_flag
    parser.add_argument(
        "--multi_garment_collection_flag",
        "--garment_random_flag",
        dest="garment_random_flag",
        type=str2bool,
        default=False,
        help="whether to randomly choose one garment from training list (alias: --garment_random_flag)",
    )
    # 可选：直接指定一件衣服的 usd 路径（相对工程根目录或绝对路径）
    parser.add_argument(
        "--usd_path",
        type=str,
        default=None,
        help="specific garment usd path (relative to project root or absolute path)",
    )
    
    return parser.parse_args()

def parse_args_val():
    parser=argparse.ArgumentParser()
    
    def str_to_ndarray(value):
        value = value.strip('"').strip("\\")
        value_list=value.split(',')
        if len(value_list)!=3:
            raise argparse.ArgumentTypeError("The length of pos and ori must be 3")
        return np.array(value_list).astype(float)
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        
    def str2int(v):
        try:
            return int(v)
        except ValueError:
            raise argparse.ArgumentTypeError('Integer value expected.')
    
    parser.add_argument("--ground_material_usd", type=str, default=None, help="ground material usd path")
    parser.add_argument("--validation_flag",type=str2bool, default=False, help="validation flag")
    parser.add_argument("--record_video_flag",type=str2bool, default=False, help="record vedio flag")
    parser.add_argument("--env_random_flag", type=str2bool, default=False, help="env random flag")
    # 同样为验证脚本增加更清晰的别名
    parser.add_argument(
        "--multi_garment_collection_flag",
        "--garment_random_flag",
        dest="garment_random_flag",
        type=str2bool,
        default=False,
        help="whether to randomly choose one garment from training list (alias: --garment_random_flag)",
    )
    parser.add_argument(
        "--usd_path",
        type=str,
        default=None,
        help="specific garment usd path (relative to project root or absolute path)",
    )
    
    parser.add_argument("--training_data_num", type=str2int, default=100, help="training data number")
    parser.add_argument("--stage_1_checkpoint_num", type=str2int, default=1500, help="Stage 1 checkpoint number")
    parser.add_argument("--stage_2_checkpoint_num", type=str2int, default=1500, help="Stage 2 checkpoint number")
    parser.add_argument("--stage_3_checkpoint_num", type=str2int, default=1500, help="Stage 3 checkpoint number")

    # Debug 开关：用于保存 VLM 输出与可视化结果
    parser.add_argument(
        "--debug",
        type=str2bool,
        default=False,
        help="enable debug mode to save VLM outputs and visualization figures",
    )

    # VLM 相关参数：通过命令行传入，而不是依赖环境变量
    parser.add_argument(
        "--vlm_base_url",
        type=str,
        default="http://127.0.0.1:8001/v1",
        help="base url of the VLM service (OpenAI-compatible endpoint)",
    )
    parser.add_argument(
        "--vlm_model_name",
        type=str,
        required=True,
        help="model name or checkpoint path used by VLM",
    )
    
    return parser.parse_args()