import copy
import os
import pathlib
import pdb
import random
import shutil
import sys
import threading
import time

import dill
import hydra
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf
from termcolor import cprint
from torch.utils.data import DataLoader

import pdb

# 为了从项目根目录运行脚本时，能够正确 import 到本仓库下的模块
sys.path.append('Model_HALO/SADP_G')

from structure_aware_diffusion_policy_garment.common.checkpoint_util import TopKCheckpointManager
from structure_aware_diffusion_policy_garment.common.pytorch_util import dict_apply, optimizer_to
from structure_aware_diffusion_policy_garment.dataset.base_dataset import BaseDataset
from structure_aware_diffusion_policy_garment.env_runner.base_runner import BaseRunner
from structure_aware_diffusion_policy_garment.model.common.lr_scheduler import get_scheduler
from structure_aware_diffusion_policy_garment.model.diffusion.ema_model import EMAModel
from structure_aware_diffusion_policy_garment.policy.sadp_g import SADP_G
from hydra.core.hydra_config import HydraConfig

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainSADPGWorkspace:
    """
    SADP_G 训练工作空间（workspace），负责：
    1. 根据 Hydra 配置构建模型 / 数据集 / 优化器 / EMA / 日志等；
    2. 管理训练状态（epoch / global_step）；
    3. 执行训练、验证、保存 checkpoint；
    4. 提供从 checkpoint 恢复 policy 和 env_runner 的接口。
    """
    include_keys = ["global_step", "epoch"]
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        """
        初始化训练工作空间。

        参数：
        - cfg: Hydra 加载的配置对象（包含 training / policy / task 等全部信息）
        - output_dir: Hydra 的输出目录（可选，一般由 Hydra 自动设置）
        """
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

        # ========== 设置随机种子，保证实验可复现 ==========
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        print("SADP_G:", os.getcwd())
        # ========== 根据配置实例化 policy 模型 ==========
        self.model: SADP_G = hydra.utils.instantiate(cfg.policy)

        self.ema_model: SADP_G = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except:  # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)

        # ========== 构建优化器 ==========
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters()
        )

        # ========== 初始化训练状态计数器 ==========
        self.global_step = 0
        self.epoch = 0

    def run(self):
        """
        主训练入口：
        - 初始化 WandB / 数据集 / DataLoader / 学习率调度 / EMA 等模块；
        - 按 epoch 执行训练与验证；
        - 按配置周期保存 checkpoint。
        """
        cfg = copy.deepcopy(self.cfg)

        # if cfg.logging.mode == "online":
        #     WANDB = True
        # else:
        #     WANDB = False
        WANDB = True

        if cfg.training.debug:
            cfg.training.num_epochs = 5
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 1
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 5
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = False
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False

        RUN_ROLLOUT = False
        RUN_VALIDATION = True  # reduce time cost

        # ========== 如果需要，从 checkpoint 恢复训练 ==========
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # ========== 构建训练数据集和 DataLoader ==========
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        assert isinstance(dataset, BaseDataset), print(
            f"dataset must be BaseDataset, got {type(dataset)}"
        )
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # ========== 构建验证数据集和 DataLoader ==========
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # 将归一化器注册到模型与 EMA 模型中，方便统一数据前/后处理
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # ========== 学习率调度器 ==========
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # ========== EMA 管理器 ==========
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        env_runner = None

        # cfg.logging.name = str(cfg.task.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        # configure logging
        if WANDB:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging,
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )

        # ========== Top-K checkpoint 管理器（本文件中主要用 save_checkpoint 手动保存） ==========
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint.topk
        )

        # ========== 将模型与优化器移动到目标设备（CPU / GPU） ==========
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # 保存一份训练 batch，用于后续可视化或采样（如需要）
        train_sampling_batch = None
        checkpoint_num = 1

        # ========== 主训练循环 ==========
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            with tqdm.tqdm(
                train_dataloader,
                desc=f"Training epoch {self.epoch}",
                leave=False,
                mininterval=cfg.training.tqdm_interval_sec,
            ) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()
                    # 将 batch 搬到目标 device（支持非阻塞）
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    # --------- 计算损失并反向传播（支持梯度累积） ---------
                    t1_1 = time.time()
                    raw_loss, loss_dict = self.model.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    t1_2 = time.time()

                    # --------- 按梯度累积步数进行一次优化器更新 ---------
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    t1_3 = time.time()
                    # --------- EMA 更新（如果启用） ---------
                    if cfg.training.use_ema:
                        ema.step(self.model)
                    t1_4 = time.time()
                    # --------- 记录日志信息（loss / lr 等） ---------
                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        "train_loss": raw_loss_cpu,
                        "global_step": self.global_step,
                        "epoch": self.epoch,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    t1_5 = time.time()
                    step_log.update(loss_dict)
                    t2 = time.time()

                    if verbose:
                        print(f"total one step time: {t2-t1:.3f}")
                        print(f" compute loss time: {t1_2-t1_1:.3f}")
                        print(f" step optimizer time: {t1_3-t1_2:.3f}")
                        print(f" update ema time: {t1_4-t1_3:.3f}")
                        print(f" logging time: {t1_5-t1_4:.3f}")

                    is_last_batch = batch_idx == (len(train_dataloader) - 1)
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        if WANDB:
                            wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) and batch_idx >= (
                        cfg.training.max_train_steps - 1
                    ):
                        break

            # epoch 结束时，用该 epoch 的平均训练 loss 作为最终指标
            train_loss = np.mean(train_losses)
            step_log["train_loss"] = train_loss

            # ========= eval for this epoch ==========
            policy = self.model
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()

            # run validation
            if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(
                        val_dataloader,
                        desc=f"Validation epoch {self.epoch}",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                    ) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(
                                batch, lambda x: x.to(device, non_blocking=True)
                            )
                            loss, loss_dict = self.model.compute_loss(batch)
                            val_losses.append(loss)
                            print(f"epoch {self.epoch}, eval loss: ", float(loss.cpu()))
                            if (
                                cfg.training.max_val_steps is not None
                            ) and batch_idx >= (cfg.training.max_val_steps - 1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        # log epoch average validation loss
                        step_log["val_loss"] = val_loss

            # ========== 按频率保存 checkpoint ==========
            if (
                (self.epoch + 1) % cfg.training.checkpoint_every
            ) == 0 and cfg.checkpoint.save_ckpt:

                if not cfg.policy.use_pc_color:
                    # 保存到共享目录 /share_data/yanruilin/checkpoints/Fold_Tops/{task_name}/
                    base_ckpt_dir = f"./Model_HALO/SADP_G/checkpoints/{self.cfg.task.name}"
                    if not os.path.exists(base_ckpt_dir):
                        os.makedirs(base_ckpt_dir, exist_ok=True)
                    save_path = f"{base_ckpt_dir}/{self.epoch + 1}.ckpt"
                else:
                    base_ckpt_dir = f"./Model_HALO/SADP_G/checkpoints/{self.cfg.task.name}_w_rgb"
                    if not os.path.exists(base_ckpt_dir):
                        os.makedirs(base_ckpt_dir, exist_ok=True)
                    save_path = f"{base_ckpt_dir}/{self.epoch + 1}.ckpt"

                self.save_checkpoint(save_path)

            # ========= eval end for this epoch ==========
            policy.train()

            # 记录本 epoch 最后一步（包含验证指标）的日志
            if WANDB:
                wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            del step_log

    def get_policy_and_runner(self, cfg, checkpoint_num=3000, task_name="Hang_Tops_stage_1"):
        """
        根据给定的 checkpoint 编号与任务名，加载 policy 与环境 runner。

        返回：
        - policy: 已经切换到 eval + cuda 的策略网络（若开启 EMA，则返回 EMA 模型）
        - env_runner: 用于 roll-out 的环境 runner
        """
        # load the latest checkpoint
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner, output_dir=None
        )
        assert isinstance(env_runner, BaseRunner)

        if not cfg.policy.use_pc_color:
            ckpt_file = pathlib.Path(
                f"./Model_HALO/SADP_G/checkpoints/{task_name}/{checkpoint_num}.ckpt"
            )
        else:
            ckpt_file = pathlib.Path(
                f"./Model_HALO/SADP_G/checkpoints/{task_name}_w_rgb/{checkpoint_num}.ckpt"
            )
        #import ipdb; ipdb.set_trace()
        print("ckpt file exist:", ckpt_file.is_file())

        if ckpt_file.is_file():
            cprint(f"Resuming from checkpoint {ckpt_file}", "magenta")
            self.load_checkpoint(path=ckpt_file)

        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda()
        return policy, env_runner

    @property
    def output_dir(self):
        """
        当前实验的输出目录。

        优先使用初始化时传入的 output_dir，否则从 HydraConfig 中读取。
        """
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def save_checkpoint(
        self,
        path=None,
        tag="latest",
        exclude_keys=None,
        include_keys=None,
        use_thread=False,
    ):
        """
        保存 checkpoint。

        保存内容包括：
        - cfg：完整配置；
        - state_dicts：所有包含 state_dict 的模块（模型、优化器、EMA 等）；
        - pickles：额外通过 dill 序列化的对象（例如 output_dir）。

        参数：
        - path: 指定保存路径；若为 None，则根据 output_dir 与 tag 自动生成路径；
        - tag: checkpoint 标签名（默认 "latest"）；
        - exclude_keys: 不需要保存 state_dict 的 key（黑名单）；
        - include_keys: 需要以 pickle 形式保存的属性（白名单）；
        - use_thread: 是否在子线程中异步保存（避免阻塞训练）。
        """
        print("saved in ", path)
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ("_output_dir",)

        path.parent.mkdir(parents=False, exist_ok=True)
        # payload 结构：
        # - cfg: 实验配置
        # - state_dicts: 所有可用 state_dict 描述的模块
        # - pickles: 其它通过 dill 序列化的对象
        payload = {"cfg": self.cfg, "state_dicts": dict(), "pickles": dict()}

        for key, value in self.__dict__.items():
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload["state_dicts"][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload["state_dicts"][key] = value.state_dict()
            elif key in include_keys:
                payload["pickles"][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda: torch.save(payload, path.open("wb"), pickle_module=dill)
            )
            self._saving_thread.start()
        else:
            torch.save(payload, path.open("wb"), pickle_module=dill)

        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())

    def get_checkpoint_path(self, tag="latest"):
        """
        根据 tag 获取 checkpoint 文件路径。

        支持：
        - "latest": 直接返回 latest.ckpt；
        - "best": 根据文件名里的 test_mean_score 选出最优 ckpt；
        - 其它 tag：目前未实现。
        """
        if tag == "latest":
            return pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")
        elif tag == "best":
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath("checkpoints")
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if "latest" in ckpt:
                    continue
                score = float(ckpt.split("test_mean_score=")[1].split(".ckpt")[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath("checkpoints", best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        """
        将 payload 中的内容加载回当前 workspace。

        参数：
        - payload: save_checkpoint 生成的字典；
        - exclude_keys: 从 state_dicts 中跳过的 key；
        - include_keys: 从 pickles 中需要反序列化回来的 key。
        """
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload["pickles"].keys()

        for key, value in payload["state_dicts"].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = dill.loads(payload["pickles"][key])

    def load_checkpoint(
        self, path=None, tag="latest", exclude_keys=None, include_keys=None, **kwargs
    ):
        """
        从 checkpoint 文件中恢复 workspace 状态。

        参数：
        - path: checkpoint 路径；若为 None，则通过 tag 自动推断；
        - tag: "latest" 或 "best"；
        - exclude_keys / include_keys: 传递给 load_payload 的过滤条件；
        - kwargs: 透传给各模块的 load_state_dict（例如 strict=False）。

        返回：
        - payload: 从磁盘加载的原始字典。
        """
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")
        self.load_payload(payload, exclude_keys=exclude_keys, include_keys=include_keys)
        return payload

    @classmethod
    def create_from_checkpoint(
        cls, path, exclude_keys=None, include_keys=None, **kwargs
    ):
        """
        通过已有 checkpoint 创建一个新的 TrainSADPGWorkspace 实例。

        步骤：
        1. 从磁盘读取 payload；
        2. 用 payload["cfg"] 创建 workspace；
        3. 调用 load_payload 恢复各模块状态。
        """
        payload = torch.load(open(path, "rb"), pickle_module=dill)
        instance = cls(payload["cfg"])
        instance.load_payload(
            payload=payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs,
        )
        return instance

    def save_snapshot(self, tag="latest"):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        （注：snapshot 直接序列化整个 workspace 对象，适合短期实验快速中断与恢复，
        若代码发生改动，老的 snapshot 可能无法再正确加载。）
        """
        path = pathlib.Path(self.output_dir).joinpath("snapshots", f"{tag}.pkl")
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open("wb"), pickle_module=dill)
        return str(path.absolute())

    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, "rb"), pickle_module=dill)

def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu")
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)


@hydra.main(
    version_base=None,
    config_path=str(
        pathlib.Path(__file__).parent.joinpath("structure_aware_diffusion_policy_garment", "config")
    ),
)
def main(cfg):
    """
    Hydra 程序入口：
    - 根据 config_path 指向的目录加载 YAML 配置；
    - 使用 cfg 构造 TrainSADPGWorkspace 并启动训练。

    使用示例（在项目根目录）：
    python Model_HALO/SADP_G/train.py task=Fold_Tops_stage_1 training.device=cuda:0
    """
    workspace = TrainSADPGWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
