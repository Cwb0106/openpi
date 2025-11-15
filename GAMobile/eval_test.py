import sys
import os
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import ipdb

import gymnasium as gym
import h5py
import numpy as np
import torch
from dacite import from_dict
from omegaconf import OmegaConf
from tqdm import tqdm

# --- <<< 导入简单的 BC Agent >>> ---
from mshab.agents.bc import Agent 
from mshab.envs.make import EnvConfig, make_env
from mshab.utils.array import to_tensor
from mshab.utils.config import parse_cfg
from mshab.utils.logger import Logger, LoggerConfig
from mani_skill.utils import common

# --- (移除了所有和 Dataset 相关的导入) ---

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
import openpi.policies.mshab_policy as mshab_policy
import openpi.transforms as _transforms


def eval(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =======================================================================
    # 创建eval环境
    # =======================================================================
    print("making eval env")

    eval_envs = make_env(
        cfg.eval_env,
        video_path=cfg.logger.eval_video_path,
    )
    print("made")

    eval_obs, _ = eval_envs.reset(seed=cfg.seed + 1_000_000)
    eval_envs.action_space.seed(cfg.seed + 1_000_000)

    # example = {
    #     'observation/image': torch.randn(IMG_SIZE, IMG_SIZE, 3), #
    #     'observation/wrist_image': torch.randn(IMG_SIZE, IMG_SIZE, 3), #
    #     'observation/state': torch.randn(JOINT_DIM), #
    #     # 'observation/gripper_position': torch.randn(1), # <-- 修正：使用 torch.randn
    #     'prompt': "open the fridge" #
    # }
    eval_obs['observation/state'] = eval_obs['observation/state'][:8]
    eval_obs['observation/image'] = eval_obs['observation/image'][:8]
    eval_obs['observation/wrist_image'] = eval_obs['observation/wrist_image'][:8]
    # eval_obs['prompt']  = eval_obs['prompt']
    # eval_obs['prompt'] = eval_obs['prompt'][:4]
    ipdb.set_trace()
    # =======================================================================
    # 加载日志
    # =======================================================================
    # logger = Logger(logger_cfg=cfg.logger)

    # =======================================================================
    # 模型
    # =======================================================================

    # --- <<< 使用简单的 BC Agent >>> ---
    config = _config.get_config("pi0_mshab_fetch")
    checkpoint_dir = "/raid/wenbo/project/openpi/checkpoints/pi0_mshab_fetch/my_8gpu_run/7000"

    # Create a trained policy (automatically detects PyTorch format)
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    print("1111")

    # Run inference (same API as JAX)
    action_chunk = policy.infer_batched(eval_obs)["actions"]
    print("这里结束了222")

    ipdb.set_trace()
    exit(1)

    # =======================================================================
    # 开始评估
    # =======================================================================
    agent.eval()
    
    print(f"开始评估，总共需要 {cfg.algo.num_eval_episodes} 个回合...")
    progress_bar = tqdm(total=cfg.algo.num_eval_episodes, desc="Evaluating")

    # 循环直到收集到足够的回合
    while len(eval_envs.return_queue) < cfg.algo.num_eval_episodes:
        eval_obs = to_tensor(eval_obs, device=device, dtype="float")
        with torch.no_grad():
            # --- <<< 修改：使用无状态的 BC 模型 >>> ---
            action = agent(eval_obs)
            
        eval_obs, _, _, _, infos = eval_envs.step(action)
        
        # --- <<< 移除 Memory 传递逻辑 >>> ---

        # 检查是否有环境完成
        if "_final_info" in infos:
            num_done = np.sum(infos["_final_info"])
            if num_done > 0:
                progress_bar.update(num_done)
    
    progress_bar.close()
    print("评估完成。")

    # =======================================================================
    # 记录结果
    # =======================================================================
    if len(eval_envs.return_queue) > 0:
        logger.store(
            "eval",
            return_per_step=common.to_tensor(eval_envs.return_queue, device=device)
            .float()
            .mean()
            / eval_envs.max_episode_steps,
            success_once=common.to_tensor(eval_envs.success_once_queue, device=device)
            .float()
            .mean(),
            success_at_end=common.to_tensor(eval_envs.success_at_end_queue, device=device)
            .float()
            .mean(),
            len=common.to_tensor(eval_envs.length_queue, device=device).float().mean(),
        )
        eval_envs.reset_queues()
    
    # 使用 step=0 记录评估结果
    logger.log(0) 

    eval_envs.close()
    logger.close()

@dataclass
class BCConfig:
    name: str = "bc"

    # Training (保留这些字段以便能加载旧的config, 但在eval中不会使用)
    lr: float = 3e-4
    batch_size: int = 512
    epochs: int = 100
    log_freq: int = 1
    save_freq: int = 1
    save_backup_ckpts: bool = False

    # Dataset (保留这些字段以便能加载旧的config, 但在eval中不会使用)
    data_dir_fp: str = None
    max_cache_size: int = 0
    trajs_per_obj: Union[str, int] = "all"
    
    # --- <<< 移除 memory_slots >>> ---

    # --- <<< 新增: 评估专用配置 >>> ---
    num_eval_episodes: int = 100
    """在评估中运行的总回合数"""

    # Running
    eval_freq: int = 2
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""

    # passed from env/eval_env cfg
    num_eval_envs: int = field(init=False)
    """the number of parallel environments"""

    def _additional_processing(self):
        assert self.name == "bc", "Wrong algo config"
        try:
            self.trajs_per_obj = int(self.trajs_per_obj)
        except:
            pass
        assert isinstance(self.trajs_per_obj, int) or self.trajs_per_obj == "all"


@dataclass
class TrainConfig:
    seed: int
    eval_env: EnvConfig
    algo: BCConfig
    logger: LoggerConfig

    wandb_id: Optional[str] = None
    resume_logdir: Optional[Union[Path, str]] = None
    model_ckpt: Optional[Union[Path, int, str]] = None

    def __post_init__(self):
        assert (
            self.resume_logdir is None or not self.logger.clear_out
        ), "Can't resume to a cleared out logdir!"

        if self.resume_logdir is not None:
            self.resume_logdir = Path(self.resume_logdir)
            old_config_path = self.resume_logdir / "config.yml"
            if old_config_path.absolute() == Path(PASSED_CONFIG_PATH).absolute():
                assert (
                    self.resume_logdir == self.logger.exp_path
                ), "if setting resume_logdir, must set logger workspace and exp_name accordingly"
            else:
                assert (
                    old_config_path.exists()
                ), f"Couldn't find old config at path {old_config_path}"
                old_config = get_mshab_train_cfg(
                    parse_cfg(default_cfg_path=old_config_path)
                )
                self.logger.workspace = old_config.logger.workspace
                self.logger.exp_path = old_config.logger.exp_path
                self.logger.log_path = old_config.logger.log_path
                self.logger.model_path = old_config.logger.model_path
                self.logger.train_video_path = old_config.logger.train_video_path
                self.logger.eval_video_path = old_config.logger.eval_video_path

            if self.model_ckpt is None:
                # 如果恢复日志，但没有指定 ckpt，则默认为 latest.pt
                self.model_ckpt = self.logger.model_path / "latest.pt"

        if self.model_ckpt is not None:
            self.model_ckpt = Path(self.model_ckpt)
            assert self.model_ckpt.exists(), f"Could not find {self.model_ckpt}"

        self.algo.num_eval_envs = self.eval_env.num_envs
        self.algo._additional_processing()

        self.logger.exp_cfg = asdict(self)
        del self.logger.exp_cfg["logger"]["exp_cfg"]
        del self.logger.exp_cfg["resume_logdir"]
        del self.logger.exp_cfg["model_ckpt"]


def get_mshab_train_cfg(cfg: TrainConfig) -> TrainConfig:
    return from_dict(data_class=TrainConfig, data=OmegaConf.to_container(cfg))


if __name__ == "__main__":

    # args = OmegaConf.load("/raid/wenbo/project/mshab/GAPartMobile/config/train_bc.yaml")
    PASSED_CONFIG_PATH = "/raid/wenbo/project/mshab/GAPartMobile/config/bc_pick.yml"
    cfg = get_mshab_train_cfg(parse_cfg(default_cfg_path=PASSED_CONFIG_PATH))



    # 调用评估函数
    eval(cfg)