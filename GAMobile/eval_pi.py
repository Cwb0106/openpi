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

# --- <<< 导入 mshab (来自 eval_bc.py) >>> ---
from mshab.envs.make import EnvConfig, make_env
from mshab.utils.array import to_tensor
from mshab.utils.config import parse_cfg
from mshab.utils.logger import Logger, LoggerConfig
from mani_skill.utils import common

# --- <<< 导入 openpi (来自 eval_pi.py) >>> ---
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
import openpi.policies.mshab_policy as mshab_policy
import openpi.transforms as _transforms


def eval(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =======================================================================
    # 创建eval环境 (来自 eval_bc.py)
    # =======================================================================
    print("making eval env")
    eval_envs = make_env(
        cfg.eval_env,
        video_path=cfg.logger.eval_video_path,
    )
    print("made")
    
    num_envs = eval_envs.num_envs # 将从 config 加载 (例如: 8)
    action_dim = eval_envs.action_space.shape[0] # (例如: 13)
    print(f"已创建 {num_envs} 个并行环境。")

    eval_obs, _ = eval_envs.reset(seed=cfg.seed + 1_000_000)
    eval_envs.action_space.seed(cfg.seed + 1_000_000)

    # =======================================================================
    # 加载日志 (来自 eval_bc.py)
    # =======================================================================
    logger = Logger(logger_cfg=cfg.logger)

    # =======================================================================
    # 模型 (来自 eval_pi.py)
    # =======================================================================
    print("正在加载 Pi0 配置 'pi0_mshab_fetch'...")
    config = _config.get_config("pi0_mshab_fetch")
    checkpoint_dir = "/raid/wenbo/project/openpi/checkpoints/pi0_mshab_fetch/my_8gpu_run/7000"

    print(f"正在从 {checkpoint_dir} 创建策略...")
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    print("Pi0 策略加载成功。")

    # =======================================================================
    # 开始评估 (!!! 采用 eval_bc.py 架构 !!!)
    # =======================================================================
    
    # --- Pi0 动作分块 (Action Chunking) 设置 ---
    REPLAN_STEPS = 5 # 每 5 步重新规划一次 (只考虑 action chunk)
    try:
        ACTION_HORIZON = config.model.action_horizon
    except AttributeError:
        ACTION_HORIZON = 10
    REPLAN_STEPS = min(REPLAN_STEPS, ACTION_HORIZON)

    action_plan_queue = None # (B, REPLAN_STEPS, ActionDim) 动作计划队列
    action_plan_step_counter = 0 # 追踪我们执行到第几步
    # --- Pi0 设置结束 ---

    eval_obs, _ = eval_envs.reset() # (来自 eval_bc.py)

    # --- 评估循环 (来自 eval_bc.py) ---
    pbar = tqdm(total=cfg.algo.num_eval_episodes, desc="Evaluating Policy")
    
    while len(eval_envs.return_queue) < cfg.algo.num_eval_episodes:
        
        # --- [!!! 关键: Pi0 策略逻辑 (无微循环) !!!] ---
        # 检查是否需要重新规划
        if action_plan_step_counter == 0:
            
            # 1. 准备 obs (policy.infer_batched 期望 numpy)
            obs_for_policy = {
                k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v 
                for k, v in eval_obs.items()
            }
            if 'prompt' in eval_obs: # 特别处理 prompt (list[str])
                obs_for_policy['prompt'] = eval_obs['prompt']

            # 2. 一次性推理完整的批次 (B=8)
            # (!!!) 警告：如果 num_envs=8 导致 OOM，这里会崩溃 (!!!)
            action_chunk_dict = policy.infer_batched(obs_for_policy)
      
            # 3. 提取我们关心的 N 步计划
            action_chunk_np = action_chunk_dict["actions"] 
            action_plan_np = action_chunk_np[:, :REPLAN_STEPS, :]
            
            # 4. 转换为 Tensor 存入队列
            action_plan_queue = torch.from_numpy(action_plan_np).to(device)
        
        # --- [!!! 动作执行 (与 eval_bc.py 相同) !!!] ---
        
        # 1. 从计划队列中获取*当前*时间步的动作 (B, A)
        action = action_plan_queue[:, action_plan_step_counter, :]
        
        # 2. 在所有环境中执行这一步
        eval_obs, _, _, _, _ = eval_envs.step(action)
        
        # 3. 更新计数器
        action_plan_step_counter = (action_plan_step_counter + 1) % REPLAN_STEPS
        
        # --- [!!! 进度条更新 (来自 eval_bc.py) !!!] ---
        pbar.n = len(eval_envs.return_queue)
        pbar.refresh()
    
    pbar.close()
    print(f"Evaluation finished. Collected {len(eval_envs.return_queue)} episodes.")
    # --- 评估循环结束 ---


    # --- 日志记录 (来自 eval_bc.py) ---
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
        # eval_envs.reset_queues()
    
    print("="*30)
    ipdb.set_trace() # 你可以在这里打断点
    success_rate = common.to_tensor(eval_envs.success_once_queue, device=device).float().mean()
    print(f"当前的 policy 成功率是: {success_rate.item() * 100:.2f}%")
    print("="*30)
    
    eval_envs.close()
    logger.close()


# =======================================================================
# Config (来自 eval_bc.py)
# =======================================================================
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
    
    # --- <<< 新增: 评估专用配置 >>> ---
    num_eval_episodes: int = 16
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
            # 在 eval_pi.py 中, 我们不检查 bc 模型的 ckpt
            # assert self.model_ckpt.exists(), f"Could not find {self.model_ckpt}"

        self.algo.num_eval_envs = self.eval_env.num_envs
        self.algo._additional_processing()

        self.logger.exp_cfg = asdict(self)
        del self.logger.exp_cfg["logger"]["exp_cfg"]
        del self.logger.exp_cfg["resume_logdir"]
        del self.logger.exp_cfg["model_ckpt"]


def get_mshab_train_cfg(cfg: TrainConfig) -> TrainConfig:
    return from_dict(data_class=TrainConfig, data=OmegaConf.to_container(cfg))


# =======================================================================
# __main__ (来自 eval_bc.py)
# =======================================================================
if __name__ == "__main__":

    # args = OmegaConf.load("/raid/wenbo/project/mshab/GAPartMobile/config/train_bc.yaml")
    PASSED_CONFIG_PATH = "/raid/wenbo/project/mshab/GAPartMobile/config/bc_pick.yml"
    cfg = get_mshab_train_cfg(parse_cfg(default_cfg_path=PASSED_CONFIG_PATH))

    # 调用评估函数
    eval(cfg)