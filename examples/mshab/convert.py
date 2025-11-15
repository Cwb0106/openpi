"""
用于将你的 mshab/HDF5 数据集转换为 LeRobot 格式 (适配 pi0_base) 的脚本。

此版本实现 (7-DoF Arm + 1-DoF Gripper) 映射。

用法:
uv run python convert_mshab_data_to_lerobot.py --data_dir /path/to/your/data
"""

import shutil
import os
import h5py
import json
import torch
import numpy as np
import tyro
from pathlib import Path
import sys

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# ==================================================================
# 关键：从你现有的 dataset.py 中导入 BCDataset
sys.path.append("/raid/wenbo/project/mshab/ManiSkill")
sys.path.append("/raid/wenbo/project/mshab/") 
from GAPartMobile.dataset.dataset_pi import PiBCDataset

# ==================================================================


# TODO: 修改为你希望在 Hugging Face Hub 上使用的仓库名称
REPO_NAME = "mshab_fetch_dataset"  

def main(data_dir: str, *, push_to_hub: bool = False):
    """
    Args:
        data_dir: 指向你的 H5 数据集的路径。
                  可以是一个包含 .h5/.json 文件的目录，
                  也可以是单个 .h5 文件的路径。
        push_to_hub: 是否将转换后的数据集推送到 Hugging Face Hub。
    """
    
    # --- 1. 清理已存在的输出目录 ---
    output_path = "/raid/wenbo/project/mshab/pre_dataset"
    # if output_path.exists():
    #     print(f"警告：正在删除已存在的目录: {output_path}")
    #     shutil.rmtree(output_path)

    # --- 2. 定义 LeRobot 数据集特征 ---
    
    # ==================================================================
    # 关键步骤
    
    # State 维度 (假设 42 不变)
    STATE_DIM = 32
    
    # 你的 *原始* mshab action (act) 的维度。
    # [0:7]=Arm(7D), [7]=Gripper(1D), [8]=Torso(1D), [9:10]=Head(2D), [11:12]=Base(2D)
    ORIGINAL_ACTION_DIM = 13
    
    # 你的 *目标* pi0 action 的维度。
    TARGET_PI0_ACTION_DIM = 32
    
    # 你的数据帧率 (FPS)。
    FPS = 10 
    # ==================================================================
    
    print(f"正在创建 LeRobot 数据集 (7-DoF Arm 映射)，配置如下:")
    print(f"  state 维度: {STATE_DIM}")
    print(f"  原始 action 维度: {ORIGINAL_ACTION_DIM}")
    print(f"  目标 (pi0) action 维度: {TARGET_PI0_ACTION_DIM}")

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="fetch",  # 根据你的 dataset.py 中的传感器名称推断
        fps=FPS,
        features={
            "image": { # 对应 'fetch_head_rgb'
                "dtype": "image",
                "shape": (128, 128, 3), # 基于你的 dataset.py
                "names": ["height", "width", "channel"],
            },
            "wrist_image": { # 对应 'fetch_hand_rgb'
                "dtype": "image",
                "shape": (128, 128, 3), # 基于你的 dataset.py
                "names": ["height", "width", "channel"],
            },
            "state": { # 对应你拼接的 'state' 向量
                "dtype": "float32",
                "shape": (STATE_DIM,), 
                "names": ["state"],
            },
            "actions": { # 对应 'act'
                "dtype": "float32",
                "shape": (TARGET_PI0_ACTION_DIM,), # <<< 必须是 16 维
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # --- 3. 查找所有 H5 文件 ---
    data_dir_path = Path(data_dir)
    h5_files_to_process = []
    
    if data_dir_path.is_file() and data_dir_path.name.endswith(".h5"):
        h5_files_to_process = [data_dir_path]
    elif data_dir_path.is_dir():
        h5_files_to_process = sorted([data_dir_path / f for f in os.listdir(data_dir_path) if f.endswith(".h5")])
    
    if not h5_files_to_process:
        raise FileNotFoundError(f"在 {data_dir} 中没有找到 .h5 文件")

    print(f"找到了 {len(h5_files_to_process)} 个 H5 文件准备转换...")

    # --- 4. 实例化 BCDataset 作为辅助工具 ---
    print("正在初始化 BCDataset (作为辅助工具)...")
    bc_dataset_helper = PiBCDataset(
        data_dir_fp=str(h5_files_to_process[0]), # 传入第一个文件路径以完成初始化
        max_cache_size=0,
        trajs_per_obj="all"
    )
    # 我们将手动迭代，所以关掉它默认打开的文件
    bc_dataset_helper.close() 

    # --- 5. 循环处理数据并写入 LeRobot 数据集 ---
    total_episodes = 0
    limit_reached = False  # <-- [新添加]
    for h5_path in h5_files_to_process:
        json_path = h5_path.with_suffix(".json")
        if not json_path.exists():
            print(f"警告: 跳过 {h5_path.name}, 找不到对应的 .json 文件。")
            continue

        print(f"\n--- 正在处理文件: {h5_path.name} ---")
        h5_file = h5py.File(h5_path, "r")
        with open(json_path, "rb") as f:
            json_data = json.load(f)

        # 循环处理文件中的每一个 episode
        for ep_json in json_data.get("episodes", []):
            ep_id = ep_json["episode_id"]
            ep_key = f"traj_{ep_id}"
            
            if ep_key not in h5_file:
                print(f"警告: 跳过 episode {ep_id}, 在H5文件中找不到 key '{ep_key}'。")
                continue
            
            ep_data = h5_file[ep_key]
            observation_data = ep_data["obs"]
            action_data = ep_data["actions"]
            
            # TODO: 找到你的任务指令 (language instruction)
            task_instruction = ep_json.get("task_description", "open the fridge") 
            
            if ep_json["elapsed_steps"] <= 0:
                continue

            # 循环处理 episode 中的每一步 (step)
            for step_num in range(ep_json["elapsed_steps"]):
                try:
                    # a. 使用 BCDataset 辅助工具加载数据
                    agent_obs = bc_dataset_helper.transform_idx(observation_data["agent"], step_num)
                    extra_obs = bc_dataset_helper.transform_idx(observation_data["extra"], step_num)
                    extra_obs["base_pos_wrt_world"] = bc_dataset_helper.transform_idx(ep_data["base_pos_wrt_world"], step_num)
                    
                    # b. 加载图像 (OpenPI 需要 RGB 图像)
                    head_rgb_tensor = bc_dataset_helper.transform_idx(observation_data["sensor_data"]["fetch_head"]["depth"], step_num)
                    hand_rgb_tensor = bc_dataset_helper.transform_idx(observation_data["sensor_data"]["fetch_hand"]["depth"], step_num)
                    
                    # c. 加载 原始 13D action
                    act_tensor = bc_dataset_helper.transform_idx(action_data, step_num)
                    
                    # d. 构造 state 向量 (与你的 dataset.py 逻辑一致)
                    del extra_obs['base_pos_wrt_world']
                    del agent_obs['qvel']
                    state_tensor = torch.cat([*agent_obs.values(), *extra_obs.values(), torch.zeros(2)])
                    
                    # e. 检查原始维度
                    if state_tensor.shape[0] != STATE_DIM:
                        print(f"错误: State 维度不匹配! 期望 {STATE_DIM}, 得到 {state_tensor.shape[0]}")
                        raise ValueError("State 维度不匹配")
                        
                    if act_tensor.shape[0] != ORIGINAL_ACTION_DIM:
                        print(f"错误: 原始 Action 维度不匹配! 期望 {ORIGINAL_ACTION_DIM}, 得到 {act_tensor.shape[0]}")
                        raise ValueError("原始 Action 维度不匹配")

                    # ===========================================================
                    # f. [新] 将 13D mshab action 转换为 16D pi0 action
                    #    (按照 7-DoF Arm + 1-DoF Gripper 映射)
                    # ===========================================================

                    # 你的 13D action 分解:
                    # act_tensor[0:7] (7D): 7-DoF Arm
                    # act_tensor[7]   (1D): Gripper
                    # act_tensor[8:11] (3D): Torso + Head (将被丢弃)
                    # act_tensor[11:13] (2D): Base
                    
                    # --- 1. 提取 7-DoF Arm ---
                    arm_action_7d = act_tensor[0:7] 
                    
                    # --- 2. 提取 Gripper ---
                    gripper_action_1d = act_tensor[7:8] # 用切片 [7:8] 来保持维度 (1D)
                    
                    # --- 3. 提取 Base ---
                    base_action_2d = act_tensor[11:13] # (2D)

                    # --- 4. 创建 0 填充 (用于 pi0 的副臂) ---
                    # (pi0[8:13] 共 6 个维度)
                    padding_6d = torch.zeros(6, dtype=act_tensor.dtype, device=act_tensor.device)
                    padding_6d_2 = torch.zeros(16, dtype=act_tensor.dtype, device=act_tensor.device)

                    # --- 5. 按 pi0 格式 (16D) 拼接 ---
                    # [ 7D Arm ] + [ 1D Gripper ] + [ 6D Padding ] + [ 2D Base ]
                    pi0_action_16d_tensor = torch.cat([
                        arm_action_7d,       # Dims 0-6
                        gripper_action_1d,   # Dim 7
                        padding_6d,          # Dims 8-13 (副臂填充)
                        base_action_2d,       # Dims 14-15 (底座)
                        padding_6d_2
                    ])

                    # --- 6. 检查最终维度 ---
                    if pi0_action_16d_tensor.shape[0] != TARGET_PI0_ACTION_DIM:
                        raise ValueError(f"最终 pi0 action 维度不是 {TARGET_PI0_ACTION_DIM}!")
                    # import ipdb
                    # ipdb.set_trace()
                    # g. 准备要存入 LeRobot 的数据帧
                    frame_data = {
                        "image": head_rgb_tensor.squeeze(0).cpu().numpy().astype(np.uint8),
                        "wrist_image": hand_rgb_tensor.squeeze(0).cpu().numpy().astype(np.uint8),
                        "state": state_tensor.cpu().numpy(),
                        "actions": pi0_action_16d_tensor.cpu().numpy(), 
                        "task": task_instruction, # 任务指令
                        # "timestamp": np.array(step_num, dtype=np.float32),     # <-- [新添加] 解决这个错误！
                    }
                    
                    # h. 添加帧
                    dataset.add_frame(frame_data)
                    
                except Exception as e:
                    print(f"错误: 处理 file {h5_path.name}, ep {ep_id}, step {step_num} 时出错: {e}")
                    # 跳过这个损坏的 step
                    pass
            
            # i. 在一个 episode 结束后保存
            dataset.save_episode()
            total_episodes += 1

         

        h5_file.close() # 处理完一个H5文件后关闭它
      
        
        print(f"文件 {h5_path.name} 处理完毕。累计 Episodes: {total_episodes}")

    # --- 6. (可选) 推送到 Hub ---
    if push_to_hub:
        print("正在将数据集推送到 Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["mshab", "fetch", "maniskill", "pi0_ready_7dof"], # TODO: 可选，添加你自己的标签
            private=False, # TODO: 如果是私有数据请设为 True
            push_videos=True,
            license="apache-2.0", # TODO: 选择你希望的许可证
        )

    print("\n--- 转换完成 ---")
    print(f"数据集已保存到: {output_path}")
    print(f"总共转换了 {total_episodes} 个 episodes。")


if __name__ == "__main__":
    import argparse
    # 1. 创建一个参数解析器
    parser = argparse.ArgumentParser(description="将 mshab 数据集转换为 LeRobot 格式 (pi0 兼容, 7-DoF 映射)。")

    # 2. 添加 --data_dir 参数
    parser.add_argument(
        "--data_dir",
        default="/raid/wenbo/project/mshab/mshab_exps/gen_data_save_trajectories/set_table/close/train/fridge",
        help="指向你的 H5 数据集的路径 (可以是一个目录或单个文件)。"
    )

    # 3. 添加 --push_to_hub 参数
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="如果设置了此标志, 则将数据集推送到 Hugging Face Hub。"
    )

    # 4. 解析从命令行传入的参数
    args = parser.parse_args()

    # 5. 使用解析到的参数来调用你的 main 函数
    main(data_dir=args.data_dir, push_to_hub=args.push_to_hub)