import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_mshab_example() -> dict:
    """Creates a random input example for the Mshab policy."""
    # 你的 state 是 42 维, 原始 action 是 13 维
    return {
        "observation/state": np.random.rand(42), 
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "open the fridge",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class MshabInputs(transforms.DataTransformFn):
    """
    将 mshab 数据集（Fetch 机器人）的输入转换为 pi0 模型期望的格式。
    用于训练和推理。
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        action_dim = 32
        state_dim = 32
        # 1. 解析图像 (与 Libero 相同)
        # 假设你的 LeRobot 数据集键是 'image' 和 'wrist_image'
        # 并且 config.py 中的 repack_transform 将它们映射到了 'observation/image'
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # 2. 构建模型输入字典

        batch_size = data["observation/state"].shape[0]


        # # inference
        # inputs = {
        #     "state": transforms.pad_to_dim(data["observation/state"], state_dim), 
        #     "image": {
        #         "base_0_rgb": base_image,
        #         "left_wrist_0_rgb": wrist_image,
        #         "right_wrist_0_rgb": np.zeros_like(base_image), # pi0 期望3个图像，用0填充
        #     },
        #     "image_mask": {
        #         # [!!! 关键修复 3: 创建批处理过的 Mask !!!]
        #         "base_0_rgb": np.full(batch_size, True),
        #         "left_wrist_0_rgb": np.full(batch_size, True),
        #         "right_wrist_0_rgb": np.full(
        #             batch_size, 
        #             True if self.model_type == _model.ModelType.PI0_FAST else False
        #         ),
        #     },
        # }

        # training
        inputs = {
            # 你的 state 是 32 维
            "state": transforms.pad_to_dim(data["observation/state"], state_dim), 
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image), # pi0 期望3个图像，用0填充
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # 3. [关键] 处理 Actions (仅在训练时)
        if "actions" in data:
            inputs["actions"] = data["actions"]
            # inputs["actions"] = transforms.pad_to_dim(data["actions"], action_dim)


        # 4. 处理 Prompt
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class MshabOutputs(transforms.DataTransformFn):
    """
    [已修复] 将 pi0 模型的 32D 动作块 (B, H, 32) 
    转换回 mshab (Fetch) 环境期望的 13D 动作块 (B, H, 13)。
    """

    def __call__(self, data: dict) -> dict:
        # data["actions"] 是 pi0 输出的 32D 动作块 (B, H, 32)
        pi0_action_32d_chunk = np.asarray(data["actions"])
        
        # (B, H, 32)
        
        # [!!! 关键修复 !!!]
        # 我们不再只取 [:, 0, :]。我们保留 H 维度。

        # 你的 32D 动作格式 (根据 convert.py):
        # [ 7D Arm ] + [ 1D Gripper ] + [ 6D Pad ] + [ 2D Base ] + [ 16D Pad ]
        
        # --- 1. 提取 7-DoF Arm ---
        arm_action_7d = pi0_action_32d_chunk[:, :, 0:7]
        
        # --- 2. 提取 Gripper ---
        gripper_action_1d = pi0_action_32d_chunk[:, :, 7:8]
        
        # --- 3. 提取 Base ---
        base_action_2d = pi0_action_32d_chunk[:, :, 14:16]
        
        # --- 4. 创建 Torso/Head 的 0 填充 ---
        # (Torso(1) + Head(2) = 3)
        # 获取 B (批次) 和 H (Horizon)
        batch_size = pi0_action_32d_chunk.shape[0]
        horizon = pi0_action_32d_chunk.shape[1]
        
        torso_head_padding_3d = np.zeros(
            (batch_size, horizon, 3), 
            dtype=pi0_action_32d_chunk.dtype
        )
        
        # --- 5. 拼接成 13D 动作块 (B, H, 13) ---
        mshab_action_13d_chunk = np.concatenate([
            arm_action_7d,         # Dims 0-6
            gripper_action_1d,     # Dim 7
            torso_head_padding_3d, # Dims 8-10 (Torso/Head)
            base_action_2d         # Dims 11-12 (Base)
        ], axis=-1)

        # 返回与环境匹配的动作块字典
        return {"actions": mshab_action_13d_chunk}

# @dataclasses.dataclass(frozen=True)
# class MshabOutputs(transforms.DataTransformFn):
#     """
#     将 pi0 模型的 16D 输出转换回 mshab (Fetch) 环境期望的 13D 动作。
#     仅用于推理。
#     """

#     def __call__(self, data: dict) -> dict:
#         # data["actions"] 是 pi0 输出的 16D 动作 (Batch, Horizon, 16)
#         # 我们只关心第一个动作 (Batch, 16)
#         # 注意: Libero policy 示例中是 [:, :7], 它假设了 (Batch, Dim)
#         # 我们假设 (Batch, Horizon, Dim)
        
#         pi0_action_16d = np.asarray(data["actions"]) # (B, H, 16)

#         # 我们只转换第一个动作步 (B, 1, 16) -> (B, 16)
#         if pi0_action_16d.ndim == 3:
#             pi0_action_16d = pi0_action_16d[:, 0, :]
        
#         # (B, 16)
        
#         # --- 1. 提取 7-DoF Arm ---
#         arm_action_7d = pi0_action_16d[:, 0:7]
        
#         # --- 2. 提取 Gripper ---
#         gripper_action_1d = pi0_action_16d[:, 7:8]
        
#         # --- 3. 提取 Base ---
#         base_action_2d = pi0_action_16d[:, 14:16]
        
#         # --- 4. 创建 Torso/Head 的 0 填充 ---
#         # (Torso(1) + Head(2) = 3)
#         torso_head_padding_3d = np.zeros((pi0_action_16d.shape[0], 3), dtype=pi0_action_16d.dtype)
        
#         # --- 5. 拼接成 13D 动作 ---
#         # [ 7D Arm ] + [ 1D Gripper ] + [ 3D Padding ] + [ 2D Base ]
#         mshab_action_13d = np.concatenate([
#             arm_action_7d,         # Dims 0-6
#             gripper_action_1d,     # Dim 7
#             torso_head_padding_3d, # Dims 8-10 (Torso/Head)
#             base_action_2d         # Dims 11-12 (Base)
#         ], axis=-1)

#         # 返回与环境匹配的字典
#         return {"actions": mshab_action_13d}