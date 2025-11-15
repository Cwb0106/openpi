import sys
import torch
import ipdb

# 添加您的 openpi 路径
sys.path.append("/raid/wenbo/project/openpi/")

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

# # --- 1. 获取 Config ---
# print("Loading config for pi0_base...")

# config = _config.get_config("pi0_base") # (已取消注释)


# # --- 2. 下载 Checkpoint ---
# print("Downloading checkpoint for pi0_base...")
# checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_base") #
# print("Download complete.")

# # --- 3. 创建 Policy ---
# print("Creating trained policy...")
# # 这将自动检测并加载 PyTorch checkpoints
# policy = policy_config.create_trained_policy(config, checkpoint_dir) #
# policy.model.to("cuda" if torch.cuda.is_available() else "cpu") # 确保模型在GPU上
# print("Policy created and moved to device.")

# # --- 4. 创建一个 *纯 PyTorch* 的模拟输入 ---
# print("Creating dummy PyTorch input for inference...")
# # 我们必须创建一个 *纯* PyTorch 字典
# # 我们使用您在脚本中定义的形状
IMG_SIZE = 128 #
JOINT_DIM = 32   #

# # 注意：所有值都应为 PyTorch Tensors
# example_input_torch = {
#     'observation/exterior_image_1_left': torch.randn(IMG_SIZE, IMG_SIZE, 3), #
#     'observation/wrist_image_left': torch.randn(IMG_SIZE, IMG_SIZE, 3), #
#     'observation/joint_position': torch.randn(JOINT_DIM), #
#     'observation/gripper_position': torch.randn(1), # <-- 修正：使用 torch.randn
#     'prompt': "do something" #
# }
# print("Dummy input created.")

# # --- 5. 运行 Inference ---
# print("Running inference...")
# # policy.infer() 会自动处理数据
# result = policy.infer(example_input_torch) #

# # --- 6. 打印结果 ---
# print("\n--- Inference Successful! ---")
# print("Actions shape:", result["actions"].shape) #

# # --- 7. 清理 ---
# del policy
# print("Policy deleted.")

example_input_torch = {
    'observation/image': torch.randn(4,IMG_SIZE, IMG_SIZE, 3), #
    'observation/wrist_image': torch.randn(4,IMG_SIZE, IMG_SIZE, 3), #
    'observation/state': torch.randn(4,JOINT_DIM), #
    # 'observation/gripper_position': torch.randn(1), # <-- 修正：使用 torch.randn
    'prompt': "open the fridge"#
}

config = _config.get_config("pi0_mshab_fetch")
checkpoint_dir = "/raid/wenbo/project/openpi/checkpoints/pi0_mshab_fetch/my_8gpu_run/7000"

# Create a trained policy (automatically detects PyTorch format)
policy = policy_config.create_trained_policy(config, checkpoint_dir)
print("1111")
# exit(1)

# Run inference (same API as JAX)
# action_chunk = policy.infer(example_input_torch)["actions"]
action_chunk = policy.infer_batched(example_input_torch)
# action_chunk_2 = policy.infer(example_input_torch)["actions"]
ipdb.set_trace()
print("2222")










