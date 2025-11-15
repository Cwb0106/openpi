import dataclasses
import ipdb
import jax
import numpy as np

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_debug_mshab_dataset():
    """
    仿照 test_with_real_dataset 为 pi0_mshab_fetch 编写的调试函数。
    """
    
    # 1. 加载你的 "pi0_mshab_fetch" 配置
    #    (这会从 src/openpi/training/config.py 加载)
    config_name = "pi0_mshab_fetch"
    print(f"正在加载配置 '{config_name}'...")
    try:
        config = _config.get_config(config_name)
    except ValueError as e:
        print(f"\n--- 错误! ---")
        print(f"无法加载配置 '{config_name}'。")
        print(f"请确保你已经在 src/openpi/training/config.py 中正确定义了它。")
        print(f"报错信息: {e}")
        return

    # 2. 覆盖 batch_size 以便快速测试
    test_batch_size = 4
    config = dataclasses.replace(config, batch_size=test_batch_size)
    print(f"已设置测试 batch_size = {test_batch_size}")

    # 3. 创建数据加载器 (使用 JAX 风格的加载器，与你的示例一致)
    #    - skip_norm_stats=True 
    #      (关键：这允许我们在没有 norm.json 文件时也能运行)
    #    - num_batches=2 (只取 2 个批次)
    print("正在创建数据加载器 (skip_norm_stats=True)...")
    loader = _data_loader.create_data_loader(
        config,
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )

    # 4. 检查 repo_id
    #    (这应与你在 config.py 中为 LeRobotMshabDataConfig 设置的 repo_id 一致)
    expected_repo_id = "mshab_fetch_dataset" 
    print(f"检查 data config... 期望 repo_id: {expected_repo_id}")
    assert loader.data_config().repo_id == expected_repo_id

    # 5. 从加载器中拉取所有批次
    print("正在拉取 2 个批次的数据...")
    # (如果 lerobot 的 Column/list bug 未修复, 此处可能会失败)
    batches = list(loader)
    print(f"成功拉取 {len(batches)} 个批次。")

    # 6. 检查批次数量
    assert len(batches) == 2

    # 7. 循环检查每个批次的数据形状
    print("正在检查每个批次的 'observations' 和 'actions' 形状...")
    
    # 从 config 中获取我们期望的维度
    expected_action_dim = config.model.action_dim
    expected_action_horizon = config.model.action_horizon
    
    # [关键] 检查 state_dim 是否被正确设置

    # expected_state_dim = config.model.state_dim
   

    for i, (observations, actions) in enumerate(batches):
        print(f"--- 批次 {i+1} ---")
        
        # 检查 Actions
        # 形状: (batch_size, action_horizon, action_dim)
        expected_action_shape = (test_batch_size, expected_action_horizon, expected_action_dim)
        print(f"  检查 Actions... 期望: {expected_action_shape}, 得到: {actions.shape}")
        
        # JAX 加载器返回 JAX 数组 (jnp), 转换为 numpy 检查

        assert np.array(actions).shape == expected_action_shape
        print(f"  Actions 形状正确!")

        # 检查 State
        # 形状: (batch_size, state_dim)
        # expected_state_shape = (test_batch_size, expected_state_dim)
        # print(f"  检查 State... 期望: {expected_state_shape}, 得到: {observations['state'].shape}")
        ipdb.set_trace()
        assert "state" in observations
        assert np.array(observations["state"]).shape == expected_state_shape
        print(f"  State 形状正确!")
        
        print(f"批次 {i+1} 形状正确！")

    print("\n" + "="*50)
    print("✅  调试测试通过！")
    print("数据加载和形状配置全部正确。")
    print(f"(state_dim={expected_state_dim}, action_dim={expected_action_dim})")
    print("="*50)


if __name__ == "__main__":
    test_debug_mshab_dataset()