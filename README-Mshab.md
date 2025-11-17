# openpi-mshabg

convert mshab dataset -> lerobot

```bash
python examples/mshab/convert.py  # maybe you need to modifed some param (e.g. name)
```

compute norm state

```bash
python scripts/compute_norm_stats.py --config-name pi0_mshab_fetch
```

finetuning

```bash
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train_pytorch.py pi0_mshab_fetch --exp_name pytorch_ddp_test --overwrite
```

eval

```bash
python GAMobile/eval.sh
```


Some files you may need to modify
1. src/openpi/policies/mshab_policy.py
2. src/openpi/training/config.py
3. src/openpi/policies/policy.py