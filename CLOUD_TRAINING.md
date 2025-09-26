# Cloud/GPU Training Guide

If CUDA isn't available locally, run training in the cloud with a GPU (A10/A30/A100 or 4090).

## Options
- Colab Pro/Pro+: Quick start, limited by session.
- Kaggle Notebooks: Free A100 (limited), persistent datasets.
- Runpod/AWS/GCP/Azure: Pay-as-you-go GPUs, full control.
- Vast.ai: Market-priced GPUs.

## One-time setup (Linux VM)
1) Install system deps and Miniconda (optional). Use Python 3.10.
2) Clone your fork and install requirements.

```bash
# On the remote GPU machine
sudo apt-get update -y && sudo apt-get install -y git
git clone https://github.com/Jbrower1991/helloLondon.git
cd helloLondon
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'device count:', torch.cuda.device_count())
PY
```

## Get your data
- Option A: Rebuild corpus on the GPU box (recommended, it's automated):
```bash
python 02_data_collection/historical_data_collector.py --max_sources 10
```
- Option B: Upload a prepared corpus (avoid syncing raw data to git):
  - Copy `data/london_historical/london_historical_corpus_comprehensive.txt` via scp/rsync.

## Train tokenizer
```bash
python 03_tokenizer/train_historical_tokenizer.py
```

## Train SLM (small model)
CPU-friendly flags now exist for smoke tests, but on GPU you can keep defaults or tune:
```bash
python 04_training/train_model_slm.py \
  --tokenizer_dir 09_models/tokenizers/london_historical_tokenizer \
  --data_dir data/london_historical \
  --output_dir 09_models/checkpoints/slm
```

Useful overrides:
- `--batch_size 8` (per-GPU)
- `--block_size 512` or `1024`
- `--max_iters 20000` (or more)
- `--eval_interval 1000` `--logging_steps 50`
- `--resume_from_checkpoint 09_models/checkpoints/slm/checkpoint-<n>.pt`

## Multi-GPU (optional)
```bash
torchrun --nproc_per_node=2 04_training/train_model_slm.py \
  --tokenizer_dir 09_models/tokenizers/london_historical_tokenizer \
  --data_dir data/london_historical \
  --output_dir 09_models/checkpoints/slm
```

## Inference after training
```bash
python 04_training/test_slm_checkpoint.py --checkpoint 09_models/checkpoints/slm/checkpoint-<n>.pt
```

## Tips
- Keep large data and artifacts out of git (see .gitignore).
- For WANDB, set env `WANDB_API_KEY` or disable in `config.py`.
- Prefer A30/A100 for faster tokenizer + training.
