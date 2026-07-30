# CoLLM: Compositional Large Language Models - Experiment Guide

This repository contains the training pipeline and ablation configs for embedding visual concepts into LLM semantic spaces across CLIP, CoCa, and BLIP architectures.

---

## 1. Environment Setup & Requirements

```bash
# 1. Create and activate a fresh Conda environment
conda create -n collm python=3.10 -y
conda activate collm

# 2. Install dependencies (enforces PyTorch with CUDA 12.1 compatibility)
pip install -r requirements.txt

# 3. Optional: Configure offline WandB logging for cluster nodes
export WANDB_MODE=offline
export WANDB_DIR=~/wandb_logs
```

---

## 2. Global Hyperparameter Summary

All configuration files share an optimized baseline setting across training stages:
- **Global Batch Size:** 256 (128 per-device $\times$ 2 GPUs via Distributed Data Parallel).
- **Optimizer & LR:** AdamW with `learning_rate: 0.0001` and gradient checkpointing enabled.
- **Precision:** `bf16` mixed precision for compute and memory optimization.
- **LoRA Configurations:**
  - **LLM Adapter:** Rank $r=16, \alpha=16$, targeting `["q_proj", "k_proj", "v_proj", "o_proj"]` (Trainable across all runs).
  - **Vision Adapter:** Rank $r=64, \alpha=64$, targeting attention/projection layers (Toggled depending on experiment).
- **Loss Function:** Compositional InfoNCE loss with trainable temperature (initialized at $\tau=0.07$).
- **MTCIR Text Processing (`use_space_join`):** Defaults to `false` (randomly samples one modification text per step). Toggle to `true` in `configs/` to join all modification descriptions into a single dense sentence per transition.
- **Evaluation:** Automated zero-shot validation on FashionIQ (`val` split) every 500 steps.
---

## 3. Experiment Matrix Overview

| Configuration Variant | Stage | Vision LoRA | Initial Pretrained Weights | Purpose / Hypothesis |
| :--- | :---: | :---: | :---: | :--- |
| `*_stage1` | 1 (LLaVA 558k) | Enabled | None (Scratch adapters) | Aligns visual and LLM embedding spaces before task-specific tuning. |
| `*_stage2_after_stage1`| 2 (MTCIR) | Disabled | Best Stage 1 checkpoint | Standard two-stage pipeline; tunes compositional retrieval over aligned features. |
| `*_stage2_only_frozen` | 2 (MTCIR) | Disabled | None | Tests if LLM LoRA + MLP adapter alone can learn compositionality directly. |
| `*_stage2_only_unfrozen`| 2 (MTCIR) | Enabled | None | Tests if end-to-end tuning of vision + LLM directly on MTCIR beats two-stage curriculum. |

---

## 4. Execution Commands

### A. CLIP Experiments (OpenCLIP ViT-L-14)
```bash
# Stage 1: Pretraining on LLaVA 558k
accelerate launch train.py --config configs/clip_stage1.yaml

# Stage 2 (Two-Stage): Fine-tuning on MTCIR using Stage 1 checkpoint
accelerate launch train.py --config configs/clip_stage2_after_stage1.yaml

# Stage 2 (Direct / Frozen): Direct training on MTCIR with frozen vision encoder
accelerate launch train.py --config configs/clip_stage2_only_frozen.yaml

# Stage 2 (Direct / Tuned): Direct training on MTCIR with active Vision LoRA
accelerate launch train.py --config configs/clip_stage2_only_unfrozen.yaml
```

### B. CoCa Experiments (CoCa ViT-L-14)
```bash
# Stage 1: Pretraining on LLaVA 558k
accelerate launch train.py --config configs/coca_stage1.yaml

# Stage 2 (Two-Stage): Fine-tuning on MTCIR using Stage 1 checkpoint
accelerate launch train.py --config configs/coca_stage2_after_stage1.yaml

# Stage 2 (Direct / Frozen): Direct training on MTCIR with frozen vision encoder
accelerate launch train.py --config configs/coca_stage2_only_frozen.yaml

# Stage 2 (Direct / Tuned): Direct training on MTCIR with active Vision LoRA
accelerate launch train.py --config configs/coca_stage2_only_unfrozen.yaml
```

### C. BLIP Experiments (BLIP Image Captioning Large)
```bash
# Stage 1: Pretraining on LLaVA 558k
accelerate launch train.py --config configs/blip_stage1.yaml

# Stage 2 (Two-Stage): Fine-tuning on MTCIR using Stage 1 checkpoint
accelerate launch train.py --config configs/blip_stage2_after_stage1.yaml

# Stage 2 (Direct / Frozen): Direct training on MTCIR with frozen vision encoder
accelerate launch train.py --config configs/blip_stage2_only_frozen.yaml

# Stage 2 (Direct / Tuned): Direct training on MTCIR with active Vision LoRA (qkv/projection targets)
accelerate launch train.py --config configs/blip_stage2_only_unfrozen.yaml
```

---

## 5. Evaluation & Inference

You can run zero-shot or fine-tuned evaluation on either **FashionIQ (FIQ)** or **CIRCO** datasets using `eval.py`. The evaluation dataset and path are configured directly inside your YAML configuration files.

### Standard Checkpoint Evaluation
By default, the script evaluates the `last_checkpoint.pt` located in the config's output directory.
```bash
# Evaluates whatever dataset (fiq/circo) is defined in the config
python eval.py --config configs/clip_stage2_after_stage1.yaml
```

### Specific Checkpoint Evaluation (Pretrained or Custom)
If you want to evaluate the full model initialized with a **specific checkpoint** (e.g., evaluating a Stage 1 pretrained checkpoint, or an older epoch), pass the `--checkpoint` argument. This bypasses the default save directory and loads the model exactly as defined by the config + your passed weights.
```bash
python eval.py \
    --config configs/clip_stage1.yaml \
    --checkpoint /path/to/specific/checkpoint.pt
```

### Switching Datasets
To switch the evaluation target between FIQ and CIRCO, simply edit the `data:` block of your target config file:
```yaml
data:
  # To evaluate FashionIQ
  eval_dataset: fiq
  
  # To evaluate CIRCO (computes mAP@K and Semantic mAP@10 directly in the terminal)
  # eval_dataset: circo
  circo_dataset_dir: /home/anirban/yashwanthm/CIRCO
```
