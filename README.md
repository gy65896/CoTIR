<div align="center">
<img align="left" width="180" height="180" src="https://github.com/gy65896/CoTIR/blob/main/img/cotir.png" alt="">
    
## [Arxiv 2026] Universal Image Restoration via Internalized<br> Chain-of-Thought Reasoning

[![ArXiv](https://img.shields.io/badge/CoTIR-ArXiv-red.svg)]()
[![Paper](https://img.shields.io/badge/CoTIR-Paper-yellow.svg)]()
[![Web](https://img.shields.io/badge/CoTIR-Web-blue.svg)](https://gy65896.github.io/projects/ArXiv2026_CoTIR/index.html)
 
</div>

<div align=center>
<img src="https://github.com/gy65896/CoTIR/blob/main/img/abstract.jpg" width="720">
</div>

---
---
>**Universal Image Restoration via Internalized Chain-of-Thought Reasoning**<br> [Yu Guo](https://scholar.google.com/citations?user=klYz-acAAAAJ&hl=zh-CN)<sup>† </sup>, [Zhengru Fang](https://scholar.google.com/citations?user=yggQMJMAAAAJ)<sup>† </sup>, [Shengfeng He](https://scholar.google.com/citations?user=rBWnK8wAAAAJ), [Senkang Hu](https://scholar.google.com/citations?user=rtPVwT8AAAAJ), [Yihang Tao](https://scholar.google.com/citations?user=YopoapwAAAAJ), [Phone Lin](https://www.csie.ntu.edu.tw/~plin/), [Yuguang Fang](https://scholar.google.com/citations?user=cs45mqMAAAAJ)<br>
(† Co-first Author)<br>
>Arxiv<br>

> **Abstract:** *Image restoration seeks to recover high-quality images from degraded inputs but becomes highly ill-posed under complex, mixed degradations. While unified all-in-one models are common, their performance declines as degradation complexity increases. Recent works adopt Chain-of-Thought (CoT) reasoning for multi-round restoration using specialized modules. However, this approach faces two key limitations: (i) increased computational cost due to multi-step processing and (ii) weak modeling of interactions between degradations during stepwise inference. We introduce CoTIR, a universal image restoration framework that internalizes CoT reasoning within a single model. Concretely, we view image restoration as a specialized subtask of image editing, which implies that a large-scale pre-trained editing model provides a more favorable optimization starting point. Building on this, we fine-tune the model for restoration and further encode structured CoT-style reasoning into the learning objective via a differentiable formulation inspired by Lagrangian optimization, enabling holistic restoration without chaining specialized restorers. To facilitate training and evaluation, we further present CoTIR-Bench, a large-scale benchmark comprising 5.2 million samples with CoT-style reasoning traces. Extensive experiments on CoTIR-Bench and broad real composite degradation scenes show that CoTIR achieves stronger perceptual quality and more competitive fidelity than both all-in-one models and multi-round restoration methods.*
---

## News 🚀
* **2026.06.16**: Code is released.


## Method Architecture

</div>
<div align=center>
<img src="https://github.com/gy65896/CoTIR/blob/main/img/method.jpg" width="1080">
</div>

## Quick Start

### Install

- python 3.10
- cuda 12.1

```
# git clone this repository
git clone https://github.com/gy65896/CoTIR.git
cd CoTIR

# create new anaconda env
conda env create -f environment.yml
```

### Pretrained Models

CoTIR checkpoints only contain the **LoRA** and **CoT adapter** weights. You also need to download the corresponding base FLUX models from Black Forest Labs (accept the license on Hugging Face before downloading):

| CoTIR Model | Base Model (required) |
|-------------|------------------------|
| CoTIR-12B | [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) |
| CoTIR-4B | [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) |
| CoTIR-9B | [FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) |

Download the base models:

```bash
git clone https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
git clone https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
git clone https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
```

Download the [CoTIR LoRA and adapter weights](https://huggingface.co/gy65896/CoTIR) and place them under `./ckpt/`:

```bash
git clone https://huggingface.co/gy65896/CoTIR
```

Before running inference or testing, update the checkpoint paths in `configs/test_cotir-4b.yaml`, `configs/test_cotir-9b.yaml`, or `configs/test_cotir-12b.yaml`:

```yaml
inference:
  num_steps: 5
  base_model_path: /path/to/FLUX.2-klein-9B      # base FLUX model
  cot_adapter_path: ./ckpt/CoTIR-9B/cot_adapter.pt  # CoTIR adapter
  lora_path: ./ckpt/CoTIR-9B/lora.pt                # CoTIR LoRA
```

| Model | Config |
|-------|--------|
| CoTIR-12B | `configs/test_cotir-12b.yaml` |
| CoTIR-4B | `configs/test_cotir-4b.yaml` |
| CoTIR-9B | `configs/test_cotir-9b.yaml` |

### Train

CoTIR-Bench will be released soon. After it is available, update `data.data_dir` and `model.base_model_path` in the training config, then run:

```bash
python train.py --config configs/train_cotir-9b.yaml
```

Multi-GPU training (example with 8 GPUs):

```bash
accelerate launch --num_processes 8 train.py --config configs/train_cotir-9b.yaml
```

Available training configs: `configs/train_cotir-4b.yaml`, `configs/train_cotir-9b.yaml`, `configs/train_cotir-12b.yaml`.

Checkpoints are saved to `saves/` by default. Set `resume: latest` in the config to resume from the latest checkpoint.

### Test

Batch inference on a benchmark JSONL file. Outputs are saved to `vague/` (vague prompt) and `precise/` (precise prompt):

```bash
python test.py \
  --model CoTIR-9B \
  --json_path ./test/test.jsonl \
  --lq_dir ./test/lq \
  --output_root ./results
```

Optional arguments:

```bash
python test.py \
  --config configs/test_cotir-9b.yaml \
  --json_path ./test/test.jsonl \
  --lq_dir ./test/lq \
  --output_root ./results \
  --batch_size 1 \
  --max_long_edge 1024 \
  --max_samples 100 \
  --skip_existing
```

### Evaluation

Evaluation scripts and instructions will be released soon.

> **Note:** Locally reproduced results may show minor deviations from the numbers reported in the paper. To obtain the exact results used in our evaluation, please download the pre-computed outputs from the corresponding branches of [CoTIR-Bench](https://huggingface.co/datasets/gy65896/CoTIR-Bench):
>
> | Paper Table | Branch | Download Command |
> |-------------|--------|------------------|
> | Table 2 | `methods` | `git clone -b methods --single-branch https://huggingface.co/datasets/gy65896/CoTIR-Bench` |
> | Table 3 | `single` | `git clone -b single --single-branch https://huggingface.co/datasets/gy65896/CoTIR-Bench` |
> | Table 4 | `cotir-bench-7d` | `git clone -b cotir-bench-7d --single-branch https://huggingface.co/datasets/gy65896/CoTIR-Bench` |

### Inference

Single-image restoration:

```bash
python inference.py \
  --config configs/test_cotir-9b.yaml \
  --lq ./path/to/lq.png \
  --prompt "Make this picture clearer." \
  --output_root outputs \
  --num_steps 5 \
  --seed 42
```

You can also pass a prompt text file:

```bash
python inference.py \
  --config configs/test_cotir-9b.yaml \
  --lq ./samples/images/000018.png \
  --prompt ./samples/prompts/000018.txt \
  --output_root outputs
```

### Inference with Gradio

Launch the interactive demo:

```bash
python inference_gradio.py --port 7862
```

Create a public link (optional):

```bash
python inference_gradio.py --port 7862 --share
```

Usage:
1. Enter the GPU id(s) to use and click **Init Model**.
2. Select a model (`CoTIR-4B` / `CoTIR-9B` / `CoTIR-12B`).
3. Upload an LQ image, optionally upload an HQ reference image, and enter a prompt.
4. Adjust `num_steps` and other options if needed, then click **Run Inference**.

Results are saved under the output directory shown in the UI.


### Citation

```

```