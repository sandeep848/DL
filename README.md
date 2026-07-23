# Ghibli Market: Dual-Adapter LoRA Style-Tuning with Stable Diffusion 1.5

This project implements a dual-adapter Low-Rank Adaptation (LoRA) style-tuning pipeline to teach Stable Diffusion 1.5 a specific visual style (Studio Ghibli aesthetic) from a small dataset of reference images. Both the **UNet** and **Text Encoder** are adapted.

---

## Team Members
* **[Team Member 1]** - Deep Learning / CV Engineer
* **[Team Member 2]** - Deep Learning / CV Engineer
* **[Team Member 3]** - Deep Learning / CV Engineer
* *(Edit names in this file before final submission)*

---

## 1. Setup Instructions

To run the training and evaluation scripts, set up a virtual environment or conda environment and install the required dependencies.

### Option A: Using Conda (Recommended)
```bash
# Create a new conda environment
conda create -n ghibli-lora python=3.10 -y
conda activate ghibli-lora

# Install PyTorch with CUDA support (adjust CUDA version if necessary)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install project dependencies
pip install -r requirements.txt
```

### Option B: Using Pip Virtualenv
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/MacOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

## 2. Project Directory Layout
* `code/dataset.py`: Implements custom dataset loading, resizing, center-cropping, and style token tokenization.
* `code/train_lora.py`: Hand-crafted PyTorch training script with PEFT LoRA wrapping, custom token embedding updates, gradient isolation, gradient accumulation, and periodic visual validation.
* `code/eval_lora.py`: Load Stable Diffusion 1.5, add style token, apply the LoRA weights, and generate high-fidelity sample images.
* `requirements.txt`: Project package dependencies list.

---

## 3. Running Training

Train the dual-adapter LoRA using the custom training script:

```bash
python code/train_lora.py \
  --data_dir style_imgs/512 \
  --instance_token "<sks>" \
  --output_dir lora_out \
  --rank 8 \
  --max_steps 800 \
  --learning_rate 1e-4 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --validation_steps 200 \
  --overwrite
```

### Main CLI Arguments:
* `--data_dir`: Path to the training images folder (e.g. `style_imgs/512`).
* `--instance_token`: Unique identifier style token to add (e.g. `<sks>`).
* `--output_dir`: Path to save output weights and validation images.
* `--rank`: LoRA rank dimension (default: 8).
* `--max_steps`: Total training steps (default: 800).
* `--overwrite`: Overwrites the output safetensors file if it already exists.

---

## 4. Running Evaluation

Generate Ghibli-style images using the trained adapter:

```bash
python code/eval_lora.py \
  --weights lora_out/pytorch_lora_weights.safetensors \
  --prompt "a busy market, in <sks> style" \
  --outdir samples \
  --num_images 3 \
  --steps 50 \
  --guidance_scale 7.5 \
  --seed 123
```

### Main CLI Arguments:
* `--weights`: Path to the trained safetensors file containing LoRA weights + token embeddings.
* `--prompt`: Full text prompt including your style token (e.g. `a busy market, in <sks> style`).
* `--outdir`: Directory where the final output images are written.
* `--num_images`: Number of images to generate (default: 3).

---

## 5. Expected Resources & Runtime

The training loop supports Automatic Mixed Precision (AMP fp16) on GPU to minimize VRAM usage and maximize speed. Below are estimated system requirements and execution times.

### Training Performance Estimates (800 Steps)

| GPU Model | VRAM Required | Speed (sec / step) | Total Training Runtime |
| :--- | :---: | :---: | :---: |
| **NVIDIA A100 (40GB/80GB)** | ~7.2 GB | ~0.15s / step | **~2 minutes** |
| **NVIDIA RTX 3090 / 4090** | ~7.5 GB | ~0.25s / step | **~3.5 minutes** |
| **NVIDIA T4 (Google Colab)** | ~6.8 GB | ~0.95s / step | **~13 minutes** |
| **CPU Only (Dry-Run)** | ~8.0 GB (System RAM) | ~15.0s / step | ~3.3 hours (Not recommended for training) |

### Inference Performance (eval_lora.py)
* **GPU**: Generates 3 images (50 steps each) in **~5-10 seconds** total.
* **CPU**: Generates 3 images (50 steps each) in **~8-12 minutes** total.

---

## 6. How the Custom Token is Saved and Reloaded
To satisfy the requirements of reproducibility and ensure that the custom style token `<sks>` is retained:
1. During **training** (`train_lora.py`), we add `<sks>` to the tokenizer, resize the CLIP text encoder embeddings, and initialize `<sks>` with the weights of the anchor word `"style"`.
2. When training finishes, we save the trained embedding row corresponding to `<sks>` inside the final safetensors weights file under the key `text_encoder_embeddings`.
3. During **evaluation** (`eval_lora.py`), the script dynamically re-injects `<sks>` into the tokenizer, resizes the text encoder embeddings, and loads the saved tensor directly into the text encoder's embedding table at the correct index before applying the LoRA attention layers. This makes the `<sks>` token self-contained and reproducible.
