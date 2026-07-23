# 🚀 Step-by-Step Execution Guide

This guide describes how to run the Ghibli Market LoRA Style-Tuning pipeline, from setup to final deliverables.

---

## Step 1: Environment Setup
Set up the python environment on your GPU runtime (e.g., Google Colab, Kaggle, or a cloud server).

### Option A: Using Conda (Recommended)
```bash
# Create and activate environment
conda create -n ghibli-lora python=3.10 -y
conda activate ghibli-lora

# Install PyTorch with CUDA support (adjust version matching your GPU driver)
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

## Step 2: Run Training
Start training on your GPU instance. Training on a standard T4 GPU takes about **13 minutes** for 800 steps.

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

### Main training parameters:
* `--data_dir`: Path to the pre-cropped Ghibli training images.
* `--instance_token`: The unique trigger word (`<sks>`) injected for style activation.
* `--output_dir`: Path where final weights and validation samples are saved.
* `--rank`: Dimension of LoRA layers (default: 8).
* `--max_steps`: Number of training optimization steps (default: 800).
* `--overwrite`: Forces writing over existing safetensors weights.

---

## Step 3: Run Evaluation
After training completes, generate sample images using the evaluation script. This will load the custom adapter and render the requested images in the `samples/` directory:

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

### Main evaluation parameters:
* `--weights`: Path to the trained safetensors file.
* `--prompt`: Full prompt containing the custom token (`a busy market, in <sks> style`).
* `--outdir`: Directory to save the output files.
* `--num_images`: Number of generated variations (renders 3 images by default).
* `--seed`: Generator seed to ensure reproducible image generation.

---

## Step 4: Finalize Deliverables
Before compiling your final submission:
1. **Add Team Names**: Edit `README.md` to add the names of your team members.
2. **Compile Report PDF**: Open `report.md` in VS Code, right-click, and select **Markdown PDF: Export (pdf)**.
3. **ZIP Archive**: Package the folder into a single ZIP file containing:
   - `lora_out/pytorch_lora_weights.safetensors`
   - `code/train_lora.py`
   - `code/eval_lora.py`
   - `samples/` (with 3 generated images)
   - `requirements.txt`
   - `README.md`
   - `report.pdf`
