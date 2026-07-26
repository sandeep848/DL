# Ghibli Market: Dual-Adapter LoRA Style-Tuning with Stable Diffusion 1.5

This project implements a dual-adapter Low-Rank Adaptation (LoRA) style-tuning pipeline to teach Stable Diffusion 1.5 a specific visual style (Studio Ghibli aesthetic) from a small dataset of reference images. Both the **UNet** and **Text Encoder** are adapted, and a custom style token `<sks>` is trained using a gradient-masked optimization path.

---

## Team Members
* **Hesham Abdalla** (hesham.abdalla@utn.de)
* **Jan Kobiolka** (jan.kobiolka@utn.de)

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
* `code/dataset.py`: Implements custom dataset loading, resizing, center-cropping, data augmentations (flips/jitter), subject-agnostic prompt templates, and **random template selection** to prevent text-image concept mismatch.
* `code/train_lora.py`: PyTorch training script with PEFT LoRA wrapping on UNet and Text Encoder, style token initialization from the `"style"` anchor word, and an **automatic PyTorch backward tensor hook** to isolate gradients to `<sks>`.
* `code/eval_lora.py`: Load Stable Diffusion 1.5, add style token, load the custom embedding vector, apply the LoRA weights, and generate images matching the **exact prompt** passed via arguments.
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

Generate Ghibli-style images using the trained adapter. The prompt is fully dynamic and can be changed to any subject:

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
* `--prompt`: Full text prompt including the trigger token (e.g. `a busy market, in <sks> style`).
* `--outdir`: Directory where the final output images are written.
* `--num_images`: Number of images to generate (default: 3).

---

## 5. Key Architecture Enhancements

### 1. Concept Alignment & Subject-Agnostic Dataset Prompts
In standard style-tuning, training images are often paired with random template captions. In alphabetically sorted datasets, this causes mismatched pairings where landscape images are trained on texts like `"character portrait"`—harming the model's vocabulary. 
To resolve this:
* We cycle through **subject-agnostic style templates** (e.g., `"a painting, in <sks> style"`, `"a 2D anime illustration, in <sks> style"`).
* The dataset loader **randomly selects** a template for each image on every epoch, ensuring a clean style representation without forcing incorrect subject associations.

### 2. PyTorch Tensor Hook for Gradient Masking
Training a new vocabulary row normally requires freezing the rest of the embedding table manually at each step. 
* We register a **PyTorch backward gradient hook** (`register_hook`) directly on the text encoder's expanded embedding weight tensor.
* During the backward pass, PyTorch automatically zero-flops all gradients for standard words, leaving only the gradient of `<sks>` active before they reach the optimizer. This isolates updates cleanly without cluttering the training loop.
