# Ghibli Market: LoRA Style-Tuning with Stable Diffusion 1.5

## Team Members

- Mahalakshmi Jayaraman
- Sandeep

---

## Project Overview

This project is part of **Deep Learning Project 4: Ghibli Market LoRA Style-Tuning with Stable Diffusion 1.5**.

The goal of this project is to fine-tune **Stable Diffusion 1.5** using **LoRA** so that the model can generate Ghibli-style busy market images.

A custom style token `<sks>` is added to the tokenizer and used in the prompt:

```text
a busy market, in <sks> style
```

The model learns the visual style from the provided Ghibli-style market image dataset. The final trained LoRA adapter is saved as:

```text
lora_out/pytorch_lora_weights.safetensors
```

---

## Dataset

The provided dataset is organized inside the `style_imgs/` folder.

```text
style_imgs/
├── original/   # Full-resolution PNG/JPG reference images
└── 512/        # 512 x 512 cropped images used for training
```

For training, we use:

```text
style_imgs/512/
```

because these images are already resized/cropped to 512 × 512 and are ready for LoRA training.

---

## Project Structure

The project folder is organized as follows:

```text
ghibli_lora_project/
├── code/
│   ├── train_lora.py
│   └── eval_lora.py
├── style_imgs/
│   ├── original/
│   └── 512/
├── lora_out/
│   └── pytorch_lora_weights.safetensors
├── samples/
├── requirements.txt
├── README.md
└── report.pdf
```

---

## Tasks Completed So Far

- Created the project folder structure.
- Added the provided dataset into:
  - `style_imgs/original/`
  - `style_imgs/512/`
- Created a Python virtual environment.
- Installed the required libraries.
- Updated the NVIDIA driver to enable CUDA.
- Verified GPU support with PyTorch.
- Created the training script `code/train_lora.py`.
- Successfully tested the training script for 20 steps.
- Started full training for 800 steps.
- Fixed the FP16 gradient issue by keeping trainable LoRA parameters in float32.

---

## Work Division

| Task | Person Responsible | Description |
|---|---|---|
| Dataset organization | Mahalakshmi | Organize the dataset into `style_imgs/original/` and `style_imgs/512/`. |
| Environment setup | Mahalakshmi | Create virtual environment and install required libraries. |
| CUDA/GPU setup | Mahalakshmi | Update NVIDIA driver and verify CUDA availability. |
| Training script | Mahalakshmi | Implement `code/train_lora.py` with tokenizer update, UNet LoRA, and text encoder LoRA. |
| LoRA training | Mahalakshmi | Run 20-step test training and full 800-step training. |
| Training debugging | Mahalakshmi | Fix CUDA, dependency, and FP16 gradient issues. |
| Evaluation script | Sandeep | Implement `code/eval_lora.py` to load the trained LoRA adapter and generate images. |
| Sample generation | Sandeep | Generate at least 3 output images and save them inside `samples/`. |
| README documentation | Both | Document project overview, setup commands, training command, evaluation command, and work division. |
| Final report | Both | Prepare the final report with method, hyperparameters, results, and limitations. |
| Final submission check | Both | Verify all required files are included in the final ZIP/tar.gz file. |

---

# Setup and Run Instructions

## 1. Create Project Folder

Open PowerShell and go to the location where the project should be stored.

Example:

```powershell
cd D:\Maha_lap\UTN\DL\assignment4
```

Create the project folder:

```powershell
mkdir ghibli_lora_project
cd ghibli_lora_project
```

Create the required folders:

```powershell
mkdir code
mkdir style_imgs
mkdir style_imgs\original
mkdir style_imgs\512
mkdir lora_out
mkdir samples
```

---

## 2. Add Dataset

Copy the provided dataset images into the following folders:

```text
style_imgs/original/
style_imgs/512/
```

The folder used for training is:

```text
style_imgs/512/
```

To check whether the images are present, run:

```powershell
dir style_imgs\512
```

You should see image files such as:

```text
.png
.jpg
.jpeg
```

---

## 3. Create Python Virtual Environment

This project was tested with:

```text
Python 3.10.11
```

Check the Python version:

```powershell
python --version
```

Create a virtual environment:

```powershell
python -m venv ghibli_lora_env
```

Activate the environment:

```powershell
.\ghibli_lora_env\Scripts\Activate.ps1
```

If PowerShell blocks activation with a script execution error, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

Then activate the environment again:

```powershell
.\ghibli_lora_env\Scripts\Activate.ps1
```

After activation, the terminal should show:

```text
(ghibli_lora_env)
```

---

## 4. Install Required Libraries

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install PyTorch with CUDA 12.1 support:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining dependencies:

```powershell
pip install diffusers transformers accelerate peft safetensors pillow tqdm
```

---

## 5. Check CUDA/GPU Support

Run:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

Expected output on our system:

```text
2.5.1+cu121
True
NVIDIA GeForce GTX 1650
```

If `torch.cuda.is_available()` returns `False`, update the NVIDIA driver and restart the laptop.

---

## 6. Training

The training script is located at:

```text
code/train_lora.py
```

The training script performs the following steps:

1. Loads Stable Diffusion 1.5.
2. Adds the custom token `<sks>` to the tokenizer.
3. Resizes the text encoder embeddings.
4. Adds LoRA adapters to the UNet.
5. Adds LoRA adapters to the text encoder.
6. Trains on the images from `style_imgs/512/`.
7. Saves the trained LoRA weights to `lora_out/pytorch_lora_weights.safetensors`.

---

## 7. Short Test Training

Before full training, run a short 20-step test:

```powershell
python code\train_lora.py `
  --data_dir style_imgs\512 `
  --instance_token "<sks>" `
  --output_dir lora_out `
  --rank 8 `
  --max_steps 20 `
  --overwrite
```

After the test finishes, check the output folder:

```powershell
dir lora_out
```

You should see:

```text
pytorch_lora_weights.safetensors
```

---

## 8. Full Training

Before full training, remove the test output:

```powershell
Remove-Item lora_out\* -Force
```

Run full training for 800 steps:

```powershell
python code\train_lora.py `
  --data_dir style_imgs\512 `
  --instance_token "<sks>" `
  --output_dir lora_out `
  --rank 8 `
  --max_steps 800 `
  --overwrite
```

After training, the final LoRA weights will be saved as:

```text
lora_out/pytorch_lora_weights.safetensors
```

---

## 9. Evaluation

The evaluation script is located at:

```text
code/eval_lora.py
```

The script loads the base Stable Diffusion 1.5 model and the trained LoRA adapter.

It generates images using the prompt:

```text
a busy market, in <sks> style
```

Run evaluation:

```powershell
python code\eval_lora.py `
  --weights lora_out\pytorch_lora_weights.safetensors `
  --prompt "a busy market, in <sks> style" `
  --outdir samples
```

Generated images will be saved inside:

```text
samples/
```

At least three generated images should be included in the final submission.

---

## 10. Hyperparameters Used

| Parameter | Value |
|---|---|
| Base model | `runwayml/stable-diffusion-v1-5` |
| Instance token | `<sks>` |
| Training prompt | `a busy market, in <sks> style` |
| Resolution | 512 |
| LoRA rank | 8 |
| Learning rate | 1e-4 |
| Max training steps | 800 |
| Batch size | 1 |
| Gradient accumulation steps | 4 |

---

## 11. Expected Hardware and Runtime

This project was tested on:

```text
OS: Windows
Python: 3.10.11
PyTorch: 2.5.1+cu121
GPU: NVIDIA GeForce GTX 1650
CUDA available: True
```

A CUDA-supported GPU is recommended.

Training on CPU is not recommended because Stable Diffusion LoRA training is very slow without GPU acceleration.

On a GTX 1650, full 800-step LoRA training may take a few hours depending on system performance and GPU memory.

---

## 12. Requirements

The required Python libraries are:

```text
torch
torchvision
torchaudio
diffusers
transformers
accelerate
peft
safetensors
pillow
t tqdm
```

Use the following command to create `requirements.txt` manually:

```powershell
pip freeze > requirements.txt
```

---

## 13. Final Submission Structure

The final ZIP or tar.gz file should contain:

```text
lora_out/pytorch_lora_weights.safetensors
code/train_lora.py
code/eval_lora.py
samples/
requirements.txt
README.md
report.pdf
```

---

## 14. Notes

The custom token `<sks>` represents the learned Ghibli-style market appearance.

The prompt used for training and evaluation is:

```text
a busy market, in <sks> style
```

The final model should generate busy market scenes with a soft illustrated appearance, warm colors, and visual characteristics similar to the provided reference images.