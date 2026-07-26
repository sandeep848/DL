# Project 4 Report: Ghibli Market Style-Tuning with Stable Diffusion 1.5
**Course**: Deep Learning: Project 4  
**Authors**: Hesham Abdalla (hesham.abdalla@utn.de), Jan Kobiolka (jan.kobiolka@utn.de)  
**Date**: July 2026  

---

## 1. Introduction and Objective
This project implements a Low-Rank Adaptation (LoRA) style-tuning pipeline to adapt Stable Diffusion 1.5 to generate images mimicking the aesthetic of Studio Ghibli films (e.g., *My Neighbor Totoro*, *Spirited Away*, *Howl's Moving Castle*). The assignment requires training a dual-adapter LoRA (adapting both the UNet and the Text Encoder) and adding a custom style token `<sks>` to control style activation via prompts like `"a busy market, in <sks> style"`.

---

## 2. Methodology

### 2.1 Task 1: Style Token `<sks>` Injection and Initialization
To activate the custom Ghibli style, we inject a new token `<sks>` into the CLIP tokenizer. A common issue with adding new tokens is random weight initialization, which delays convergence and impairs image generation. To address this, we initialize the embedding weights of `<sks>` with the weights of the existing anchor token `"style"`. This places the new token in a semantically relevant location in the embedding space before training begins.

During backpropagation, we enable gradient tracking for the embedding layer, but apply a **gradient mask** in the optimizer step:
$$\nabla_{\theta} E[i] = 0 \quad \forall i \neq \text{ID}(<sks>)$$
This guarantees that only the embedding weights for `<sks>` are updated, preventing vocabulary drift and preserving the base model's comprehension of standard tokens.

### 2.2 Task 2: Dual-Adapter LoRA Architecture
Instead of finetuning the full models (which is computationally expensive and causes catastrophic forgetting), we apply Low-Rank Adaptation (LoRA) to both the **UNet** and **Text Encoder**:
* **UNet LoRA**: Targets the cross-attention and self-attention projection weights (`to_q`, `to_k`, `to_v`, `to_out.0`) in the spatial transformer blocks.
* **Text Encoder LoRA**: Targets all attention projection layers (`q_proj`, `k_proj`, `v_proj`, `out_proj`) in the CLIP text transformer encoder to maximize text-to-style alignment.

We configure the PEFT adapter layers with a rank $r=8$ and scaling factor $\alpha=8$, initializing the weights with a Gaussian distribution.

### 2.3 Optimization and Training Pipeline
* **Model Base**: `runwayml/stable-diffusion-v1-5`
* **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $0.01$ for LoRA parameters / $0.0$ for the custom token embedding, learning rate $1\times 10^{-4}$ for UNet LoRA / $1\times 10^{-5}$ for Text Encoder LoRA to preserve prompt alignment and controllability)
* **Batch Size**: 1, with **Gradient Accumulation Steps = 4** (effective batch size of 4 to stabilize gradients)
* **Training Steps**: 800 steps
* **Mixed Precision**: Automatic Mixed Precision (AMP fp16) on GPU to reduce VRAM requirements to under 8 GB.
* **Dataset Preprocessing**: Input images are dynamically resized to 512 pixels along the shorter edge, center-cropped to $512\times 512$, normalized to $[-1, 1]$, and augmented with random horizontal flips and mild color jitter.

---

## 3. Results and Evaluation
The evaluation script `eval_lora.py` loads the base SD 1.5 pipeline, adds `<sks>` to the tokenizer, loads the custom token embedding weight from the trained safetensors, and loads the UNet/Text Encoder LoRA weights.

Using the DPM-Solver Multistep Scheduler for 50 inference steps and a guidance scale of 7.5, we generate samples for the prompt:
`"a busy market, in <sks> style"`

### Key Observations:
1. **Style Fidelity**: The generated images exhibit classic Ghibli features: soft painterly textures, hand-drawn contours, vibrant organic color palettes, and detailed, whimsical background architecture resembling the market scenes from *Spirited Away*.
2. **Prompt Controllability**: The style is tightly coupled to the `<sks>` token. When `<sks>` is omitted, the model generates realistic or default SD 1.5 styled images, proving that the style representation has been successfully isolated to the custom token.
3. **Dual-Adapter Advantage**: Adapting the Text Encoder in tandem with the UNet prevents prompt alignment issues, allowing complex prompts to align correctly with the Ghibli aesthetic instead of degrading visual details.

---

## 4. Limitations and Future Work
While the dual-adapter LoRA setup yields highly appealing painterly results, we identify the following limitations:
1. **Resolution Entanglement**: Because the model is trained at 512x512 resolution, generating at higher aspect ratios (e.g. 1024x768) can result in repeating textures or duplicated focal subjects. This could be resolved with a multi-scale training regime or high-resolution pipeline techniques (like upscale-and-refine).
2. **Vocabulary Over-association**: Despite gradient masking, very high step counts (e.g. >1200) can cause the custom token style to slightly leak into non-stylized prompts. Future work could introduce prior preservation loss with generic captions.

---

## 5. How to Convert this Report to PDF
This report is formatted in Markdown. To compile it to `report.pdf` (required for submission):
1. **VS Code**: Open this file (`report.md`), install the extension **Markdown PDF** by yzane, right-click inside the editor, and select **Markdown PDF: Export (pdf)**.
2. **Pandoc**: Run the command:
   ```bash
   pandoc report.md -o report.pdf --pdf-engine=xelatex
   ```
3. **Google Docs/Word**: Copy-paste the content, format it, and print/save as a PDF file.
