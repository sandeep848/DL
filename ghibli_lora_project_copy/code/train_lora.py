import os
import math
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from torchvision import transforms

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict


MODEL_NAME = "runwayml/stable-diffusion-v1-5"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--instance_token", type=str, default="<sks>")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class StyleImageDataset(Dataset):
    def __init__(self, data_dir, instance_token, resolution=512):
        self.data_dir = Path(data_dir)
        self.instance_token = instance_token
        self.prompt = f"a busy market, in {instance_token} style"

        self.image_paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"]:
            self.image_paths.extend(self.data_dir.glob(ext))
            self.image_paths.extend(self.data_dir.glob(ext.upper()))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)
        return {
            "pixel_values": pixel_values,
            "prompt": self.prompt,
        }


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise ValueError(
            f"Output dir {output_dir} already exists and is not empty. "
            f"Use --overwrite if you want to overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Using device: {device}")

    # ----------------------------
    # Load tokenizer and models
    # ----------------------------
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

    # ----------------------------
    # Add new style token
    # ----------------------------
    num_added_tokens = tokenizer.add_tokens(args.instance_token)
    if num_added_tokens == 0:
        print(f"Token {args.instance_token} already exists in tokenizer.")
    text_encoder.resize_token_embeddings(len(tokenizer))

    token_id = tokenizer.convert_tokens_to_ids(args.instance_token)

    # Initialize new token embedding from the word "style"
    initializer_token = "style"
    init_token_id = tokenizer.encode(initializer_token, add_special_tokens=False)[0]
    with torch.no_grad():
        text_encoder.get_input_embeddings().weight[token_id] = \
            text_encoder.get_input_embeddings().weight[init_token_id]

    # ----------------------------
    # Freeze base models
    # ----------------------------
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)


    # ----------------------------
    # Add LoRA adapters
    # ----------------------------
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )

    text_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.0,
        bias="none",
    )

    unet = get_peft_model(unet, unet_lora_config)
    text_encoder = get_peft_model(text_encoder, text_lora_config)

    # Also train ONLY the new token embedding row
    text_encoder.get_input_embeddings().weight.requires_grad = True

    # Gradient checkpointing to reduce memory
    unet.enable_gradient_checkpointing()
    if hasattr(text_encoder, "gradient_checkpointing_enable"):
        text_encoder.gradient_checkpointing_enable()

    vae.to(device, dtype=dtype)
    unet.to(device, dtype=dtype)
    text_encoder.to(device, dtype=dtype)

    # Important fix:
    # Keep trainable LoRA parameters in float32 to avoid
    # "Attempting to unscale FP16 gradients" error.
    for model in [unet, text_encoder]:
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.float()

    # ----------------------------
    # Dataset
    # ----------------------------
    dataset = StyleImageDataset(
        data_dir=args.data_dir,
        instance_token=args.instance_token,
        resolution=args.resolution,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # ----------------------------
    # Optimizer
    # ----------------------------
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    trainable_params += [p for p in text_encoder.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    # ----------------------------
    # Training loop
    # ----------------------------
    unet.train()
    text_encoder.train()

    global_step = 0
    progress_bar = tqdm(total=args.max_steps, desc="Training")

    while global_step < args.max_steps:
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
            prompts = batch["prompt"]

            text_inputs = tokenizer(
                list(prompts),
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
                dtype=torch.long,
            )

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                encoder_hidden_states = text_encoder(input_ids)[0]
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                ).sample

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            # Keep gradients only for the new token embedding row
            emb = text_encoder.get_input_embeddings().weight
            if emb.grad is not None:
                mask = torch.ones(emb.grad.shape[0], dtype=torch.bool, device=emb.grad.device)
                mask[token_id] = False
                emb.grad[mask] = 0

            if (global_step + 1) % args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item() * args.grad_accum_steps:.4f}"})

            if global_step >= args.max_steps:
                break

    progress_bar.close()

    # ----------------------------
    # Save ONE LoRA weights file
    # ----------------------------

    # Remove PEFT wrapper prefixes like "base_model.model."
    # so Diffusers can correctly load the LoRA adapter.
    unet_lora_state_dict = get_peft_model_state_dict(unet)
    clean_unet_lora_state_dict = {}

    for key, value in unet_lora_state_dict.items():
        new_key = key.replace("base_model.model.", "")
        clean_unet_lora_state_dict[new_key] = value

    text_lora_state_dict = get_peft_model_state_dict(text_encoder)
    clean_text_lora_state_dict = {}

    for key, value in text_lora_state_dict.items():
        new_key = key.replace("base_model.model.", "")
        clean_text_lora_state_dict[new_key] = value

    unet_lora_state_dict = convert_state_dict_to_diffusers(clean_unet_lora_state_dict)
    text_lora_state_dict = convert_state_dict_to_diffusers(clean_text_lora_state_dict)

    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(output_dir),
        unet_lora_layers=unet_lora_state_dict,
        text_encoder_lora_layers=text_lora_state_dict,
        weight_name="pytorch_lora_weights.safetensors",
        safe_serialization=True,
    )

    print(f"\nSaved LoRA weights to: {output_dir / 'pytorch_lora_weights.safetensors'}")
    

if __name__ == "__main__":
    main()