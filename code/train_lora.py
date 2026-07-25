import os
import sys
import builtins

# Override built-in print to force flushing and gracefully handle Windows console unicode encoding errors
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    encoding = getattr(sys.stdout, 'encoding', 'utf-8') or 'utf-8'
    clean_args = []
    for arg in args:
        if isinstance(arg, str):
            clean_args.append(arg.encode(encoding, errors='replace').decode(encoding))
        else:
            clean_args.append(arg)
    builtins.print(*clean_args, **kwargs)

# Suppress TensorFlow logging to clean up execution output and speed up imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Disable oneDNN custom operations warnings from TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("🎨 Ghibli Market LoRA: Starting initialization...")

import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from safetensors.torch import save_file

print("Importing Stable Diffusion & PEFT libraries (this can take up to a minute on CPU)...", flush=True)
# Import Stable Diffusion and PEFT components
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict

# Custom Dataset
from dataset import GhibliDataset


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Dual-Adapter LoRA Style-Tuning on Stable Diffusion 1.5")
    
    # Required arguments
    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True, 
        help="Path to directory containing Ghibli style images"
    )
    parser.add_argument(
        "--instance_token", 
        type=str, 
        required=True, 
        help="Unique identifier token for the style (e.g. <sks>)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory to save the trained LoRA weights"
    )
    
    # Hyper-parameters
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="runwayml/stable-diffusion-v1-5", 
        help="Pretrained Stable Diffusion model name or path"
    )
    parser.add_argument(
        "--resolution", 
        type=int, 
        default=512, 
        help="Input image resolution for training"
    )
    parser.add_argument(
        "--rank", 
        type=int, 
        default=8, 
        help="LoRA rank dimension"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-4, 
        help="Learning rate for LoRA weights and token embedding"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1, 
        help="Batch size for training"
    )
    parser.add_argument(
        "--max_steps", 
        type=int, 
        default=800, 
        help="Maximum training steps"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=4, 
        help="Number of updates steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--validation_steps", 
        type=int, 
        default=None, 
        help="Generate validation images every X steps (defaults to 0 on CPU, 200 on GPU)"
    )
    
    # Toggles
    parser.add_argument(
        "--overwrite", 
        action="store_true", 
        help="Overwrite existing weights file in output directory"
    )
    parser.add_argument(
        "--no_augmentation", 
        action="store_true", 
        help="Disable data augmentations during dataset loading"
    )
    parser.add_argument(
        "--cache_latents", 
        action="store_true", 
        help="Pre-encode dataset images to latents and cache them in RAM to speed up training"
    )

    return parser.parse_args()

def clean_peft_state_dict(peft_state_dict, prefix):
    """Strip base_model prefixes and adapter names from PEFT state dict and add target prefix (unet or text_encoder)"""
    clean_dict = {}
    for k, v in peft_state_dict.items():
        key = k
        if key.startswith("base_model.model."):
            key = key[len("base_model.model."):]
        # Strip PEFT adapter name suffix (e.g. '.default') to ensure compatibility with standard Diffusers loaders
        key = key.replace(".default.", ".")
        clean_dict[f"{prefix}.{key}"] = v
    return clean_dict

def main():
    args = parse_args()
    
    # 1. Output directory verification
    os.makedirs(args.output_dir, exist_ok=True)
    weights_path = os.path.join(args.output_dir, "pytorch_lora_weights.safetensors")
    if os.path.exists(weights_path) and not args.overwrite:
        print(f"Error: Weights file '{weights_path}' already exists.")
        print("Use the --overwrite flag to overwrite the existing file.")
        return
        
    # 2. Setup environment and device
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}", flush=True)
    
    # Configure dynamic defaults for validation steps based on CPU/GPU
    if args.validation_steps is None:
        args.validation_steps = 0 if device.type == "cpu" else 200
        
    if args.validation_steps > 0:
        print(f"Validation images will be generated every {args.validation_steps} steps.", flush=True)
    else:
        print("Intermediate validation image generation is disabled.", flush=True)
        
    if args.cache_latents and not args.no_augmentation:
        print("Notice: Caching latents requires disabling data augmentation. Disabling data augmentation...", flush=True)
        args.no_augmentation = True
    
    # 3. Load pretrained models
    print(f"Loading pretrained models from: {args.model_name}")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_name, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_name, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.model_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.model_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.model_name, subfolder="unet")
    
    # 4. Task 1: Add new style token to tokenizer & resize text_encoder embeddings
    num_added_tokens = tokenizer.add_tokens(args.instance_token)
    print(f"Added token: {args.instance_token} to tokenizer. Number of added tokens: {num_added_tokens}")
    text_encoder.resize_token_embeddings(len(tokenizer))
    
    # Initialize the new token embedding with the 'style' token representation
    token_id = tokenizer.convert_tokens_to_ids(args.instance_token)
    style_token_ids = tokenizer.encode("style", add_special_tokens=False)
    if len(style_token_ids) > 0:
        style_token_id = style_token_ids[0]
        print(f"Initializing embedding of '{args.instance_token}' (ID {token_id}) with weights of 'style' (ID {style_token_id})")
        with torch.no_grad():
            token_embeds = text_encoder.get_input_embeddings().weight.data
            token_embeds[token_id] = token_embeds[style_token_id].clone()
    else:
        print("Warning: Could not find 'style' token in vocabulary for initialization.")
        
    # 5. Freeze base weights
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # 6. Task 2: Configure PEFT LoRA for both UNet and Text Encoder
    print("Wrapping models with PEFT LoRA adapters...")
    
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        init_lora_weights="gaussian"
    )
    unet = get_peft_model(unet, unet_lora_config)
    unet.print_trainable_parameters()
    
    text_encoder_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        init_lora_weights="gaussian"
    )
    text_encoder = get_peft_model(text_encoder, text_encoder_lora_config)
    text_encoder.print_trainable_parameters()
    
    # Enable gradient tracking on the text encoder embeddings matrix so the new token is trained
    text_encoder.get_input_embeddings().weight.requires_grad_(True)
    
    # 7. Device positioning and dtype
    vae.to(device)
    unet.to(device)
    text_encoder.to(device)
    
    # Fallback to fp32 on CPU, AMP fp16 on GPU
    weight_dtype = torch.float32
    if device.type == "cuda":
        weight_dtype = torch.float16
        vae.to(dtype=weight_dtype)
        unet.to(dtype=weight_dtype)
        text_encoder.to(dtype=weight_dtype)
        
    # 8. Setup optimizer and dataset
    # We optimize the LoRA weights of UNet and Text Encoder, AND the custom embedding token
    # We optimize the LoRA weights of UNet and Text Encoder, AND the custom embedding token.
    # We must isolate the embedding parameter and set its weight_decay to 0.0 to prevent
    # decaying the base vocabulary of the text encoder.
    embedding_param = text_encoder.get_input_embeddings().weight
    text_encoder_params = [
        p for p in filter(lambda p: p.requires_grad, text_encoder.parameters()) 
        if p is not embedding_param
    ]
    unet_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    
    params_to_optimize = [
        {"params": unet_params, "lr": args.learning_rate},
        {"params": text_encoder_params, "lr": args.learning_rate},
        {"params": [embedding_param], "lr": args.learning_rate, "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(params_to_optimize, weight_decay=1e-2)
    
    dataset = GhibliDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        instance_token=args.instance_token,
        resolution=args.resolution,
        use_augmentation=not args.no_augmentation
    )
    
    if args.cache_latents:
        print("Caching VAE latents in memory to optimize training performance...", flush=True)
        dataset.cache_latents_with_vae(vae, device, weight_dtype)
        
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # 9. Training Loop Setup
    global_step = 0
    epochs = (args.max_steps * args.gradient_accumulation_steps // len(train_dataloader)) + 1
    
    print(f"Beginning training loop. Total epochs: {epochs}, Max steps: {args.max_steps}")
    progress_bar = tqdm(total=args.max_steps, desc="Training")
    
    # Setup mixed-precision Scaler if on GPU
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    
    unet.train()
    text_encoder.train()
    
    for epoch in range(epochs):
        if global_step >= args.max_steps:
            break
            
        for step, batch in enumerate(train_dataloader):
            if global_step >= args.max_steps:
                break
                
            # Process batch
            input_ids = batch["input_ids"].to(device)
            
            # Forward pass inside context manager for mixed precision
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                if "latents" in batch:
                    # Use pre-cached VAE latents (CPU/GPU-agnostic)
                    latents = batch["latents"].to(device, dtype=weight_dtype)
                else:
                    # Run VAE encoder on-the-fly (e.g. if data augmentation is on)
                    pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
                    scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
                    latents = vae.encode(pixel_values).latent_dist.sample() * scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, 
                    noise_scheduler.config.num_train_timesteps, 
                    (latents.shape[0],), 
                    device=device
                ).long()
                
                # Add noise (forward diffusion)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get encoder hidden states for prompt
                encoder_hidden_states = text_encoder(input_ids)[0]
                
                # Predict noise residuals
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Calculate MSE Loss
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                loss = loss / args.gradient_accumulation_steps
                
            # Backward pass
            scaler.scale(loss).backward()
            
            # Zero out gradients for all token embeddings except the custom style token (<sks>)
            # This isolates updates to the style token representation, protecting vocabulary base weights
            if text_encoder.get_input_embeddings().weight.grad is not None:
                grad = text_encoder.get_input_embeddings().weight.grad
                mask = torch.zeros(grad.shape[0], 1, device=grad.device)
                mask[token_id] = 1.0
                grad.data.mul_(mask)
                
            # Step optimizer after gradient accumulation steps
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params_to_optimize[0]["params"] + params_to_optimize[1]["params"], 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item() * args.gradient_accumulation_steps})
                
                # 10. Intermediate Validation Generation
                if args.validation_steps > 0 and global_step % args.validation_steps == 0:
                    unet.eval()
                    text_encoder.eval()
                    validation_prompt = f"a busy market, in {args.instance_token} style"
                    print(f"\n[Step {global_step}] Generating validation image...", flush=True)
                    
                    try:
                        # Construct a temporary pipeline reusing active components to save system RAM and VRAM
                        validation_pipe = StableDiffusionPipeline(
                            vae=vae,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            unet=unet,
                            scheduler=noise_scheduler,
                            safety_checker=None,
                            feature_extractor=None,
                            requires_safety_checker=False,
                            torch_dtype=weight_dtype
                        )
                        
                        # Generate image
                        with torch.no_grad():
                            image = validation_pipe(
                                validation_prompt,
                                num_inference_steps=30,
                                guidance_scale=7.5
                            ).images[0]
                            
                        # Save validation image
                        val_dir = os.path.join(args.output_dir, "validation")
                        os.makedirs(val_dir, exist_ok=True)
                        image.save(os.path.join(val_dir, f"step_{global_step}.png"))
                        print(f"Validation image saved to {val_dir}/step_{global_step}.png")
                    except Exception as e:
                        print(f"Validation generation failed: {e}")
                        
                    # Revert to training state
                    unet.train()
                    text_encoder.train()
                    
    progress_bar.close()
    
    # 11. Extract and Clean PEFT LoRA state dicts
    print("Preparing adapters for saving...")
    unet_state_dict = get_peft_model_state_dict(unet, adapter_name="default")
    text_encoder_state_dict = get_peft_model_state_dict(text_encoder, adapter_name="default")
    
    clean_unet_dict = clean_peft_state_dict(unet_state_dict, "unet")
    clean_text_encoder_dict = clean_peft_state_dict(text_encoder_state_dict, "text_encoder")
    
    # Compile everything into a single weights file
    save_dict = {**clean_unet_dict, **clean_text_encoder_dict}
    
    # Save the trained embedding of the style token representation
    # We store only the specific row matching the added token to keep weights small
    trained_embedding = text_encoder.get_input_embeddings().weight.data[token_id].cpu()
    save_dict["text_encoder_embeddings"] = trained_embedding
    
    # 12. Save File
    print(f"Saving final adapter weights to {weights_path}...")
    save_file(save_dict, weights_path)
    print("Training finished successfully! All deliverables produced.")

if __name__ == "__main__":
    main()
