import os
import sys
import builtins

# Override built-in print to force flushing and gracefully handle console encoding issues
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

# Suppress warnings and logs to clean up output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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

# Using Cosine schedule with restarts to escape local plateaus
from transformers import CLIPTokenizer, CLIPTextModel, get_cosine_with_hard_restarts_schedule_with_warmup
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
    
    # Optional arguments with project defaults
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="style_imgs/512", 
        help="Path to directory containing Ghibli style images (default: style_imgs/512)"
    )
    parser.add_argument(
        "--instance_token", 
        type=str, 
        default="<sks>", 
        help="Unique identifier token for the style (default: <sks>)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="lora_out", 
        help="Directory to save the trained LoRA weights (default: lora_out)"
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
        default=16, 
        help="LoRA rank dimension (default: 16)"
    )
    parser.add_argument(
        "--noise_offset", 
        type=float, 
        default=0.08, 
        help="Noise offset to apply to initial noise for richer color contrast (default: 0.08)"
    )
    parser.add_argument(
        "--initializer_token", 
        type=str, 
        default="style", 
        help="Vocabulary token word used to initialize the style token embedding (default: 'style')"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-4, 
        help="Learning rate for LoRA weights"
    )
    parser.add_argument(
        "--text_encoder_lr", 
        type=float, 
        default=8e-6, 
        help="Learning rate for the text encoder LoRA adapter weights (default: 8e-6)"
    )
    parser.add_argument(
        "--embedding_lr", 
        type=float, 
        default=1e-4, 
        help="Learning rate for the custom style token embedding (default: 1e-4)"
    )
    parser.add_argument(
        "--snr_gamma", 
        type=float, 
        default=5.0, 
        help="Min-SNR loss weighting gamma value (default: 5.0)"
    )
    parser.add_argument(
        "--warmup_steps", 
        type=int, 
        default=200, 
        help="Warmup steps for Cosine LR scheduler (default: 200)"
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
        default=1000, 
        help="Maximum training steps (default: 1000)"
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
        help="Generate validation images every X steps (defaults to 200 on GPU)"
    )
    
    # Toggles
    parser.add_argument(
        "--overwrite", 
        action="store_true", 
        default=True,
        help="Overwrite existing weights file in output directory (default: True)"
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
    
    # Initialize the new token embedding with the initializer token representation (default: 'style')
    token_id = tokenizer.convert_tokens_to_ids(args.instance_token)
    anchor_token = args.initializer_token
    style_token_ids = tokenizer.encode(anchor_token, add_special_tokens=False)
    if len(style_token_ids) == 0 and anchor_token != "style":
        anchor_token = "style"
        style_token_ids = tokenizer.encode(anchor_token, add_special_tokens=False)
        
    if len(style_token_ids) > 0:
        style_token_id = style_token_ids[0]
        print(f"Initializing embedding of '{args.instance_token}' (ID {token_id}) with weights of '{anchor_token}' (ID {style_token_id})")
        with torch.no_grad():
            token_embeds = text_encoder.get_input_embeddings().weight.data
            token_embeds[token_id] = token_embeds[style_token_id].clone()
    else:
        print(f"Warning: Could not find '{anchor_token}' token in vocabulary for initialization.")
        
    # 5. Freeze base weights
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # 6. Task 2: Configure PEFT LoRA for both UNet and Text Encoder
    print("Wrapping models with PEFT LoRA adapters...")
    
    # Targeting both cross-attention and linear convolutional projections to capture style texture
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"],
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
    
    # 7. Register PyTorch gradient hook to isolate updates to ONLY the custom trigger token.
    # This prevents vocabulary drift automatically during backpropagation.
    embedding_param = text_encoder.get_input_embeddings().weight
    def register_embedding_grad_hook(target_token_id):
        def hook(grad):
            mask = torch.zeros(grad.shape[0], 1, device=grad.device)
            mask[target_token_id] = 1.0
            return grad * mask
        return embedding_param.register_hook(hook)
    
    grad_hook_handle = register_embedding_grad_hook(token_id)
    
    # 8. Device positioning and dtype
    vae.to(device)  # Keep VAE in float32 for high precision reconstructions
    unet.to(device)
    text_encoder.to(device)
    
    weight_dtype = torch.float32
    if device.type == "cuda":
        weight_dtype = torch.float16
        
    # 9. Setup optimizer and dataset
    # We isolate the embedding parameter and set its weight_decay to 0.0 to prevent vocabulary decay.
    text_encoder_params = [
        p for p in filter(lambda p: p.requires_grad, text_encoder.parameters()) 
        if p is not embedding_param
    ]
    unet_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    
    params_to_optimize = [
        {"params": unet_params, "lr": args.learning_rate},
        {"params": text_encoder_params, "lr": args.text_encoder_lr},
        {"params": [embedding_param], "lr": args.embedding_lr, "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(params_to_optimize, weight_decay=1e-2)
    lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
        num_cycles=3
    )
    
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
    
    # 10. Training Loop Setup
    global_step = 0
    accumulation_step = 0
    epochs = (args.max_steps * args.gradient_accumulation_steps // len(train_dataloader)) + 1
    
    print(f"Beginning training loop. Total epochs: {epochs}, Max steps: {args.max_steps}")
    progress_bar = tqdm(total=args.max_steps, desc="Training")
    
    # Setup mixed-precision Scaler if on GPU
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    
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
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                if "latents" in batch:
                    latents = batch["latents"].to(device, dtype=weight_dtype)
                else:
                    # Run VAE encoder at full float32 precision for structural accuracy
                    pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
                    scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
                    latents = vae.encode(pixel_values).latent_dist.sample() * scaling_factor
                    latents = latents.to(dtype=weight_dtype)
                
                # Sample noise with optional offset noise to improve color contrast
                noise = torch.randn_like(latents)
                if args.noise_offset > 0:
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device, dtype=latents.dtype
                    )
                
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
                
                # Calculate Loss with optional Min-SNR Gamma weighting
                if args.snr_gamma > 0:
                    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
                    alpha_prod_t = alphas_cumprod[timesteps]
                    snr = alpha_prod_t / (1 - alpha_prod_t)
                    gamma = torch.full_like(snr, args.snr_gamma)
                    snr_weight = torch.stack([snr, gamma], dim=-1).min(dim=-1)[0] / snr
                    loss_unweighted = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
                    loss = (loss_unweighted.mean(dim=list(range(1, len(loss_unweighted.shape)))) * snr_weight).mean()
                else:
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                    
                loss = loss / args.gradient_accumulation_steps
                
            # Backward pass (hook handles masking automatically)
            scaler.scale(loss).backward()
            
            accumulation_step += 1
                
            # Step optimizer after gradient accumulation steps
            if accumulation_step % args.gradient_accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                all_trainable_params = [p for group in params_to_optimize for p in group["params"]]
                torch.nn.utils.clip_grad_norm_(all_trainable_params, 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss.item() * args.gradient_accumulation_steps})
                
                # 11. Intermediate Validation Generation
                if args.validation_steps > 0 and global_step % args.validation_steps == 0:
                    unet.eval()
                    text_encoder.eval()
                    validation_prompt = f"a busy market, in {args.instance_token} style"
                    print(f"\n[Step {global_step}] Generating validation image...", flush=True)
                    
                    try:
                        # Construct a temporary pipeline reusing active components
                        validation_pipe = StableDiffusionPipeline(
                            vae=vae,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            unet=unet,
                            scheduler=noise_scheduler,
                            safety_checker=None,
                            feature_extractor=None,
                            requires_safety_checker=False
                        )
                        validation_pipe.vae.to(dtype=torch.float32)  # High-precision validation decode
                        
                        # Wrap VAE decode to automatically handle float16 latents coming from the UNet
                        original_decode = validation_pipe.vae.decode
                        def float32_decode(latents, *args, **kwargs):
                            return original_decode(latents.to(dtype=torch.float32), *args, **kwargs)
                        validation_pipe.vae.decode = float32_decode
                        
                        with torch.inference_mode():
                            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
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
    
    # Remove gradient hook handle before saving to avoid cleanup issues
    grad_hook_handle.remove()
    
    # 12. Extract and Clean PEFT LoRA state dicts
    print("Preparing adapters for saving...")
    unet_state_dict = get_peft_model_state_dict(unet, adapter_name="default")
    text_encoder_state_dict = get_peft_model_state_dict(text_encoder, adapter_name="default")
    
    clean_unet_dict = clean_peft_state_dict(unet_state_dict, "unet")
    clean_text_encoder_dict = clean_peft_state_dict(text_encoder_state_dict, "text_encoder")
    
    # Compile everything into a single weights file
    save_dict = {**clean_unet_dict, **clean_text_encoder_dict}
    
    # Save the trained embedding of the style token representation
    trained_embedding = text_encoder.get_input_embeddings().weight.data[token_id].cpu()
    save_dict["text_encoder_embeddings"] = trained_embedding
    
    # Save file
    print(f"Saving final adapter weights to {weights_path}...")
    save_file(save_dict, weights_path)
    print("Training finished successfully! All deliverables produced.")

if __name__ == "__main__":
    main()
