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

print("🎨 Ghibli Market LoRA: Starting evaluation script...", flush=True)

import argparse
import torch

print("Importing Stable Diffusion evaluation libraries (this can take up to a minute on CPU)...", flush=True)
from safetensors.torch import load_file
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Dual-Adapter LoRA for Stable Diffusion 1.5")
    
    # Required arguments
    parser.add_argument(
        "--weights", 
        type=str, 
        required=True, 
        help="Path to the trained safetensors LoRA weights file"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        required=True, 
        help="The style activation prompt (e.g. 'a busy market, in <sks> style')"
    )
    parser.add_argument(
        "--outdir", 
        type=str, 
        required=True, 
        help="Directory to save the generated images"
    )
    
    # Optional evaluation configuration
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="runwayml/stable-diffusion-v1-5", 
        help="Base Stable Diffusion model name or path"
    )
    parser.add_argument(
        "--num_images", 
        type=int, 
        default=3, 
        help="Number of images to render (default: 3)"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=50, 
        help="Number of inference steps (default: 50)"
    )
    parser.add_argument(
        "--guidance_scale", 
        type=float, 
        default=7.5, 
        help="Classifier-Free Guidance (CFG) scale (default: 7.5)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=123, 
        help="Seed for generation reproducibility"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.outdir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")
    
    # Precision setting
    weight_dtype = torch.float32
    if device.type == "cuda":
        weight_dtype = torch.float16
        
    # 1. Load the base Stable Diffusion pipeline
    print(f"Loading base Stable Diffusion pipeline: {args.model_name}")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        safety_checker=None,
        torch_dtype=weight_dtype
    )
    
    # Use DPM-Solver Multistep Scheduler for high-quality, fast generation
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # 2. Identify the instance token from the prompt (e.g. "<sks>")
    # We search the prompt for common bracket-style custom tokens
    instance_token = None
    if "<sks>" in args.prompt:
        instance_token = "<sks>"
    else:
        # Fallback to search for any bracket token in prompt
        import re
        tokens = re.findall(r"<[^>]+>", args.prompt)
        if tokens:
            instance_token = tokens[0]
            print(f"Detected custom token in prompt: {instance_token}")
            
    if instance_token is None:
        print("Warning: No custom bracket token (like <sks>) detected in prompt.")
        print("The LoRA model expects a custom token. Proceeding without custom token resizing.")
        
    # 3. Load LoRA weights and token embedding from safetensors
    print(f"Loading custom weights from: {args.weights}")
    state_dict = load_file(args.weights)
    
    # Process custom token embedding if present
    if instance_token is not None:
        # Add token to tokenizer
        num_added = pipe.tokenizer.add_tokens(instance_token)
        pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
        token_id = pipe.tokenizer.convert_tokens_to_ids(instance_token)
        
        if "text_encoder_embeddings" in state_dict:
            trained_embedding = state_dict.pop("text_encoder_embeddings")
            print(f"Loading trained embedding of shape {trained_embedding.shape} for token {instance_token} (ID {token_id})")
            with torch.no_grad():
                pipe.text_encoder.get_input_embeddings().weight.data[token_id] = trained_embedding.to(
                    device=pipe.text_encoder.get_input_embeddings().weight.device, dtype=weight_dtype
                )
        else:
            print("Warning: 'text_encoder_embeddings' key not found in weights file.")
            print("The custom token will use default resized embedding weights.")
            
    # Load remaining LoRA weights (UNet and text encoder)
    pipe.load_lora_weights(state_dict)
    
    # Setup execution accelerators
    pipe.to(device)
    if device.type == "cuda":
        # Enable memory-efficient operations on GPU
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        
    # 4. Generate Images
    print(f"Generating {args.num_images} images for prompt: '{args.prompt}'")
    
    # Create generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    for idx in range(args.num_images):
        print(f"Rendering image {idx + 1} of {args.num_images}...")
        
        with torch.no_grad():
            image = pipe(
                args.prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator
            ).images[0]
            
        save_path = os.path.join(args.outdir, f"sample_{idx + 1}.png")
        image.save(save_path)
        print(f"Image saved to: {save_path}")
        
    print(f"All images successfully saved to: {args.outdir}")

if __name__ == "__main__":
    main()
