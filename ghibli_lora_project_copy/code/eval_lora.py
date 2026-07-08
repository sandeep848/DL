import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


MODEL_NAME = "runwayml/stable-diffusion-v1-5"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="a busy market, in <sks> style")
    parser.add_argument("--outdir", type=str, default="samples")
    parser.add_argument("--num_images", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # GTX 1650 can sometimes generate blank/black images with float16.
    # Use float32 for stable image generation.
    dtype = torch.float32

    print(f"Using device: {device}")
    print(f"Loading base model: {MODEL_NAME}")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()

    print(f"Loading LoRA weights from: {args.weights}")
    # pipe.load_lora_weights(args.weights)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    print(f"Generating {args.num_images} images...")
    print(f"Prompt: {args.prompt}")

    for i in range(args.num_images):
        image = pipe(
            prompt=args.prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
            height=512,
            width=512,
        ).images[0]

        save_path = outdir / f"sample_{i+1}.png"
        image.save(save_path)
        print(f"Saved: {save_path}")

    print("Evaluation complete.")


if __name__ == "__main__":
    main()