import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class GhibliDataset(Dataset):
    """
    A custom Dataset for loading Ghibli style reference images and pairing them 
    with subject-agnostic style token prompts for Stable Diffusion LoRA training.
    """
    def __init__(self, data_dir, tokenizer, instance_token, resolution=512, use_augmentation=True):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.instance_token = instance_token
        self.resolution = resolution
        
        # Supported image formats
        self.image_extensions = (".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG", ".JPEG", ".WEBP")
        self.image_paths = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.endswith(self.image_extensions)
        ])
        self.image_paths = sorted(set(self.image_paths))  # Deduplicate and sort for reproducibility
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in data directory: {data_dir}")
            
        # Define image transforms
        transform_list = []
        
        if use_augmentation:
            # Random resized crop forces the model to learn fine textures and outlines 
            # at multiple scales/crops, preventing small background faces from blurring.
            transform_list.extend([
                transforms.RandomResizedCrop(resolution, scale=(0.85, 1.0), ratio=(1.0, 1.0), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(p=0.5),
                # Emphasizes outlines and drawings so the model learns cleaner line-art borders
                transforms.RandomAdjustSharpness(sharpness_factor=1.8, p=0.5),
                transforms.ColorJitter(brightness=0.04, contrast=0.06, saturation=0.04, hue=0.005),
            ])
        else:
            transform_list.extend([
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
            ])
            
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Scale pixel values to [-1, 1]
        ])
        
        self.transform = transforms.Compose(transform_list)
        
        # Define subject-agnostic Studio Ghibli style prompt templates
        self.prompt_templates = [
            f"in {self.instance_token} style",
            f"a painting, in {self.instance_token} style",
            f"a 2D anime illustration, in {self.instance_token} style",
            f"a hand-drawn animation scene, in {self.instance_token} style",
            f"a beautiful artwork, in {self.instance_token} style",
            f"a classic 2D animation frame, in {self.instance_token} style"
        ]
        
        # Pre-tokenize all prompt templates to save computational time
        self.tokenized_prompts = [
            self.tokenizer(
                p,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt"
            ).input_ids[0]
            for p in self.prompt_templates
        ]
        
        self.cached_latents = None

    def __len__(self):
        return len(self.image_paths)

    def __repr__(self):
        return (f"GhibliDataset(data_dir='{self.data_dir}', "
                f"num_images={len(self.image_paths)}, "
                f"resolution={self.resolution}, "
                f"token='{self.instance_token}', "
                f"num_prompts={len(self.prompt_templates)})")

    def cache_latents_with_vae(self, vae, device, weight_dtype):
        """Pre-encode all images in the dataset to latents using VAE and cache them in CPU RAM."""
        from tqdm import tqdm
        self.cached_latents = []
        vae.eval()
        with torch.no_grad():
            for idx in tqdm(range(len(self.image_paths)), desc="Caching VAE latents"):
                img_path = self.image_paths[idx]
                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(f"Error reading image {img_path}: {e}. Retrying first image.", flush=True)
                    image = Image.open(self.image_paths[0]).convert("RGB")
                
                # Keep VAE encoding in float32 to prevent rounding errors/clipping in details
                pixel_values = self.transform(image).unsqueeze(0).to(device, dtype=torch.float32)
                scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
                # Encode to latent space and sample representation
                latents = vae.encode(pixel_values).latent_dist.sample() * scaling_factor
                # Store on CPU to conserve RAM/VRAM
                self.cached_latents.append(latents.squeeze(0).cpu().to(dtype=weight_dtype))

    def __getitem__(self, idx):
        # Randomly select a generic template for this sample on each call to prevent overfitting to a fixed mapping
        prompt_idx = random.randint(0, len(self.tokenized_prompts) - 1)
        input_ids = self.tokenized_prompts[prompt_idx]
        
        if self.cached_latents is not None:
            return {
                "latents": self.cached_latents[idx],
                "input_ids": input_ids
            }
            
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error reading image {img_path}: {e}. Retrying first image.", flush=True)
            image = Image.open(self.image_paths[0]).convert("RGB")
            
        pixel_values = self.transform(image)
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids
        }
