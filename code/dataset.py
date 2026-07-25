import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class GhibliDataset(Dataset):
    """
    A custom Dataset for loading Ghibli style reference images and pairing them 
    with the style token prompt for Stable Diffusion LoRA training.
    """
    def __init__(self, data_dir, tokenizer, instance_token, resolution=512, use_augmentation=True):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.instance_token = instance_token
        self.resolution = resolution
        
        # Supported image formats
        self.image_extensions = (".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG", ".JPEG", ".WEBP")
        self.image_paths = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.endswith(self.image_extensions)
        ]
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in data directory: {data_dir}")
            
        # Define image transforms
        transform_list = [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
        ]
        
        # Apply standard augmentations for style generalization if enabled
        if use_augmentation:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01),
            ])
            
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Scale pixel values to [-1, 1]
        ])
        
        self.transform = transforms.Compose(transform_list)
        
        # Define prompt template for style learning
        self.prompt = f"a painting in {self.instance_token} style"
        
        # Pre-tokenize the caption
        self.input_ids = self.tokenizer(
            self.prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids[0]
        
        self.cached_latents = None

    def __len__(self):
        return len(self.image_paths)

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
                
                pixel_values = self.transform(image).unsqueeze(0).to(device, dtype=weight_dtype)
                scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
                # Encode to latent space and sample representation
                latents = vae.encode(pixel_values).latent_dist.sample() * scaling_factor
                # Store on CPU to conserve RAM/VRAM
                self.cached_latents.append(latents.squeeze(0).cpu())

    def __getitem__(self, idx):
        if self.cached_latents is not None:
            return {
                "latents": self.cached_latents[idx],
                "input_ids": self.input_ids
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
            "input_ids": self.input_ids
        }
