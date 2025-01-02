import os
import requests
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def create_dataset(output_dir, num_images=100):
    """
    Create a dataset of high-res and corresponding low-res images
    """
    # Create directories if they don't exist
    high_res_dir = os.path.join(output_dir, "high_res")
    low_res_dir = os.path.join(output_dir, "low_res")
    os.makedirs(high_res_dir, exist_ok=True)
    os.makedirs(low_res_dir, exist_ok=True)

    def process_image(img_path, index):
        # Read original image
        img = Image.open(img_path).convert('RGB')
        
        # Create high-res version (256x256)
        high_res = img.resize((256, 256), Image.LANCZOS)
        
        # Create low-res version (64x64)
        low_res = img.resize((64, 64), Image.BICUBIC)
        
        # Save images
        high_res.save(os.path.join(high_res_dir, f"abacus_{index:04d}_high.png"))
        low_res.save(os.path.join(low_res_dir, f"abacus_{index:04d}_low.png"))

    # Process existing images in a directory
    def process_directory(input_dir):
        images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for idx, img_name in enumerate(tqdm(images)):
            img_path = os.path.join(input_dir, img_name)
            process_image(img_path, idx)

    # Apply data augmentation to expand dataset
    def augment_image(img):
        augmentations = [
            lambda x: x,  # Original
            lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),  # Horizontal flip
            lambda x: x.rotate(90),  # 90 degree rotation
            lambda x: x.rotate(180),  # 180 degree rotation
            lambda x: x.rotate(270),  # 270 degree rotation
        ]
        return [aug(img) for aug in augmentations]

if __name__ == "__main__":
    # Example usage
    input_dir = "raw_abacus_images"  # Directory containing your original abacus images
    output_dir = "dataset"
    create_dataset(output_dir)