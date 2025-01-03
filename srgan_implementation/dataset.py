import os
from PIL import Image
from pathlib import Path
import shutil
from tqdm import tqdm

class ImageProcessor:
    def __init__(self, input_dir, output_base_dir):
        self.input_dir = Path(input_dir)
        self.output_base_dir = Path(output_base_dir)
        
        # Create output directories
        self.high_res_dir = self.output_base_dir / 'high_res'
        self.low_res_dir = self.output_base_dir / 'low_res'
        
        # Create directories if they don't exist
        self.high_res_dir.mkdir(parents=True, exist_ok=True)
        self.low_res_dir.mkdir(parents=True, exist_ok=True)
        
        # Set sizes
        self.high_res_size = (256, 256)
        self.low_res_size = (64, 64)
        
    def verify_input_directory(self):
        """Verify input directory and count valid images"""
        if not self.input_dir.exists():
            raise Exception(f"Input directory {self.input_dir} does not exist!")
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in self.input_dir.iterdir() 
                      if f.suffix.lower() in valid_extensions]
        
        if not image_files:
            raise Exception(f"No valid images found in {self.input_dir}")
            
        print(f"Found {len(image_files)} valid images in input directory")
        return image_files
    
    def process_single_image(self, image_path, index):
        """Process a single image and save high and low res versions"""
        try:
            # Open and convert to RGB
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create high-res version
                high_res = img.resize(self.high_res_size, Image.LANCZOS)
                high_res_path = self.high_res_dir / f"image_{index:04d}.png"
                high_res.save(high_res_path, "PNG")
                
                # Create low-res version
                low_res = img.resize(self.low_res_size, Image.LANCZOS)
                low_res_path = self.low_res_dir / f"image_{index:04d}.png"
                low_res.save(low_res_path, "PNG")
                
                print(f"Processed {image_path.name} -> {high_res_path.name}")
                return True
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return False
    
    def process_images(self):
        """Process all images and verify results"""
        try:
            # Verify input directory
            image_files = self.verify_input_directory()
            
            # Process each image
            successful = 0
            for idx, image_path in enumerate(tqdm(image_files)):
                if self.process_single_image(image_path, idx):
                    successful += 1
            
            # Verify output
            high_res_count = len(list(self.high_res_dir.glob('*.png')))
            low_res_count = len(list(self.low_res_dir.glob('*.png')))
            
            print("\nProcessing Summary:")
            print(f"Total images processed: {successful}")
            print(f"Images in high_res directory: {high_res_count}")
            print(f"Images in low_res directory: {low_res_count}")
            
            if high_res_count == 0 or low_res_count == 0:
                print("\nWARNING: Output directories are empty!")
                print("Please check the following:")
                print("1. Input directory contains valid images")
                print("2. Script has write permissions to output directories")
                print("3. Sufficient disk space is available")
            
        except Exception as e:
            print(f"Error during processing: {e}")

# Example usage
def main():
    # Get the current working directory
    current_dir = Path.cwd()
    print(f"Current working directory: {current_dir}")
    
    # Set up paths
    input_dir = current_dir / "raw_abacus_images"
    output_dir = current_dir / "processed_dataset"
    
    # Create processor
    processor = ImageProcessor(input_dir, output_dir)
    
    # Process images
    processor.process_images()

if __name__ == "__main__":
    main()


