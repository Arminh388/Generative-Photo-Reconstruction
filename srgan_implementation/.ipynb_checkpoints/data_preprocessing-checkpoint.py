

from PIL import Image
import os

def convert_jpg_to_png(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all the files in the directory
    for img_file in os.listdir(input_dir):
        if img_file.lower().endswith('.jpg'):  # Check if the file is a JPG
            try:
                # Open the JPG image
                img_path = os.path.join(input_dir, img_file)
                img = Image.open(img_path)

                # Convert image to 'RGB' if necessary (PNG doesn't support 'RGBA' transparency for non-transparent images)
                img = img.convert('RGB')

                # Save the image as PNG
                new_img_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}.png")
                img.save(new_img_path, 'PNG')

                print(f"Converted {img_file} to PNG")

            except Exception as e:
                print(f"Failed to convert {img_file}: {e}")

# Example usage:
input_directory = 'processed_dataset/high_res'  # Path to folder with JPG images
output_directory = 'saved'  # Path where PNG images will be saved

convert_jpg_to_png(input_directory, output_directory)
