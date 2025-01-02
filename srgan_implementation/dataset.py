import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config


class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.class_names = os.listdir(root_dir)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Iterate through directories and collect data
        for index, name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root_dir, name))
            self.data += list(zip(files, [index] * len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            img_file, label = self.data[index]
            root_and_dir = os.path.join(self.root_dir, self.class_names[label])

        # Load image
            image = Image.open(os.path.join(root_and_dir, img_file))


    # If image has alpha channel (RGBA), convert to RGB
            if image.mode == 'RGBA':
                image = image.convert('RGB')
                
            from torchvision.transforms import functional as F
            image = F.resize(image, size=(256, 256)) 
    # Convert to NumPy array after mode check
            image = np.array(image)

            image = config.both_transforms(image=image)["image"]

            high_res = config.highres_transform(image=image)["image"]

            low_res = config.lowres_transform(image=image)["image"]

        # Remove alpha channel if present (done at this stage instead of in the transform pipeline)
            if low_res.shape[0] == 4:
                low_res = low_res[:3, :, :]
            if high_res.shape[0] == 4:
                high_res = high_res[:3, :, :]

            return low_res, high_res
        except Exception as e:
            print(f"Error loading image at index {index}: {e}")
            raise

def test():
    dataset = MyImageFolder(root_dir="new_data/")
    # Set num_workers=0 if debugging multiprocessing issues
    loader = DataLoader(dataset, batch_size=1, num_workers=8)
   
    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)


if __name__ == "__main__":
    test()
