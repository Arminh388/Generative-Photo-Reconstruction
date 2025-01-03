import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import vgg19
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Dataset class
class ImageDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform_lr, transform_hr):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_files = sorted(os.listdir(lr_dir))
        self.hr_files = sorted(os.listdir(hr_dir))
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        lr_image = Image.open(lr_path).convert("RGB")
        hr_image = Image.open(hr_path).convert("RGB")
        lr_image = self.transform_lr(lr_image)
        hr_image = self.transform_hr(hr_image)
        return lr_image, hr_image

# Initialize dataset and dataloaders
transform_lr = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])
transform_hr = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

hr_dir = "/home/armandobean/QMIND/Generative-Photo-Reconstruction/srgan_implementation/processed_dataset/high_res"
lr_dir = "/home/armandobean/QMIND/Generative-Photo-Reconstruction/srgan_implementation/processed_dataset/low_res"

dataset = ImageDataset(lr_dir, hr_dir, transform_lr, transform_hr)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Losses
mse_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()

# VGG Feature Extractor
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features[:36])).eval()  # Use up to relu5_4
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.features(x)

vgg_extractor = VGGFeatureExtractor().cuda()

# Initialize models
generator = Generator().cuda()
discriminator = Discriminator().cuda()

# Optimizers
gen_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)

# Training loop
epochs = 50
for epoch in range(epochs):
    for lr_imgs, hr_imgs in dataloader:
        lr_imgs = lr_imgs.cuda()
        hr_imgs = hr_imgs.cuda()

        # Train Discriminator
        sr_imgs = generator(lr_imgs)
        real_preds = discriminator(hr_imgs)
        fake_preds = discriminator(sr_imgs.detach())
        real_loss = bce_loss(real_preds, torch.ones_like(real_preds))
        fake_loss = bce_loss(fake_preds, torch.zeros_like(fake_preds))
        disc_loss = (real_loss + fake_loss) / 2

        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()

        # Train Generator
        fake_preds = discriminator(sr_imgs)
        adv_loss = bce_loss(fake_preds, torch.ones_like(fake_preds))
        content_loss = mse_loss(vgg_extractor(sr_imgs), vgg_extractor(hr_imgs))
        pixel_loss = mse_loss(sr_imgs, hr_imgs)
        gen_loss = 0.006 * adv_loss + 0.5 * content_loss + 0.5 * pixel_loss

        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}] - D Loss: {disc_loss.item():.4f}, G Loss: {gen_loss.item():.4f}")

    # Save model checkpoints
    torch.save(generator.state_dict(), f"generator_epoch_{epoch+1}.pth")
    torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch+1}.pth")
