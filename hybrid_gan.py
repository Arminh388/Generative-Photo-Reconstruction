import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Generator for Low-Resolution Images (GAN)
class LowResGenerator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(LowResGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, img_shape),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Generator for Super-Resolution (SRGAN)
class HighResGenerator(nn.Module):
    def __init__(self):
        super(HighResGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
            *[ResidualBlock(64) for _ in range(16)],
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Residual Block for SRGAN
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return out

# Discriminator for Low-Resolution Images (GAN)
class LowResDiscriminator(nn.Module):
    def __init__(self, img_shape):
        super(LowResDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_shape, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

# Discriminator for High-Resolution Images (SRGAN)
class HighResDiscriminator(nn.Module):
    def __init__(self):
        super(HighResDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            *[nn.Sequential(
                nn.Conv2d(64 * (2 ** i), 128 * (2 ** i), kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128 * (2 ** i)),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128 * (2 ** i), 128 * (2 ** i), kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128 * (2 ** i)),
                nn.LeakyReLU(0.2)
            ) for i in range(3)],
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Hyperparameters
latent_dim = 100
low_res_shape = 28 * 28  # MNIST low-res image size
batch_size = 64
epochs = 100
lr = 0.0002

# Initialize networks
low_res_generator = LowResGenerator(latent_dim, low_res_shape)
high_res_generator = HighResGenerator()
low_res_discriminator = LowResDiscriminator(low_res_shape)
high_res_discriminator = HighResDiscriminator()

# Optimizers
optimizer_G_low = optim.Adam(low_res_generator.parameters(), lr=lr)
optimizer_G_high = optim.Adam(high_res_generator.parameters(), lr=lr)
optimizer_D_low = optim.Adam(low_res_discriminator.parameters(), lr=lr)
optimizer_D_high = optim.Adam(high_res_discriminator.parameters(), lr=lr)

# Loss functions
adversarial_loss = nn.BCELoss()
content_loss = nn.L1Loss()

# Data loading (MNIST for low-res)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    for i, (low_res_imgs, _) in enumerate(dataloader):
        # Adversarial ground truths
        valid = torch.ones(low_res_imgs.size(0), 1)
        fake = torch.zeros(low_res_imgs.size(0), 1)

        # Generate low-res images
        z = torch.randn(low_res_imgs.size(0), latent_dim)
        gen_low_res = low_res_generator(z)

        # Generate high-res images from low-res
        gen_low_res_reshaped = gen_low_res.view(-1, 1, 28, 28)
        gen_high_res = high_res_generator(gen_low_res_reshaped)

        # Train Low-Res Discriminator
        optimizer_D_low.zero_grad()
        real_loss_low = adversarial_loss(low_res_discriminator(low_res_imgs.view(low_res_imgs.size(0), -1)), valid)
        fake_loss_low = adversarial_loss(low_res_discriminator(gen_low_res.detach()), fake)
        d_loss_low = (real_loss_low + fake_loss_low) / 2
        d_loss_low.backward()
        optimizer_D_low.step()

        # Train High-Res Discriminator
        optimizer_D_high.zero_grad()
        real_loss_high = adversarial_loss(high_res_discriminator(low_res_imgs), valid)
        fake_loss_high = adversarial_loss(high_res_discriminator(gen_high_res.detach()), fake)
        d_loss_high = (real_loss_high + fake_loss_high) / 2
        d_loss_high.backward()
        optimizer_D_high.step()

        # Train Generators
        optimizer_G_low.zero_grad()
        optimizer_G_high.zero_grad()
        g_loss_low = adversarial_loss(low_res_discriminator(gen_low_res), valid)
        g_loss_high = adversarial_loss(high_res_discriminator(gen_high_res), valid) + 0.001 * content_loss(gen_high_res, low_res_imgs)
        g_loss = g_loss_low + g_loss_high
        g_loss.backward()
        optimizer_G_low.step()
        optimizer_G_high.step()

        if i % 100 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                  f"[D Low loss: {d_loss_low.item()}] [D High loss: {d_loss_high.item()}] "
                  f"[G Low loss: {g_loss_low.item()}] [G High loss: {g_loss_high.item()}]")

    # Save generated images
    if epoch % 10 == 0:
        with torch.no_grad():
            z = torch.randn(16, latent_dim)
            gen_low_res = low_res_generator(z).view(-1, 1, 28, 28)
            gen_high_res = high_res_generator(gen_low_res)
            plt.figure(figsize=(8, 8))
            for i in range(16):
                plt.subplot(4, 4, i+1)
                plt.imshow(gen_high_res[i].squeeze(), cmap="gray")
                plt.axis("off")
            plt.show()
