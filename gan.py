import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# Generator
class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(noise_dim, 7 * 7 * 256),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, 5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(7 * 7 * 128, 1)
        )

    def forward(self, x):
        return self.main(x)


# GAN class (with training loop)
class VanillaGAN:
    def __init__(self, noise_dim, device):
        self.noise_dim = noise_dim
        self.device = device
        self.generator = Generator(noise_dim).to(device)
        self.discriminator = Discriminator().to(device)

        # Loss and optimizers
        self.criterion = nn.BCEWithLogitsLoss()
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def train_step(self, real_images):
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)

        # Discriminator forward pass on real images
        self.discriminator_optimizer.zero_grad()
        real_outputs = self.discriminator(real_images)
        real_loss = self.criterion(real_outputs, real_labels)

        # Discriminator forward pass on fake images
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake_images = self.generator(noise)
        fake_outputs = self.discriminator(fake_images.detach())  # Detach to not update generator
        fake_loss = self.criterion(fake_outputs, fake_labels)

        # Update Discriminator
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.discriminator_optimizer.step()

        # Generator forward pass (update based on fake outputs)
        self.generator_optimizer.zero_grad()
        fake_outputs = self.discriminator(fake_images)
        g_loss = self.criterion(fake_outputs, real_labels)  # Want fake images to be classified as real
        g_loss.backward()
        self.generator_optimizer.step()

        return d_loss.item(), g_loss.item()

    def generate_and_save_images(self, epoch, noise, filename="generated_images.png"):
        self.generator.eval()
        with torch.no_grad():
            fake_images = self.generator(noise).cpu()
            fake_images = fake_images.view(fake_images.size(0), 28, 28)  # Reshape to 28x28

            fig = plt.figure(figsize=(4, 4))
            for i in range(fake_images.size(0)):
                plt.subplot(4, 4, i + 1)
                plt.imshow(fake_images[i], cmap='gray')
                plt.axis('off')

            plt.savefig(f'{filename}_epoch_{epoch + 1}.png')
            plt.show()

    def plot_loss(self, d_losses, g_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(d_losses, label='Discriminator Loss')
        plt.plot(g_losses, label='Generator Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Discriminator & Generator Losses')
        plt.legend()
        plt.show()


# DataLoader setup
def get_mnist_loader(batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Training the GAN
def train_gan(gan, data_loader, epochs=5, log_interval=100):
    d_losses, g_losses = [], []

    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(data_loader):
            real_images = real_images.to(gan.device)

            # Train the model
            d_loss, g_loss = gan.train_step(real_images)

            # Log the losses
            d_losses.append(d_loss)
            g_losses.append(g_loss)

            # Print losses at intervals
            if i % log_interval == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(data_loader)}], '
                      f'Discriminator Loss: {d_loss:.4f}, Generator Loss: {g_loss:.4f}')

        # Generate and save images at the end of each epoch
        noise = torch.randn(16, gan.noise_dim, device=gan.device)
        gan.generate_and_save_images(epoch, noise, filename="generated_images")

    # After training, plot the loss curves
    gan.plot_loss(d_losses, g_losses)


if __name__ == "__main__":
    # Hyperparameters and device setup
    NOISE_DIM = 100
    BATCH_SIZE = 256
    NUM_EPOCHS = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize GAN
    gan = VanillaGAN(noise_dim=NOISE_DIM, device=device)

    # Load dataset
    train_loader = get_mnist_loader(batch_size=BATCH_SIZE)

    # Start training
    train_gan(gan, train_loader, epochs=NUM_EPOCHS)
