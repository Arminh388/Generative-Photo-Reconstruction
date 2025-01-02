import torch
import config
import numpy as np
from torch import nn
from torch import optim
from utils import load_checkpoint, save_checkpoint, plot_examples
from LossFunc import VGGLoss
from torch.utils.data import DataLoader
from srganclass import Generator, Discriminator
from tqdm import tqdm
from dataset import MyImageFolder

torch.backends.cudnn.benchmark = True


# Modified training function with improved loss weights and monitoring
def train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss):
    loop = tqdm(loader, leave=True)
    gen_losses = []
    disc_losses = []
    
    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)
        
        # Train Discriminator
        with torch.cuda.amp.autocast():
            fake = gen(low_res)
            disc_real = disc(high_res)
            disc_fake = disc(fake.detach())
            
            # Use label smoothing
            real_labels = torch.ones_like(disc_real) * 0.9
            fake_labels = torch.zeros_like(disc_fake)
            
            disc_loss_real = bce(disc_real, real_labels)
            disc_loss_fake = bce(disc_fake, fake_labels)
            loss_disc = disc_loss_real + disc_loss_fake

        opt_disc.zero_grad()
        loss_disc.backward()
        # Gradient clipping for discriminator
        torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
        opt_disc.step()

        # Train Generator
        with torch.cuda.amp.autocast():
            disc_fake = disc(fake)
            
            # Content loss (VGG)
            content_loss = vgg_loss(fake, high_res) * 0.01
            
            # Adversarial loss
            adversarial_loss = bce(disc_fake, torch.ones_like(disc_fake)) * 0.001
            
            # Pixel loss (L1)
            pixel_loss = torch.mean(torch.abs(fake - high_res)) * 0.1
            
            # Total generator loss
            gen_loss = content_loss + adversarial_loss + pixel_loss

        opt_gen.zero_grad()
        gen_loss.backward()
        # Gradient clipping for generator
        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        opt_gen.step()
        
        # Track losses
        gen_losses.append(gen_loss.item())
        disc_losses.append(loss_disc.item())
        
        if idx % 100 == 0:
            avg_gen_loss = sum(gen_losses[-100:]) / len(gen_losses[-100:])
            avg_disc_loss = sum(disc_losses[-100:]) / len(disc_losses[-100:])
            print(f"\nAverage Generator Loss: {avg_gen_loss:.4f}")
            print(f"Average Discriminator Loss: {avg_disc_loss:.4f}")

    return np.mean(gen_losses), np.mean(disc_losses)


def main():
    dataset = MyImageFolder(root_dir="new_data/")
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=config.NUM_WORKERS,
    )


    gen = Generator(in_channels=3).to(config.DEVICE)
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.GEN_LR, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=config.DISC_LR, betas=(0.9, 0.999))
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()


    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            opt_gen,
            config.GEN_LR,
        )
        load_checkpoint(
           config.CHECKPOINT_DISC, disc, opt_disc, config.DISC_LR,
        )

    for epoch in range(config.NUM_EPOCHS):
        print("Current epoch:" ,epoch)
        train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss)

        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
        
        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)


if __name__ == "__main__":
    main()
