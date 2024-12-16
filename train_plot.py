import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_msssim import ssim
import matplotlib.pyplot as plt

from VideoMatteDataset import VideoMatteDataset
from gan_model import GeneratorWithVGG, DiscriminatorWithEfficientNet
from training_utils import wgan_loss, compute_gradient_penalty

# Setup dataset and dataloader
dataset = VideoMatteDataset(
    videomatte_dir='/content/drive/MyDrive/comp646/dataset/VideoMatte240K_JPEG_SD/train',
    background_image_dir='/content/drive/MyDrive/comp646/dataset/Backgrounds',
    size=512,
    seq_length=1,
    seq_sampler=TrainFrameSampler(),
)

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize model, optimizer, and other training parameters
generator = GeneratorWithVGG().to(device)
discriminator = DiscriminatorWithEfficientNet().to(device)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

# Use mixed precision
scaler = torch.cuda.amp.GradScaler()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for i, (fgr, bgr) in enumerate(dataloader):
        fgr, bgr = fgr.to(device), bgr.to(device)

        # Discriminator update
        optimizer_D.zero_grad()
        real_outputs = discriminator(fgr, bgr)
        fake_bgr = generator(fgr, bgr)
        fake_outputs = discriminator(fgr, fake_bgr.detach())
        d_loss_real = wgan_loss(real_outputs, True)
        d_loss_fake = wgan_loss(fake_outputs, False)
        gradient_penalty = compute_gradient_penalty(discriminator, fgr, bgr, fgr, fake_bgr, device)
        d_loss = d_loss_real + d_loss_fake + gradient_penalty
        d_loss.backward()
        optimizer_D.step()

        # Generator update
        if i % 5 == 0:
            optimizer_G.zero_grad()
            regenerated_outputs = discriminator(fgr, fake_bgr)
            g_loss = wgan_loss(regenerated_outputs, True)
            g_loss.backward()
            optimizer_G.step()

        # Evaluate and log
        with torch.no_grad():
            ssim_adapted = ssim(fgr.float(), fake_bgr.float(), data_range=1.0, size_average=True)
            ssim_original = ssim(fgr.float(), bgr.float(), data_range=1.0, size_average=True)
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        ssim_adapted_scores.append(ssim_adapted.item())
        ssim_original_scores.append(ssim_original.item())

        # Print training progress
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, SSIM Adapted: {ssim_adapted.item():.4f}, SSIM Original: {ssim_original.item():.4f}")

    # Plot progress after each epoch
    plot_losses_and_ssim(d_losses, g_losses, ssim_adapted_scores, ssim_original_scores)

print("Training complete.")
