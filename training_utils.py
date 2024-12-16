import torch

def wgan_loss(output, is_real):
    return -torch.mean(output) if is_real else torch.mean(output)

def compute_gradient_penalty(discriminator, real_samples_fgr, real_samples_bgr, fake_samples_fgr, fake_samples_bgr, device):
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples_fgr.size(0), 1, 1, 1), device=device)
    interpolates_fgr = (alpha * real_samples_fgr + (1 - alpha) * fake_samples_fgr).requires_grad_(True)
    interpolates_bgr = (alpha * real_samples_bgr + (1 - alpha) * fake_samples_bgr).requires_grad_(True)

    d_interpolates = discriminator(interpolates_fgr, interpolates_bgr)

    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=[interpolates_fgr, interpolates_bgr],
                                    grad_outputs=torch.ones(d_interpolates.size(), device=device),
                                    create_graph=True, retain_graph=True, only_inputs=True)
    gradients = torch.sqrt(torch.sum(gradients[0] ** 2, dim=(1, 2, 3)) + torch.sum(gradients[1] ** 2, dim=(1, 2, 3)))
    gradient_penalty = ((gradients - 1) ** 2).mean() * 10  # 10 is lambda, the penalty coefficient
    return gradient_penalty