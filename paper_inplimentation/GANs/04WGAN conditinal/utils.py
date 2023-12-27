import torch
import torch.nn as nn


def gradient_penalty(critic,labels ,real, fake, device='cpu'):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.randn((BATCH_SIZE, 1, 1, 1)).reshape(1, C, H, W).to(device)
    interpolated_image = real * epsilon + fake * (1 - epsilon)

    # Calculate critic scores
    mixed_scores = critic(interpolated_image, labels)

    gradient = torch.autograd.grad(
        inputs=interpolated_image,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph = True,
        retain_graph=True 
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty  = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty