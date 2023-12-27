""" 
Training  A WGAN network on MNIST dataset with Discriminator and Generator imported from model.py
"""
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import summary
from model import Discriminator, Generator, initializer_weights
from utils import gradient_penalty


# Hyperparameters etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# LEARNING_RATE = 5e-5
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NUM_CLASSES = 10
GEN_EMBEDDING = 100
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
# WEIGHT_CLIP = 0.01
LAMBDA_GP = 10



transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        )
    ]
)

# dataset = datasets.MNIST(root='datasets/', train=True, transform=transforms, download=True) 
dataset = datasets.ImageFolder(root="location to dataset", transform=transforms)

dataloader = DataLoader(dataset, dataset, batch_size=BATCH_SIZE, shuffle=True)
gen =  Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMAGE_SIZE, GEN_EMBEDDING).to(device=device)
critic =  Generator(CHANNELS_IMG, FEATURES_DISC, NUM_CLASSES, IMAGE_SIZE).to(device=device)
initializer_weights(gen)
initializer_weights(critic)

# opt_gen = optim.RMSprop(gen.parameters(), lr=LEARNING_RATE, )
# opt_critic = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE, )
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

fixed_noise = torch.randn(32, Z_DIM, 1,1).to(device)
writer_real = summary(f'logs/real')
writer_fake = summary(f'logs/fake')
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, labels) in enumerate(dataloader):
        real = real.to(device)
        cur_batch_size = real.shape[0]
        labels = labels.to(device) 
        
        for _ in range():
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake = gen(noise, labels)
            critic_real = critic(real, labels).reshape(-1)
            critic_fake = critic(fake, labels).reshape(-1)
            gp = gradient_penalty(critic, labels,real, fake, device)
            # loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake) )
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake) ) + LAMBDA_GP * gp)
            critic.zero_grad()
            loss_critic.backward(retain_graph= True)
            opt_critic.step()

            # for p in critic.parameters():
            #     p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        #  Train Generator: min -E[critic(gen_fake)]
        output = critic(fake, labels).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()


       

        # Print Losses Occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                    Loss: D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(noise, labels)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_real.add_image("Fake", img_grid_fake, global_step=step)

            step +=1
