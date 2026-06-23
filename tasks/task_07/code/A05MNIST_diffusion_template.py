import torch  # For Pytorch methods
import torch.nn as nn
import torch.optim as optim  # For Optimizer
from torch.utils.data import DataLoader  # For Data Loader
from torch.utils.tensorboard import SummaryWriter  # For Tensor Board Visualization

import torchvision
from torchvision import transforms  # For image transforms
import torchvision.datasets as datasets  # For Data Set

from denoising_diffusion_pytorch import Unet, GaussianDiffusion

# Hyperparameters
LEARNING_RATE = 4e-4
BATCH_SIZE = 128  # Batch size
N_EPOCHS = 100
IMAGE_SIZE = 28
TIME_STEPS = 1000
SAMPLING_TIMESTEPS = 250


# we define a transform that converts the image to tensor
myTransforms = transforms.Compose([transforms.ToTensor()])

# the MNIST dataset is available through torchvision.datasets
print("loading MNIST digits dataset")
dataset = datasets.MNIST(
    root="dataset/", transform=myTransforms, download=True)
# let's create a dataloader to load the data in batches
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.MNIST(
    root="dataset/", train=False, download=False, transform=myTransforms)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


DIM = 32
DIM_MULTS = (1, 2, 5)
model = Unet(dim=DIM, dim_mults=DIM_MULTS, flash_attn=False, channels=1)

diffusion = GaussianDiffusion(
    model,
    image_size=IMAGE_SIZE,
    timesteps=TIME_STEPS,  # number of steps
    # number of sampling timesteps (using ddim for faster inference [see ddim paper])
    sampling_timesteps=SAMPLING_TIMESTEPS,
)

optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for epoch in range(N_EPOCHS):
    # implement training loop. You get the loss by calling the diffusion function
    # `loss = diffusion(training_images)`
    pass

# you can obtain sampled images (i.e. the backward pass) by calling the sample function
# `sampled_images = diffusion.sample(batch_size = 4)`
