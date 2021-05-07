import os

import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms

from discriminator import Discriminator
from generator import Generator

CHECKPOINT_DIRECTORY = "checkpoints"


# Manually set a seed for reproducibility.
torch.manual_seed(111)

# Use CUDA device if one exists on the system.
device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_set = torchvision.datasets.MNIST(
    root="..", train=True, download=True, transform=transform
)

batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)

# Visualize the training data.
real_samples, mnist_labels = next(iter(train_loader))
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(real_samples[i].reshape(28, 28), cmap="gray_r")
    plt.xticks([])
    plt.yticks([])

discriminator = Discriminator().to(device=device)
generator = Generator().to(device=device)

lr = 0.0001
num_epochs = 50
loss_function = nn.BCELoss()

D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
G_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

# Create a checkpoints directory if one doesn't already exist.
if not os.path.exists(CHECKPOINT_DIRECTORY):
    os.mkdir(CHECKPOINT_DIRECTORY)

for epoch in range(num_epochs):
    for n, (real_samples, mnist_labels) in enumerate(train_loader):
        # Mix real images with fake images as input to the discriminator.
        real_samples = real_samples.to(device=device)
        real_samples_labels = torch.ones((batch_size, 1)).to(device=device)
        latent_space_samples = torch.randn((batch_size, 100)).to(device=device)
        fake_samples = generator(latent_space_samples)
        fake_samples_labels = torch.zeros((batch_size, 1)).to(device=device)
        all_samples = torch.cat((real_samples, fake_samples))
        all_samples_labels = torch.cat((real_samples_labels, fake_samples_labels))

        # Train the discriminator
        D_optimizer.zero_grad()
        D_output = discriminator(all_samples)
        D_loss = loss_function(D_output, all_samples_labels)
        D_loss.backward()
        D_optimizer.step()

        # Resample the latent space for training the generator.
        latent_space_samples = torch.randn((batch_size, 100)).to(device=device)

        # Train the generator.
        G_optimizer.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        G_loss = loss_function(output_discriminator_generated, real_samples_labels)
        G_loss.backward()
        G_optimizer.step()

        if n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {D_loss}")
            print(f"Epoch: {epoch} Loss G.: {G_loss}")

torch.save(generator.state_dict(), "generator.pt")
torch.save(discriminator.state_dict(), "discriminator.pt")
