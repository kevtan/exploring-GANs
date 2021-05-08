import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms
from tqdm import tqdm, trange

from discriminator import Discriminator
from generator import Generator

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

for epoch in trange(num_epochs, desc="Epochs"):
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
            tqdm.write(f"Epoch: {epoch:<5}", end="")
            tqdm.write(f"Discriminator Loss: {D_loss:5.3f}", end="\t")
            tqdm.write(f"Generator Loss: {G_loss:5.3f}")

torch.save(generator.state_dict(), "generator.pt")
torch.save(discriminator.state_dict(), "discriminator.pt")
