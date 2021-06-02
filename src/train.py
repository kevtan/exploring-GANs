import os
from datetime import date, datetime
from typing import Optional

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm, trange

from models.discriminator.fc_simple import Discriminator
from models.generator.fc_simple import Generator

# Manually set a seed for reproducibility.
torch.manual_seed(111)

#
# Training hyperparameters.
#
lr = 0.0001
num_epochs = 200

#
# Checkpoint to resume training from.
#
# Example: checkpoints/2021-06-02/01:09:06/epoch-49.tar
#
checkpoint: Optional[str] = None
tensorboard_checkpoint: Optional[str] = None

checkpoint_dict: Optional[dict] = None

if checkpoint is not None:
    checkpoint_dict = torch.load(checkpoint)

# Create a TensorBoard writer.
writer = SummaryWriter(tensorboard_checkpoint)

# Use CUDA device if one exists on the system.
device = None
if torch.cuda.is_available():
    print("Training on the GPU")
    device = torch.device("cuda")
else:
    print("Training on the CPU")
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
# real_samples, mnist_labels = next(iter(train_loader))
# for i in range(16):
#     ax = plt.subplot(4, 4, i + 1)
#     plt.imshow(real_samples[i].reshape(28, 28), cmap="gray_r")
#     plt.xticks([])
#     plt.yticks([])

experiment_directory: Optional[str] = None
if checkpoint is None:
    # Start training from the beginning.
    ROOT_CHECKPOINTS_DIRECTORY = "checkpoints"
    today = date.today().strftime("%Y-%m-%d")
    today_directory = f"{ROOT_CHECKPOINTS_DIRECTORY}/{today}"
    hour_minute_second = datetime.now().strftime("%H:%M:%S")
    experiment_directory = f"{today_directory}/{hour_minute_second}"
    if not os.path.exists(ROOT_CHECKPOINTS_DIRECTORY):
        os.mkdir(ROOT_CHECKPOINTS_DIRECTORY)
    if not os.path.exists(today_directory):
        os.mkdir(today_directory)
    if not os.path.exists(experiment_directory):
        os.mkdir(experiment_directory)
else:
    # Start training from a checkpoint
    last_sep_pos = checkpoint.rfind("/")
    experiment_directory = checkpoint[:last_sep_pos]
print(f"Experiment Directory: {experiment_directory}")

discriminator = Discriminator().to(device=device)
generator = Generator().to(device=device)

if checkpoint is not None:
    discriminator.load_state_dict(checkpoint_dict["discriminator_state_dict"])
    generator.load_state_dict(checkpoint_dict["generator_state_dict"])

loss_function = nn.BCELoss()

D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
G_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

if checkpoint is not None:
    D_optimizer.load_state_dict(checkpoint_dict["discriminator_optimizer_state_dict"])
    G_optimizer.load_state_dict(checkpoint_dict["generator_optimizer_state_dict"])

# Choose a fixed set of noise vectors to track the generator's progress for
# this experiment. This set of noise vectors will be saved in the checkpoint
# directory for this experiment in a file named `fixed_noise_vectors.csv`.
FIXED_NOISE_VECTORS_NUMBER = 20
FIXED_NOISE_VECTORS_FILENAME = "fixed_noise_vectors.csv"
fixed_noise_vector_dimension = 100

fixed_noise_vectors: Optional[torch.Tensor] = None
if checkpoint is None:
    # Create new fixed noise vectors.
    fixed_noise_vectors = torch.randn(
        (FIXED_NOISE_VECTORS_NUMBER, fixed_noise_vector_dimension)
    ).to(device=device)
    np.savetxt(
        f"{experiment_directory}/{FIXED_NOISE_VECTORS_FILENAME}",
        fixed_noise_vectors.cpu().numpy(),
    )
else:
    # Restore old fixed noise vectors.
    fixed_noise_vectors = torch.tensor(
        np.loadtxt(f"{experiment_directory}/{FIXED_NOISE_VECTORS_FILENAME}"),
        dtype=torch.float,
    ).to(device=device)


epoch_offset: int = 0
if checkpoint is not None:
    last_hyphen = checkpoint.rfind("-")
    last_period = checkpoint.rfind(".")
    epoch_offset = int(checkpoint[last_hyphen + 1 : last_period]) + 1

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
            tqdm.write(f"Epoch: {epoch_offset + epoch:<5}", end="")
            tqdm.write(f"Discriminator Loss: {D_loss:5.3f}", end="\t")
            tqdm.write(f"Generator Loss: {G_loss:5.3f}")

    # Feed fixed noise vectors into the generator and save results.
    fixed_noise_vector_images = generator(fixed_noise_vectors).detach()
    image_grid = torchvision.utils.make_grid(
        fixed_noise_vector_images, normalize=True, nrow=5
    )
    writer.add_image(
        tag="generation_results",
        img_tensor=image_grid,
        global_step=epoch_offset + epoch,
    )

    # Log generator and discriminator loss every epoch.
    writer.add_scalar(
        tag="generator_loss",
        scalar_value=G_loss.item(),
        global_step=epoch_offset + epoch,
    )
    writer.add_scalar(
        tag="discriminator_loss",
        scalar_value=D_loss.item(),
        global_step=epoch_offset + epoch,
    )

    # Create a checkpoint at every epoch
    checkpoint_data = {
        "epoch": epoch_offset + epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "generator_optimizer_state_dict": G_optimizer.state_dict(),
        "discriminator_optimizer_state_dict": D_optimizer.state_dict(),
        "generator_loss": G_loss.item(),
        "discriminator_loss": D_loss.item(),
    }
    checkpoint_name = f"epoch-{epoch_offset + epoch}.tar"
    torch.save(checkpoint_data, f"{experiment_directory}/{checkpoint_name}")
