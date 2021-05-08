import os

import torch
from matplotlib import pyplot as plt

from generator import Generator

def main():
    # Prompt the user for a checkpoint to restore from.
    # TODO: give the user a list of recent checkpoints to choose from
    checkpoint_file = input("Checkpoint: ")

    # Check to see that the checkpoint actually exists.
    if not os.path.exists(checkpoint_file):
        print("すみません, that checkpoint doesn't exist.")
        return

    # Load in the checkpoint.
    checkpoint = torch.load(checkpoint_file)

    # Restore the generator from the end of training.
    generator = Generator()
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    batch_size = 32
    latent_space_samples = torch.randn(batch_size, 100)
    generated_samples = generator(latent_space_samples)
    generated_samples = generated_samples.detach()  # TODO: what does detach() do?

    # Plot the results
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
        plt.xticks([])
        plt.yticks([])

    # TODO: figure out a better place to save this image
    image_path = "performance.png"
    plt.savefig(image_path)

if __name__ == "__main__":
    main()
