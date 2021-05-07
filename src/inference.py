import torch
from matplotlib import pyplot as plt

from generator import Generator

# Restore the generator from the end of training.
generator = Generator()
generator.load_state_dict(torch.load("generator.pt"))
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
plt.show()
