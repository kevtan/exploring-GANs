from torch import nn


class Discriminator(nn.Module):
    """
    This is the discriminator described in Ch. 3 of "GANS in Action" by Jakub
    Langr and Vladimir Bok.
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), 784)
        return self.model(x)


if __name__ == "__main__":
    print(Discriminator())
