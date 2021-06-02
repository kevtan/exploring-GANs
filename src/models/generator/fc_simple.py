from torch import nn


class Generator(nn.Module):
    """
    This is the generator described in Ch. 3 of "GANS in Action" by Jakub Langr
    and Vladimir Bok.
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 1, 28, 28)
        return output


if __name__ == "__main__":
    print(Generator())
