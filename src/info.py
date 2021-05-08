from torch import cuda
from termcolor import cprint


def main():
    print("CUDA:\t", end="")
    if cuda.is_available():
        cprint("Yes", "green")
    else:
        cprint("No", "red")


if __name__ == "__main__":
    main()
