# FlairGAN

マディさんとケビンさんの implementation of StyleGAN.

## Instructions

1. Create a virtual environment in the root directory using the command:

   ```bash
   python3 -m venv .venv
   ```

2. Activate the virtual environment using the command:

   ```bash
   source .venv/bin/activate
   ```

3. Install the required dependencies using one of the following commands:

   ```bash
   pip3 install -r requirements_macos.txt
   pip3 install -r requirements_linux.txt
   ```

## Cleanliness

Use `black` and `isort` to keep the code clean! They are installed as part of
the development dependencies.

## Training

We experimented with training on various platforms and found that the training speed varied widely between platforms. The following numbers were obtained using the original FCGAN network architecture.

- On a 2017 MacBook Pro with a 3.1 GHz Dual-Core Intel Core i5 processor and 8 GB 2133 MHz LPDDR3 RAM, each epoch took around 89.21 s.
- With a AMD Ryzen 7 3700X 8-Core Processor and 32 GB 3200 MHz DDR4 RAM, each epoch took around 56.77 s.
- Using an NVIDIA 3060 Ti GPU, each epoch took just 15.55 s.

## Inference

To see the inference results on Linux:

- Make sure you SSH-ed with the `-YC` options to enable GUI commands.
- Install `imagemagick` with `sudo apt install imagemagick`.
- This gives you a tool called `display`.
- Use `display` to display your image from the command line.

## Utilities

There is currently a utility script `info.py` that informs you whether or not
CUDA is enabled on your system.

```text
❯ python info.py
CUDA:   Yes
```

