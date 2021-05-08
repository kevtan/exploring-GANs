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
   pip install -r requirements_macos.txt
   pip install -r requirements_linux.txt
   ```

## Cleanliness

Use `black` and `isort` to keep the code clean! They are installed as part of
the development dependencies.

## Training

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

