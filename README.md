# Variational Autoencoder (VAE): MNIST Image Completion

A Variational Autoencoder built from scratch in **PyTorch** and trained on **MNIST**.
During training the **top half of every image is blanked out**. The model learns to rebuild the **full digit** from only the bottom half, which makes it a simple image-completion (inpainting) model.

## How it works
- **Encoder:** 784 → 800 (ReLU) → mean and log-variance of a 100-dimensional latent space
- **Reparameterization trick:** `z = mu + eps * std`, so the model can still be trained with backpropagation
- **Decoder:** 100 → 800 (ReLU) → 784 (Sigmoid)
- **Loss:** reconstruction loss (binary cross-entropy against the *original* full image) + KL divergence
- **Training:** Adam (lr 1e-3), batch size 128, 50 epochs, with a tqdm progress bar

After training, the script shows the cropped input, the original and the reconstructed images next to each other.

## Run it
```bash
pip install torch torchvision matplotlib tqdm
python vae.py
```
MNIST downloads automatically into `./data`.

## Tech
Python · PyTorch · torchvision · Matplotlib
