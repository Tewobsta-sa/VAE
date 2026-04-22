# VAE for MNIST — inpainting & generation

A small Variational Autoencoder that can

1. Reconstruct MNIST digits.
2. **Inpaint** cropped / masked regions (fill in the missing half of an image).
3. **Generate** new digits by sampling from the prior.

MNIST is downloaded automatically via `torchvision` and kept under `./data/`.
The 60k MNIST training set is split 90/10 into train and validation. The
10k MNIST test set is never touched during training and is what is used
for reconstruction and inpainting demos.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python vae.py train --epochs 20
```

Checkpoints go to `./outputs/vae.pt`. The best-val checkpoint is kept.

During training, with 50% probability per batch, a random crop mask is
applied to each image and the model is asked to fill it in. The other
50% of batches are full reconstruction. This single model therefore
handles both full reconstruction and arbitrary-mask inpainting.

## Fill cropped images (inpainting)

```bash
python vae.py inpaint --mask top     # fill in the top half
python vae.py inpaint --mask bottom
python vae.py inpaint --mask left
python vae.py inpaint --mask right
python vae.py inpaint --mask center  # fill in a central square
```

Produces `./outputs/inpaint_<mask>.png` with three rows: original,
cropped input, and the filled output.

## Generate new images

```bash
python vae.py generate --n 16
```

Produces `./outputs/generated.png` — a grid of novel digits sampled from
`z ~ N(0, I)` and decoded.

## Reconstruct (no masking)

```bash
python vae.py reconstruct --n 8
```

Produces `./outputs/reconstruct.png` (input row + reconstruction row).

## Notes

- All three commands (`reconstruct`, `inpaint`, `generate`) evaluate on
  the held-out MNIST test set, not on training data.
- The model is a small MLP VAE (hidden dim 512, latent dim 32) so it
  trains quickly on CPU. On GPU training is almost instant.
- The encoder is mask-aware: it takes both `x * mask` and `mask` as
  input, so a pixel that is `0` because it was cropped is distinguishable
  from a pixel that is genuinely black.
