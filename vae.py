"""Variational Autoencoder for MNIST.

Features:
    * Proper train / val / test split (MNIST train is split 90/10 into
      train/val; MNIST test is held out and never seen during training).
    * Image reconstruction.
    * Inpainting: fill in cropped / masked regions. The encoder receives
      both the (masked) image and the mask, so the same network handles
      full reconstruction and arbitrary masks.
    * Unconditional generation: sample new digits from the prior.

CLI:
    python vae.py train
    python vae.py reconstruct
    python vae.py inpaint --mask top
    python vae.py generate

Outputs (checkpoints + sample images) go under ./outputs/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMG_SIZE = 28
INPUT_DIM = IMG_SIZE * IMG_SIZE
HIDDEN_DIM = 512
LATENT_DIM = 32
DEFAULT_EPOCHS = 20
DEFAULT_BATCH = 128
DEFAULT_LR = 1e-3
VAL_FRAC = 0.1
DATA_DIR = "./data"
OUT_DIR = Path("./outputs")
CKPT_PATH = OUT_DIR / "vae.pt"
SEED = 42


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class VAE(nn.Module):
    """MLP VAE that is aware of an observation mask.

    The encoder sees ``(x * mask, mask)`` so it can distinguish between
    a truly black pixel and an unknown pixel. With ``mask == 1`` the
    model behaves as a plain VAE; with a partial mask it learns to
    inpaint the missing region.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        latent_dim: int = LATENT_DIM,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([x * mask, mask], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decoder(z))

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, mask)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bce = F.binary_cross_entropy(recon, target, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld, bce, kld


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------
def all_ones_mask(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.ones(batch_size, INPUT_DIM, device=device)


def random_mask(batch_size: int, device: torch.device) -> torch.Tensor:
    """Per-sample random crop mask used during training.

    1 = observed, 0 = unknown (cropped out).
    """
    mask = torch.ones(batch_size, IMG_SIZE, IMG_SIZE, device=device)
    kinds = torch.randint(0, 4, (batch_size,))
    for i in range(batch_size):
        k = int(kinds[i].item())
        if k == 0:  # top half
            mask[i, : IMG_SIZE // 2, :] = 0
        elif k == 1:  # bottom half
            mask[i, IMG_SIZE // 2 :, :] = 0
        elif k == 2:  # left half
            mask[i, :, : IMG_SIZE // 2] = 0
        else:  # random rectangular block
            h = int(torch.randint(8, 16, (1,)).item())
            w = int(torch.randint(8, 16, (1,)).item())
            y = int(torch.randint(0, IMG_SIZE - h + 1, (1,)).item())
            x = int(torch.randint(0, IMG_SIZE - w + 1, (1,)).item())
            mask[i, y : y + h, x : x + w] = 0
    return mask.view(batch_size, -1)


def fixed_mask(mode: str, batch_size: int, device: torch.device) -> torch.Tensor:
    mask = torch.ones(batch_size, IMG_SIZE, IMG_SIZE, device=device)
    if mode == "top":
        mask[:, : IMG_SIZE // 2, :] = 0
    elif mode == "bottom":
        mask[:, IMG_SIZE // 2 :, :] = 0
    elif mode == "left":
        mask[:, :, : IMG_SIZE // 2] = 0
    elif mode == "right":
        mask[:, :, IMG_SIZE // 2 :] = 0
    elif mode == "center":
        mask[:, 8:20, 8:20] = 0
    else:
        raise ValueError(f"unknown mask mode: {mode}")
    return mask.view(batch_size, -1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def get_loaders(batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    val_size = int(len(train_full) * VAL_FRAC)
    train_size = len(train_full) - val_size
    gen = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(train_full, [train_size, val_size], generator=gen)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------
def run_epoch(
    model: VAE,
    loader: DataLoader,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    train: bool,
    mask_prob: float = 0.5,
) -> tuple[float, float, float]:
    model.train(train)
    total = total_bce = total_kld = 0.0
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    desc = "train" if train else "val"
    with ctx:
        for imgs, _ in tqdm(loader, desc=desc, leave=False):
            imgs = imgs.to(device).view(imgs.size(0), -1)
            bs = imgs.size(0)
            if train and torch.rand(1).item() < mask_prob:
                mask = random_mask(bs, device)
            else:
                mask = all_ones_mask(bs, device)

            recon, mu, logvar = model(imgs, mask)
            loss, bce, kld = vae_loss(recon, imgs, mu, logvar)

            if train:
                assert optimizer is not None
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()
            n += bs
    return total / n, total_bce / n, total_kld / n


def train_cmd(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    train_loader, val_loader, _ = get_loaders(args.batch_size)
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr, tr_bce, tr_kld = run_epoch(model, train_loader, optimizer, device, train=True)
        va, va_bce, va_kld = run_epoch(model, val_loader, None, device, train=False)
        print(
            f"epoch {epoch:02d} | "
            f"train {tr:.4f} (bce {tr_bce:.2f} kld {tr_kld:.2f}) | "
            f"val {va:.4f} (bce {va_bce:.2f} kld {va_kld:.2f})"
        )
        if va < best_val:
            best_val = va
            torch.save(
                {"model": model.state_dict(), "latent_dim": model.latent_dim},
                CKPT_PATH,
            )
            print(f"  saved checkpoint to {CKPT_PATH} (val {va:.4f})")

    print(f"done. best val loss = {best_val:.4f}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def load_model(device: torch.device) -> VAE:
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"No checkpoint at {CKPT_PATH}. Run `python vae.py train` first."
        )
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    model = VAE(latent_dim=ckpt.get("latent_dim", LATENT_DIM)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _test_batch(batch_size: int, n: int, device: torch.device) -> torch.Tensor:
    _, _, test_loader = get_loaders(batch_size)
    imgs, _ = next(iter(test_loader))
    n = min(n, imgs.size(0))
    return imgs[:n].to(device).view(n, -1)


def reconstruct_cmd(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    imgs = _test_batch(args.batch_size, args.n, device)
    n = imgs.size(0)
    with torch.no_grad():
        recon, _, _ = model(imgs, all_ones_mask(n, device))
    grid = torch.cat(
        [
            imgs.view(n, 1, IMG_SIZE, IMG_SIZE),
            recon.view(n, 1, IMG_SIZE, IMG_SIZE),
        ],
        dim=0,
    )
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    path = OUT_DIR / "reconstruct.png"
    save_image(grid, path, nrow=n)
    print(f"saved {path} (row 1: input, row 2: reconstruction)")


def inpaint_cmd(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    imgs = _test_batch(args.batch_size, args.n, device)
    n = imgs.size(0)
    mask = fixed_mask(args.mask, n, device)
    with torch.no_grad():
        recon, _, _ = model(imgs, mask)
        # Splice: keep observed pixels, use recon for masked region.
        filled = imgs * mask + recon * (1 - mask)
    grid = torch.cat(
        [
            imgs.view(n, 1, IMG_SIZE, IMG_SIZE),
            (imgs * mask).view(n, 1, IMG_SIZE, IMG_SIZE),
            filled.view(n, 1, IMG_SIZE, IMG_SIZE),
        ],
        dim=0,
    )
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    path = OUT_DIR / f"inpaint_{args.mask}.png"
    save_image(grid, path, nrow=n)
    print(f"saved {path} (rows: original, cropped input, filled output)")


def generate_cmd(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    torch.manual_seed(args.seed)
    z = torch.randn(args.n, model.latent_dim, device=device)
    with torch.no_grad():
        samples = model.decode(z).view(-1, 1, IMG_SIZE, IMG_SIZE)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    path = OUT_DIR / "generated.png"
    nrow = max(1, int(args.n ** 0.5))
    save_image(samples, path, nrow=nrow)
    print(f"saved {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VAE for MNIST: inpainting + generation.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="train the VAE")
    p_train.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p_train.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p_train.add_argument("--lr", type=float, default=DEFAULT_LR)
    p_train.set_defaults(func=train_cmd)

    p_rec = sub.add_parser("reconstruct", help="reconstruct held-out test images")
    p_rec.add_argument("--n", type=int, default=8)
    p_rec.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p_rec.set_defaults(func=reconstruct_cmd)

    p_inp = sub.add_parser("inpaint", help="fill in cropped / masked test images")
    p_inp.add_argument("--n", type=int, default=8)
    p_inp.add_argument(
        "--mask",
        type=str,
        default="top",
        choices=["top", "bottom", "left", "right", "center"],
    )
    p_inp.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p_inp.set_defaults(func=inpaint_cmd)

    p_gen = sub.add_parser("generate", help="sample new digits from the prior")
    p_gen.add_argument("--n", type=int, default=16)
    p_gen.add_argument("--seed", type=int, default=SEED)
    p_gen.set_defaults(func=generate_cmd)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
