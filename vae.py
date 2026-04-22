import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Hyperparameters
# =========================
latent_dim = 64
batch_size = 128
epochs = 20
lr = 1e-3

# =========================
# Data (Train + Test split)
# =========================
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =========================
# Convolutional VAE
# =========================
class ConvVAE(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # 28 -> 14
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # 14 -> 7
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        # Decoder
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 7 -> 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),   # 14 -> 28
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc(z)
        x = x.view(-1, 64, 7, 7)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

model = ConvVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# =========================
# Loss
# =========================
def loss_function(recon, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon, x, reduction="sum")
    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KL

# =========================
# Training
# =========================
for epoch in range(epochs):
    model.train()
    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for x, _ in loop:
        x = x.to(device)

        # Crop input (LEFT HALF REMOVED)
        target = x.clone()
        x[:, :, :, :14] = 0

        optimizer.zero_grad()

        recon, mu, logvar = model(x)
        loss = loss_function(recon, target, mu, logvar)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1}, Total Loss: {total_loss:.2f}")

# =========================
# TESTING (separate data)
# =========================
model.eval()

data, _ = next(iter(test_loader))
data = data.to(device)

target = data.clone()
corrupted = data.clone()

# crop center
corrupted[:, :, :, 7:21] = 0
corrupted[:, :, 7:21, :] = 0

with torch.no_grad():
    recon, _, _ = model(corrupted)

# =========================
# VISUALIZATION
# =========================
plt.figure(figsize=(10, 6))

for i in range(5):
    # input
    plt.subplot(3, 5, i+1)
    plt.imshow(corrupted[i].cpu().squeeze(), cmap="gray")
    plt.title("Input")
    plt.axis("off")

    # original
    plt.subplot(3, 5, i+6)
    plt.imshow(target[i].cpu().squeeze(), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # reconstruction
    plt.subplot(3, 5, i+11)
    plt.imshow(recon[i].cpu().squeeze(), cmap="gray")
    plt.title("Output")
    plt.axis("off")

plt.show()

# =========================
# IMAGE GENERATION
# =========================
z = torch.randn(16, latent_dim).to(device)

with torch.no_grad():
    samples = model.decode(z)

plt.figure(figsize=(6, 6))

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(samples[i].cpu().squeeze(), cmap="gray")
    plt.axis("off")

plt.show()