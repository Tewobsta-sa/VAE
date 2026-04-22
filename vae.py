import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================
# Hyperparameters
# =========================
input_dim = 784   # 28x28 images
hidden_dim = 800
latent_dim = 100
batch_size = 128
epochs = 50
lr = 1e-3

# =========================
# Data
# =========================
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# =========================
# VAE Model
# =========================
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# =========================
# Loss Function
# =========================
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KL


# =========================
# Training
# =========================
model = VAE()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    total_loss = 0

    loop = tqdm(train_loader, leave=True)

    for data, _ in loop:
        data = data.view(-1, 784)
        original = data.clone()
        data[:, :392] = 0 

        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss = loss_function(recon, original, mu, logvar)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update progress bar
        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} Total Loss: {total_loss:.2f}")


# =========================
# Test Reconstruction
# =========================
data_iter = iter(train_loader)
images, _ = next(data_iter)

images_flat = images.view(-1, 784)
recon_images, _, _ = model(images_flat)

# Show original vs reconstructed
plt.figure(figsize=(10, 6))

for i in range(5):
    # Cropped input
    plt.subplot(3, 5, i+1)
    plt.imshow(data[i].view(28,28).detach().numpy(), cmap='gray')
    plt.title("Input")
    plt.axis('off')

    # Original
    plt.subplot(3, 5, i+6)
    plt.imshow(original[i].view(28,28).detach().numpy(), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Reconstructed
    plt.subplot(3, 5, i+11)
    plt.imshow(recon[i].view(28,28).detach().numpy(), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

plt.show()


# =========================
# Generate New Images
# =========================
z = torch.randn(10, latent_dim)
generated = model.decode(z)

plt.figure(figsize=(8, 2))

for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(generated[i].view(28, 28).detach().numpy(), cmap='gray')
    plt.axis('off')

plt.show()