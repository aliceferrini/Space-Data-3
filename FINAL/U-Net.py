import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.model_selection import KFold
from datetime import datetime


def _double_conv(in_ch, out_ch, dropout_rate):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Dropout2d(dropout_rate),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CNN(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, dropout_rate=0.2):
        super(CNN, self).__init__()
        c = base_channels

        # Encoder
        self.enc1 = _double_conv(in_channels, c,     dropout_rate)
        self.enc2 = _double_conv(c,           c * 2, dropout_rate)
        self.enc3 = _double_conv(c * 2,       c * 4, dropout_rate)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = _double_conv(c * 4, c * 8, dropout_rate)

        # Decoder
        self.up3   = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3  = _double_conv(c * 8, c * 4, dropout_rate)

        self.up2   = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2  = _double_conv(c * 4, c * 2, dropout_rate)

        self.up1   = nn.ConvTranspose2d(c * 2, c,     kernel_size=2, stride=2)
        self.dec1  = _double_conv(c * 2, c,     dropout_rate)

        self.out_conv = nn.Conv2d(c, in_channels, kernel_size=1)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        # Encoder path
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))

        # Bottleneck
        b = self.bottleneck(self.pool(s3))

        # Decoder path with skip connections
        x = self.dec3(torch.cat([self.up3(b),  s3], dim=1))
        x = self.dec2(torch.cat([self.up2(x),  s2], dim=1))
        x = self.dec1(torch.cat([self.up1(x),  s1], dim=1))

        return self.sigmoid(self.out_conv(x))


def normalize_data(data, data_min=0.0, data_max=255.0):
    return (data - data_min) / (data_max - data_min)


def mean_squared_error(predictions, targets):
    return torch.mean((predictions - targets) ** 2)


def train_model(train_dataset, val_dataset,
                in_channels=1, base_channels=32,
                dropout_rate=0.2, lr=1e-3, batch_size=32, num_epochs=300):

    # --------------------
    # 1) Set your parameters
    # --------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training will run on: {device}")

    # --------------------
    # 2) Create DataLoaders
    # --------------------
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    # --------------------
    # 3) Define model, loss, optimizer
    # --------------------
    model = CNN(in_channels, base_channels, dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    train_losses, val_losses = [], []

    # --------------------
    # 4) Training loop over epochs
    # --------------------
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for noisy_batch, clean_batch in train_loader:
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)

            optimizer.zero_grad()
            predictions = model(noisy_batch)
            loss = mean_squared_error(predictions, clean_batch)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
        epoch_train_loss /= len(train_loader)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for noisy_batch, clean_batch in val_loader:
                noisy_batch = noisy_batch.to(device)
                clean_batch = clean_batch.to(device)
                predictions = model(noisy_batch)
                epoch_val_loss += mean_squared_error(predictions, clean_batch).item()
        epoch_val_loss /= len(val_loader)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]  "
                  f"Train Loss: {epoch_train_loss:.6f}  |  Val Loss: {epoch_val_loss:.6f}")

    return model, train_losses, val_losses


def plot_noisy_clean_predicted(model, dataset, index, title, device):
    model.eval()
    noisy_img, clean_img = dataset[index]

    with torch.no_grad():
        predicted = model(noisy_img.unsqueeze(0).to(device)).squeeze(0).cpu()

    # TODO: reshape tensors as needed for imshow (C, H, W) -> (H, W)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(noisy_img.squeeze().numpy(),     cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Noisy Input')
    axes[1].imshow(clean_img.squeeze().numpy(),     cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Clean Ground Truth')
    axes[2].imshow(predicted.squeeze().numpy(),     cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Predicted (Denoised)')
    for ax in axes:
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    # Save the composed figure to the script directory with a safe filename
    safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title).strip()
    save_dir = os.path.dirname(os.path.abspath(__file__))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"{safe_title}_idx{index}_{stamp}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved figure to: {save_path}")
    # plt.show() removed to avoid blocking execution in non-interactive environments
    


if __name__ == "__main__":

    # (A) Load data (shape: (N, H, W))
    _dir = "/cluster/scratch/azecchin/spacedata/images"
    noisy_data = np.load(os.path.join(_dir, "noisy_train_19k_harder.npy")).astype(np.float32)
    clean_data = np.load(os.path.join(_dir, "clean_train_19k_harder.npy")).astype(np.float32)

    # Normalise to [0, 1]
    noisy_data = normalize_data(noisy_data)
    clean_data = normalize_data(clean_data)

    # Add channel dimension: (N, H, W) -> (N, 1, H, W)
    noisy_tensor = torch.tensor(noisy_data).unsqueeze(1)
    clean_tensor = torch.tensor(clean_data).unsqueeze(1)
    dataset = TensorDataset(noisy_tensor, clean_tensor)

    # 5-fold cross-validation
    K = 5
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    indices = np.arange(len(dataset))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_train_losses = []
    all_val_losses   = []
    best_val_loss    = float('inf')
    best_model       = None
    best_train_subset = None
    best_val_subset   = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n--- Fold {fold + 1}/{K} ---")
        train_subset = Subset(dataset, train_idx)
        val_subset   = Subset(dataset, val_idx)

        model, train_losses, val_losses = train_model(
            train_subset, val_subset,
            in_channels=1,
            base_channels=32,
            dropout_rate=0.2,
            lr=1e-3,
            batch_size=256,
            num_epochs=50,
        )
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        if val_losses[-1] < best_val_loss:
            best_val_loss     = val_losses[-1]
            best_model        = model
            best_train_subset = train_subset
            best_val_subset   = val_subset

    # Plot per-fold curves (faint) + average (bold)
    avg_train = np.mean(all_train_losses, axis=0)
    avg_val   = np.mean(all_val_losses,   axis=0)

    plt.figure(figsize=(8, 4))
    for tl, vl in zip(all_train_losses, all_val_losses):
        plt.plot(tl, alpha=0.25, color='steelblue')
        plt.plot(vl, alpha=0.25, color='darkorange')
    plt.plot(avg_train, label='Avg Train Loss', color='steelblue',  linewidth=2)
    plt.plot(avg_val,   label='Avg Val Loss',   color='darkorange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'{K}-Fold Cross-Validation Loss')
    plt.legend()
    plt.tight_layout()
    # Save CV loss plot next to script
    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, f"{K}-fold_CV_loss.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved CV loss plot to: {save_path}")
    # plt.show() removed to avoid blocking execution in non-interactive environments

    # Save best model weights
    save_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(save_dir, "unet_best.pth")
    torch.save(best_model.state_dict(), model_path)
    print(f"Saved best model weights to: {model_path}")

    # Plot a sample from the best fold
    best_model.eval()
    plot_noisy_clean_predicted(
        best_model, best_train_subset, index=0,
        title="Best Fold — Training Sample: Noisy vs Clean vs Predicted",
        device=device,
    )
    plot_noisy_clean_predicted(
        best_model, best_val_subset, index=0,
        title="Best Fold — Validation Sample: Noisy vs Clean vs Predicted",
        device=device,
    )

