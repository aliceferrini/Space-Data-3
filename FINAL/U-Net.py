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


class Model(nn.Module):
    """
    U-Net for denoising 64x64 lunar surface images.
    Spatial flow: 64 -> 32 -> 16 -> 8 (bottleneck) -> 16 -> 32 -> 64
    """
    def __init__(self, in_channels=1, base_channels=32, dropout_rate=0.2):
        super(Model, self).__init__()
        c = base_channels  # 32

        # Encoder: 64x64 -> 32x32 -> 16x16 -> 8x8
        self.enc1 = _double_conv(in_channels, c,     dropout_rate)  # 64x64, 32ch
        self.enc2 = _double_conv(c,           c * 2, dropout_rate)  # 32x32, 64ch
        self.enc3 = _double_conv(c * 2,       c * 4, dropout_rate)  # 16x16, 128ch
        self.pool = nn.MaxPool2d(2)

        # Bottleneck at 8x8 — wide receptive field captures grid periodicity
        self.bottleneck = nn.Sequential(
            _double_conv(c * 4, c * 8, dropout_rate),               # 8x8, 256ch
            nn.Conv2d(c * 8, c * 8, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(inplace=True),
        )

        # Decoder: 8x8 -> 16x16 -> 32x32 -> 64x64
        self.up3  = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = _double_conv(c * 8, c * 4, dropout_rate)        # 16x16, 128ch

        self.up2  = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = _double_conv(c * 4, c * 2, dropout_rate)        # 32x32, 64ch

        self.up1  = nn.ConvTranspose2d(c * 2, c,     kernel_size=2, stride=2)
        self.dec1 = _double_conv(c * 2, c,     dropout_rate)        # 64x64, 32ch

        self.out_conv = nn.Conv2d(c, in_channels, kernel_size=1)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))

        b = self.bottleneck(self.pool(s3))

        x = self.dec3(torch.cat([self.up3(b), s3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), s2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1))

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
    if not torch.cuda.is_available():
        print("CUDA not available. Training will run on CPU, which may be slow.")
    print(f"Training will run on: {device}")

    # --------------------
    # 2) Create DataLoaders
    # --------------------
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)


    # --------------------
    # 3) Define model, loss, optimizer
    # --------------------
    model = Model(in_channels, base_channels, dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30
    )

    train_losses, val_losses = [], []

    # --------------------
    # 4) Training loop over epochs
    # --------------------
    best_val_loss_epoch = float('inf')
    best_state_dict = None
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for noisy_batch, clean_batch in train_loader:
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)

            # Augmentation
            if torch.rand(1).item() > 0.5:
                noisy_batch = torch.flip(noisy_batch, dims=[3])  # Horizontal flip
                clean_batch = torch.flip(clean_batch, dims=[3])
            if torch.rand(1).item() > 0.5:
                noisy_batch = torch.flip(noisy_batch, dims=[2])  # Vertical flip
                clean_batch = torch.flip(clean_batch, dims=[2])
            k = torch.randint(0, 4, (1,)).item()
            noisy_batch = torch.rot90(noisy_batch, k, dims=[2, 3])  # Rotate by 0, 90, 180, or 270 degrees
            clean_batch = torch.rot90(clean_batch, k, dims=[2, 3])

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
        if epoch_val_loss < best_val_loss_epoch:
            best_val_loss_epoch = epoch_val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        scheduler.step(epoch_val_loss)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]  "
                  f"Train Loss: {epoch_train_loss:.6f}  |  Val Loss: {epoch_val_loss:.6f}")

    model.load_state_dict(best_state_dict)
    return model, train_losses, val_losses, best_val_loss_epoch


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
    # Save the composed figure to the output directory with a safe filename
    safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title).strip()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"{safe_title}_idx{index}_{stamp}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved figure to: {save_path}")
    # plt.show() removed to avoid blocking execution in non-interactive environments
    


if __name__ == "__main__":

    # Create output folder if it doesn't exist
    save_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(save_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # (A) Load data (shape: (N, H, W))
    _dir = "/cluster/scratch/azecchin/spacedata/images"
    noisy_data = np.load(os.path.join(_dir, "noisy_train_19k_harder.npy")).astype(np.float32)
    clean_data = np.load(os.path.join(_dir, "clean_train_19k_harder.npy")).astype(np.float32)
    print(f"Loaded noisy data shape: {noisy_data.shape}, dtype: {noisy_data.dtype}")

    # Normalise to [0, 1]
    noisy_data = normalize_data(noisy_data)
    clean_data = normalize_data(clean_data)
    print(f"Data normalized to [0, 1]")

    # Add channel dimension: (N, H, W) -> (N, 1, H, W)
    noisy_tensor = torch.tensor(noisy_data).unsqueeze(1)
    clean_tensor = torch.tensor(clean_data).unsqueeze(1)
    dataset = TensorDataset(noisy_tensor, clean_tensor)

    # 5-fold cross-validation
    K = 5
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    indices = np.arange(len(dataset))
    print(f"Starting {K}-fold cross-validation with {len(dataset)} samples...")

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

        model, train_losses, val_losses, fold_best_val = train_model(
            train_subset, val_subset,
            in_channels=1,
            base_channels=32,
            dropout_rate=0,
            lr=1e-3,
            batch_size=256,
            num_epochs=50,
        )
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        if fold_best_val < best_val_loss:
            best_val_loss     = fold_best_val
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
    # Save CV loss plot in the output directory with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"{K}-fold_CV_loss_{timestamp}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved CV loss plot to: {save_path}")
    # plt.show() removed to avoid blocking execution in non-interactive environments

    # Save best model weights
    save_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(output_dir, f"unet_best_{timestamp}.pth")
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

