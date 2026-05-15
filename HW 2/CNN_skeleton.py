import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.model_selection import KFold


# Define the CNN model
class CNN(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, dropout_rate=0.2):
        super(CNN, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # TODO: add convolutional blocks (Conv2d, BatchNorm2d, ReLU, MaxPool2d)
        )

        # Decoder
        self.decoder = nn.Sequential(
            # TODO: add transpose convolutional blocks (ConvTranspose2d, BatchNorm2d, ReLU)
        )

        # Output layer
        self.output_layer = nn.Sequential(
            # TODO: final Conv2d + activation to reconstruct the image
        )

    def forward(self, x):
        # TODO: pass through encoder, decoder, output layer
        raise NotImplementedError


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
    plt.show()


def visualize_noisy_clean_slider(noisy_imgs, clean_imgs):
    """Interactive figure: image-index slider + blend slider (0=clean, 1=noisy)."""
    N = len(noisy_imgs)
    noisy_disp = noisy_imgs / 255.0 if noisy_imgs.max() > 1.0 else noisy_imgs.copy()
    clean_disp = clean_imgs / 255.0 if clean_imgs.max() > 1.0 else clean_imgs.copy()

    fig, ax = plt.subplots(figsize=(6, 7))
    plt.subplots_adjust(bottom=0.22)

    def blended(idx, alpha):
        return (1 - alpha) * clean_disp[idx] + alpha * noisy_disp[idx]

    im = ax.imshow(blended(0, 0.5), cmap='gray', vmin=0, vmax=1)
    ax.set_title("Image 0  |  0 = clean  ←  blend  →  1 = noisy")
    ax.axis('off')

    ax_idx   = plt.axes([0.15, 0.10, 0.70, 0.03])
    ax_blend = plt.axes([0.15, 0.04, 0.70, 0.03])
    s_idx    = Slider(ax_idx,   'Image #', 0, N - 1, valinit=0,   valstep=1)
    s_blend  = Slider(ax_blend, 'Blend',   0.0, 1.0, valinit=0.5)

    def update(_):
        idx   = int(s_idx.val)
        alpha = s_blend.val
        im.set_data(blended(idx, alpha))
        ax.set_title(f"Image {idx}  |  0 = clean  ←  blend  →  1 = noisy")
        fig.canvas.draw_idle()

    s_idx.on_changed(update)
    s_blend.on_changed(update)
    plt.show()


if __name__ == "__main__":

    # (A) Load data (shape: (N, H, W))
    _dir = os.path.dirname(os.path.abspath(__file__))
    noisy_data = np.load(os.path.join(_dir, "noisy_images_small_1k (1).npy")).astype(np.float32)
    clean_data = np.load(os.path.join(_dir, "clean_images_small_1k (1).npy")).astype(np.float32)

    # Visualise raw images with interactive sliders before training
    visualize_noisy_clean_slider(noisy_data, clean_data)

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
            batch_size=32,
            num_epochs=300,
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
    plt.show()

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
