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
    def __init__(self, in_channels=1, base_channels=64, dropout_rate=0.2):
        super(CNN, self).__init__()
        # Large 7x7 kernel: each neuron sees a 7x7 neighbourhood from the start
        self.conv1 = nn.Conv2d(in_channels,        base_channels,     kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm2d(base_channels)
        self.drop1 = nn.Dropout2d(dropout_rate)
        # 5x5 kernel: widen receptive field further
        self.conv2 = nn.Conv2d(base_channels,      base_channels * 2, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm2d(base_channels * 2)
        self.drop2 = nn.Dropout2d(dropout_rate)
        # 3x3 refinement layers
        self.conv3 = nn.Conv2d(base_channels * 2,  base_channels * 2, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(base_channels * 2)
        self.conv4 = nn.Conv2d(base_channels * 2,  base_channels,     kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(base_channels)
        # 1x1 output projection back to single channel
        self.conv5 = nn.Conv2d(base_channels,      in_channels,       kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.drop1(torch.relu(self.bn1(self.conv1(x))))
        x = self.drop2(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.sigmoid(self.conv5(x))
        return x


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
    plt.show()


if __name__ == "__main__":

    # (A) Load data (shape: (N, H, W))
    _dir = os.path.dirname(os.path.abspath(__file__))
    noisy_data = np.load(os.path.join(_dir, "noisy_images_small_1k (1).npy")).astype(np.float32)
    clean_data = np.load(os.path.join(_dir, "clean_images_small_1k (1).npy")).astype(np.float32)

    # Normalise to [0, 1]
    noisy_data = normalize_data(noisy_data)
    clean_data = normalize_data(clean_data)

    # Add channel dimension: (N, H, W) -> (N, 1, H, W)
    noisy_tensor = torch.tensor(noisy_data).unsqueeze(1)
    clean_tensor = torch.tensor(clean_data).unsqueeze(1)
    dataset = TensorDataset(noisy_tensor, clean_tensor)

    # 5-fold cross-validation
    K = 3
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
            num_epochs=100,
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

