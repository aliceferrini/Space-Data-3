import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for Euler cluster (no display)
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import KFold


# ── Model ─────────────────────────────────────────────────────────────────────
class CNN(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, dropout_rate=0.2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,        base_channels,     kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm2d(base_channels)
        self.drop1 = nn.Dropout2d(dropout_rate)
        self.conv2 = nn.Conv2d(base_channels,      base_channels * 2, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm2d(base_channels * 2)
        self.drop2 = nn.Dropout2d(dropout_rate)
        self.conv3 = nn.Conv2d(base_channels * 2,  base_channels * 2, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(base_channels * 2)
        self.conv4 = nn.Conv2d(base_channels * 2,  base_channels,     kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(base_channels)
        self.conv5 = nn.Conv2d(base_channels,      in_channels,       kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.drop1(torch.relu(self.bn1(self.conv1(x))))
        x = self.drop2(torch.relu(self.bn2(self.conv2(x))))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.sigmoid(self.conv5(x))
        return x


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_data(data, data_min=0.0, data_max=255.0):
    return (data - data_min) / (data_max - data_min)


criterion = nn.L1Loss()


def l1_loss(predictions, targets):
    return criterion(predictions, targets)


# ── Training ──────────────────────────────────────────────────────────────────
def train_model(train_dataset, val_dataset,
                in_channels=1, base_channels=64,
                dropout_rate=0.2, lr=1e-3, batch_size=64, num_epochs=100):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    pin = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=pin)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=pin)

    model     = CNN(in_channels, base_channels, dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for noisy_batch, clean_batch in train_loader:
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)
            optimizer.zero_grad()
            loss = l1_loss(model(noisy_batch), clean_batch)
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
                epoch_val_loss += l1_loss(model(noisy_batch), clean_batch).item()
        epoch_val_loss /= len(val_loader)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:>3}/{num_epochs}]  "
                  f"Train L1: {epoch_train_loss:.6f}  |  Val L1: {epoch_val_loss:.6f}")

    return model, train_losses, val_losses


# ── Plotting ──────────────────────────────────────────────────────────────────
def _safe(title):
    return "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title).strip()


def plot_train_sample(model, dataset, index, title, device, save_dir):
    """Show clean / noisy / denoised for a labelled training sample."""
    model.eval()
    noisy_img, clean_img = dataset[index]
    with torch.no_grad():
        predicted = model(noisy_img.unsqueeze(0).to(device)).squeeze(0).cpu()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(noisy_img.squeeze().numpy(),  cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Noisy Input')
    axes[1].imshow(clean_img.squeeze().numpy(),  cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Clean Ground Truth')
    axes[2].imshow(predicted.squeeze().numpy(),  cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Predicted (Denoised)')
    for ax in axes:
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{_safe(title)}_idx{index}.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def plot_test_sample(model, noisy_img_tensor, index, title, device, save_dir):
    """Show noisy / denoised for a test sample (no clean GT available)."""
    model.eval()
    with torch.no_grad():
        predicted = model(noisy_img_tensor.unsqueeze(0).to(device)).squeeze(0).cpu()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(noisy_img_tensor.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Noisy Input')
    axes[1].imshow(predicted.squeeze().numpy(),        cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Predicted (Denoised)')
    for ax in axes:
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{_safe(title)}_idx{index}.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    _dir     = os.path.dirname(os.path.abspath(__file__))
    save_dir = _dir

    # ── UPDATE these filenames to match the Task 3 data on the cluster ────────
    TRAIN_NOISY = "noisy_train_19k_harder.npy"
    TRAIN_CLEAN = "clean_train_19k_harder.npy"
    TEST_NOISY  = "noisy_val_500_harder.npy"  
    # ─────────────────────────────────────────────────────────────────────────

    noisy_data = np.load(os.path.join(_dir, TRAIN_NOISY)).astype(np.float32)
    clean_data = np.load(os.path.join(_dir, TRAIN_CLEAN)).astype(np.float32)

    noisy_data = normalize_data(noisy_data)
    clean_data = normalize_data(clean_data)

    noisy_tensor = torch.tensor(noisy_data).unsqueeze(1)
    clean_tensor = torch.tensor(clean_data).unsqueeze(1)
    dataset = TensorDataset(noisy_tensor, clean_tensor)

    # ── Hyperparameters — Configuration: base CNN + L1 loss ──────────────────
    BASE_CHANNELS = 64
    DROPOUT       = 0.2
    LR            = 1e-3
    BATCH_SIZE    = 64
    NUM_EPOCHS    = 100

    # ── 5-fold cross-validation ───────────────────────────────────────────────
    K       = 5
    kf      = KFold(n_splits=K, shuffle=True, random_state=42)
    indices = np.arange(len(dataset))
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_train_losses  = []
    all_val_losses    = []
    best_val_loss     = float('inf')
    best_model        = None
    best_train_subset = None
    best_val_subset   = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n=== Fold {fold + 1}/{K} ===")
        train_subset = Subset(dataset, train_idx)
        val_subset   = Subset(dataset, val_idx)

        model, train_losses, val_losses = train_model(
            train_subset, val_subset,
            in_channels=1,
            base_channels=BASE_CHANNELS,
            dropout_rate=DROPOUT,
            lr=LR,
            batch_size=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
        )
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        if val_losses[-1] < best_val_loss:
            best_val_loss     = val_losses[-1]
            best_model        = model
            best_train_subset = train_subset
            best_val_subset   = val_subset

    # ── CV loss plot ──────────────────────────────────────────────────────────
    avg_train = np.mean(all_train_losses, axis=0)
    avg_val   = np.mean(all_val_losses,   axis=0)

    plt.figure(figsize=(8, 4))
    for tl, vl in zip(all_train_losses, all_val_losses):
        plt.plot(tl, alpha=0.25, color='steelblue')
        plt.plot(vl, alpha=0.25, color='darkorange')
    plt.plot(avg_train, label='Avg Train L1', color='steelblue',  linewidth=2)
    plt.plot(avg_val,   label='Avg Val L1',   color='darkorange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss')
    plt.title(f'{K}-Fold Cross-Validation — Base CNN + L1 Loss')
    plt.legend()
    plt.tight_layout()
    cv_path = os.path.join(save_dir, "5fold_CV_L1_loss.png")
    plt.savefig(cv_path, dpi=200)
    plt.close()
    print(f"Saved CV plot to: {cv_path}")

    # ── Save best model ───────────────────────────────────────────────────────
    model_path = os.path.join(save_dir, "model_best.pth")
    torch.save(best_model.state_dict(), model_path)
    print(f"Saved best model to: {model_path}")

    # ── Training set image visualizations (2 images) ─────────────────────────
    best_model.eval()
    for idx in [0, 1]:
        plot_train_sample(
            best_model, best_train_subset, index=idx,
            title=f"Train Sample {idx} — Base CNN L1",
            device=device, save_dir=save_dir,
        )

    # ── Validation set image visualizations (2 images) ───────────────────────
    for idx in [0, 1]:
        plot_train_sample(
            best_model, best_val_subset, index=idx,
            title=f"Val Sample {idx} — Base CNN L1",
            device=device, save_dir=save_dir,
        )

    # ── Test set: denoise all images and save as .npz ─────────────────────────
    test_noisy_raw    = np.load(os.path.join(_dir, TEST_NOISY)).astype(np.float32)
    test_noisy_norm   = normalize_data(test_noisy_raw)
    test_noisy_tensor = torch.tensor(test_noisy_norm).unsqueeze(1)

    best_model.eval()
    denoised_list = []
    pin = device.type == 'cuda'
    test_loader = DataLoader(
        TensorDataset(test_noisy_tensor),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=pin,
    )
    with torch.no_grad():
        for (batch,) in test_loader:
            out = best_model(batch.to(device)).cpu()
            denoised_list.append(out)

    denoised_np = torch.cat(denoised_list, dim=0).squeeze(1).numpy()  # (N, H, W)
    denoised_np = (denoised_np * 255.0).clip(0, 255).astype(np.float32)

    pred_path = os.path.join(save_dir, "ferrini_prediction.npz")
    np.savez(pred_path, denoised_images=denoised_np)
    print(f"Saved test predictions to: {pred_path}")

    # ── Test set image visualizations (2 images, no clean GT) ────────────────
    for idx in [0, 1]:
        plot_test_sample(
            best_model, test_noisy_tensor[idx],
            index=idx,
            title=f"Test Sample {idx} — Base CNN L1",
            device=device, save_dir=save_dir,
        )

    print("\nDone.")
