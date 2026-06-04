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


_l1_criterion = nn.L1Loss()
_l2_criterion = nn.MSELoss()


def compute_mean_psnr(denoised_tensor, reference_tensor, max_val=1.0):
    psnr_vals = []
    for pred, ref in zip(denoised_tensor, reference_tensor):
        mse = torch.mean((pred - ref) ** 2).item()
        if mse == 0:
            psnr_vals.append(float('inf'))
        else:
            psnr_vals.append(20.0 * np.log10(max_val / np.sqrt(mse)))
    return float(np.mean(psnr_vals))


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

    train_l1_losses, train_l2_losses = [], []
    val_l1_losses,   val_l2_losses   = [], []

    best_val_l1   = float('inf')
    best_val_l2   = float('inf')
    best_l1_state = None
    best_l2_state = None

    for epoch in range(num_epochs):
        # ── train ──────────────────────────────────────────────────────────────
        model.train()
        epoch_train_l1 = 0.0
        epoch_train_l2 = 0.0
        for noisy_batch, clean_batch in train_loader:
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)
            optimizer.zero_grad()
            output = model(noisy_batch)
            loss = _l1_criterion(output, clean_batch)
            loss.backward()
            optimizer.step()
            epoch_train_l1 += loss.item()
            with torch.no_grad():
                epoch_train_l2 += _l2_criterion(output, clean_batch).item()
        epoch_train_l1 /= len(train_loader)
        epoch_train_l2 /= len(train_loader)

        # ── validate ───────────────────────────────────────────────────────────
        model.eval()
        epoch_val_l1 = 0.0
        epoch_val_l2 = 0.0
        with torch.no_grad():
            for noisy_batch, clean_batch in val_loader:
                noisy_batch = noisy_batch.to(device)
                clean_batch = clean_batch.to(device)
                output = model(noisy_batch)
                epoch_val_l1 += _l1_criterion(output, clean_batch).item()
                epoch_val_l2 += _l2_criterion(output, clean_batch).item()
        epoch_val_l1 /= len(val_loader)
        epoch_val_l2 /= len(val_loader)

        train_l1_losses.append(epoch_train_l1)
        train_l2_losses.append(epoch_train_l2)
        val_l1_losses.append(epoch_val_l1)
        val_l2_losses.append(epoch_val_l2)

        # checkpoint best-epoch states (no extra memory allocation per epoch,
        # just one snapshot per improvement)
        if epoch_val_l1 < best_val_l1:
            best_val_l1   = epoch_val_l1
            best_l1_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch_val_l2 < best_val_l2:
            best_val_l2   = epoch_val_l2
            best_l2_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step(epoch_val_l1)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:>3}/{num_epochs}]  "
                  f"Train L1: {epoch_train_l1:.6f}  Train L2: {epoch_train_l2:.6f}  |  "
                  f"Val L1: {epoch_val_l1:.6f}  Val L2: {epoch_val_l2:.6f}")

    return (model,
            best_l1_state, best_val_l1,
            best_l2_state, best_val_l2,
            train_l1_losses, train_l2_losses,
            val_l1_losses,   val_l2_losses)


# ── Plotting ──────────────────────────────────────────────────────────────────
def _safe(title):
    return "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title).strip()


def plot_cv_losses(all_train, all_val, loss_name, k, save_dir, is_training_loss=True):
    avg_train = np.mean(all_train, axis=0)
    avg_val   = np.mean(all_val,   axis=0)

    fig, ax = plt.subplots(figsize=(9, 4))
    for tl, vl in zip(all_train, all_val):
        ax.plot(tl, alpha=0.20, color='steelblue')
        ax.plot(vl, alpha=0.20, color='darkorange')

    ax.plot(avg_train, label=f'Avg Train {loss_name}', color='steelblue',  linewidth=2)
    ax.plot(avg_val,   label=f'Avg Val {loss_name}',   color='darkorange', linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(loss_name)
    ax.legend(loc='upper right')

    note = "(backprop)" if is_training_loss else "(monitor only — backprop uses L1)"
    plt.title(f'{k}-Fold Cross-Validation — CNN — {loss_name} loss {note}')
    plt.tight_layout()
    path = os.path.join(save_dir, f"5fold_CV_{loss_name}_loss.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved CV plot to: {path}")


def plot_train_sample(model, dataset, index, title, device, save_dir):
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


def denoise_dataset(model, noisy_tensor, batch_size, device):
    model.eval()
    pin = device.type == 'cuda'
    loader = DataLoader(TensorDataset(noisy_tensor), batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=pin)
    parts = []
    with torch.no_grad():
        for (batch,) in loader:
            parts.append(model(batch.to(device)).cpu())
    return torch.cat(parts, dim=0)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    _dir     = os.path.dirname(os.path.abspath(__file__))
    save_dir = _dir

    # ── UPDATE these filenames to match the Task 3 data on the cluster ────────
    TRAIN_NOISY = "noisy_train_19k_harder.npy"
    TRAIN_CLEAN = "clean_train_19k_harder.npy"
    TEST_NOISY  = "noisy_val_500_harder.npy"
    TEST_CLEAN  = None  # set to filename if a clean test reference is available
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

    all_train_l1, all_train_l2 = [], []
    all_val_l1,   all_val_l2   = [], []

    global_best_val_l1   = float('inf')
    global_best_val_l2   = float('inf')
    global_best_l1_state = None
    global_best_l2_state = None
    best_train_subset    = None
    best_val_subset      = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n=== Fold {fold + 1}/{K} ===")
        train_subset = Subset(dataset, train_idx)
        val_subset   = Subset(dataset, val_idx)

        (model,
         best_l1_state, best_val_l1,
         best_l2_state, best_val_l2,
         train_l1, train_l2,
         val_l1,   val_l2) = train_model(
            train_subset, val_subset,
            in_channels=1,
            base_channels=BASE_CHANNELS,
            dropout_rate=DROPOUT,
            lr=LR,
            batch_size=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
        )

        all_train_l1.append(train_l1)
        all_train_l2.append(train_l2)
        all_val_l1.append(val_l1)
        all_val_l2.append(val_l2)

        if best_val_l1 < global_best_val_l1:
            global_best_val_l1   = best_val_l1
            global_best_l1_state = best_l1_state
            best_train_subset    = train_subset
            best_val_subset      = val_subset

        if best_val_l2 < global_best_val_l2:
            global_best_val_l2   = best_val_l2
            global_best_l2_state = best_l2_state

    # ── CV loss plots ─────────────────────────────────────────────────────────
    plot_cv_losses(all_train_l1, all_val_l1, 'L1', K, save_dir, is_training_loss=True)
    plot_cv_losses(all_train_l2, all_val_l2, 'L2', K, save_dir, is_training_loss=False)

    # ── Save loss arrays ──────────────────────────────────────────────────────
    np.save(os.path.join(save_dir, "train_l1_losses.npy"), np.array(all_train_l1))
    np.save(os.path.join(save_dir, "train_l2_losses.npy"), np.array(all_train_l2))
    np.save(os.path.join(save_dir, "val_l1_losses.npy"),   np.array(all_val_l1))
    np.save(os.path.join(save_dir, "val_l2_losses.npy"),   np.array(all_val_l2))
    print("Saved loss arrays: train_l1_losses.npy, train_l2_losses.npy, "
          "val_l1_losses.npy, val_l2_losses.npy")

    # ── Reconstruct best models from saved state dicts ────────────────────────
    model_l1 = CNN(1, BASE_CHANNELS, DROPOUT).to(device)
    model_l1.load_state_dict({k: v.to(device) for k, v in global_best_l1_state.items()})
    model_l1.eval()

    model_l2 = CNN(1, BASE_CHANNELS, DROPOUT).to(device)
    model_l2.load_state_dict({k: v.to(device) for k, v in global_best_l2_state.items()})
    model_l2.eval()

    torch.save(global_best_l1_state, os.path.join(save_dir, "model_best_L1.pth"))
    torch.save(global_best_l2_state, os.path.join(save_dir, "model_best_L2.pth"))
    print(f"Saved model_best_L1.pth  (best val L1: {global_best_val_l1:.6f})")
    print(f"Saved model_best_L2.pth  (best val L2: {global_best_val_l2:.6f})")

    # ── Training/validation image visualizations (best L1 model) ─────────────
    for idx in [0, 1]:
        plot_train_sample(model_l1, best_train_subset, index=idx,
                          title=f"Train Sample {idx} — Best L1 Model",
                          device=device, save_dir=save_dir)
    for idx in [0, 1]:
        plot_train_sample(model_l1, best_val_subset, index=idx,
                          title=f"Val Sample {idx} — Best L1 Model",
                          device=device, save_dir=save_dir)

    # ── Test set: denoise with both models, compute PSNR, save predictions ────
    test_noisy_raw    = np.load(os.path.join(_dir, TEST_NOISY)).astype(np.float32)
    test_noisy_norm   = normalize_data(test_noisy_raw)
    test_noisy_tensor = torch.tensor(test_noisy_norm).unsqueeze(1)

    if TEST_CLEAN is not None:
        test_clean_norm   = normalize_data(np.load(os.path.join(_dir, TEST_CLEAN)).astype(np.float32))
        test_reference    = torch.tensor(test_clean_norm).unsqueeze(1)
        psnr_ref_label    = "clean GT"
    else:
        test_reference = test_noisy_tensor
        psnr_ref_label = "noisy input (no clean GT provided — set TEST_CLEAN to enable proper PSNR)"

    print(f"\nPSNR reference: {psnr_ref_label}")
    for model, label, pred_fname in [
        (model_l1, "L1", "ferrini_prediction_L1.npz"),
        (model_l2, "L2", "ferrini_prediction_L2.npz"),
    ]:
        denoised = denoise_dataset(model, test_noisy_tensor, BATCH_SIZE, device)
        mean_psnr = compute_mean_psnr(denoised, test_reference)
        print(f"Model {label} — Mean PSNR: {mean_psnr:.4f} dB")

        denoised_np = (denoised.squeeze(1).numpy() * 255.0).clip(0, 255).astype(np.float32)
        np.savez(os.path.join(save_dir, pred_fname), denoised_images=denoised_np)
        print(f"Saved test predictions to: {pred_fname}")

        for idx in [0, 1]:
            plot_test_sample(model, test_noisy_tensor[idx], index=idx,
                             title=f"Test Sample {idx} — Best {label} Model",
                             device=device, save_dir=save_dir)

    print("\nDone.")
