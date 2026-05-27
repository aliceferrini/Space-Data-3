import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset, random_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import optuna


# ==========================================
# 1. Architecture: ResNet
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return torch.relu(out)


class ResNetDenoiser(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, num_blocks=4, dropout_rate=0.2):
        super(ResNetDenoiser, self).__init__()
        self.start_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_blocks)]
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Conv2d(base_channels, in_channels, kernel_size=1),
            nn.Sigmoid() # WINNING CONFIG: Sigmoid maps perfectly to [0, 1]
        )

    def forward(self, x):
        x = self.start_conv(x)
        x = self.res_blocks(x)
        return self.final_conv(x)


# ==========================================
# 2. Pipeline Helpers
# ==========================================
def normalize_data(data, data_min=0.0, data_max=255.0):
    # WINNING CONFIG: Linear scaling to [0, 1] bounds
    return (data - data_min) / (data_max - data_min)


def mean_squared_error(predictions, targets):
    # WINNING CONFIG: Squares errors to heavily punish and flatten grid lines
    return torch.mean((predictions - targets) ** 2)


def train_model(train_dataset, val_dataset,
                in_channels=1, base_channels=64,
                dropout_rate=0.2, lr=1e-3, batch_size=256, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # SPEED OPTIMIZED: persistent_workers=True keeps CPU workers alive across epochs
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True, persistent_workers=True)

    model = ResNetDenoiser(in_channels, base_channels, 4, dropout_rate)

    if torch.cuda.device_count() > 1:
        print(f"Accelerating training across {torch.cuda.device_count()} GPUs using DataParallel!")
        model = nn.DataParallel(model)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    scaler = torch.cuda.amp.GradScaler()

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for noisy_batch, clean_batch in train_loader:

            if torch.rand(1).item() > 0.5:
                noisy_batch = torch.flip(noisy_batch, dims=[2])
                clean_batch = torch.flip(clean_batch, dims=[2])
            if torch.rand(1).item() > 0.5:
                noisy_batch = torch.flip(noisy_batch, dims=[3])
                clean_batch = torch.flip(clean_batch, dims=[3])

            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                predictions = model(noisy_batch)
                loss = mean_squared_error(predictions, clean_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item()
        epoch_train_loss /= len(train_loader)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for noisy_batch, clean_batch in val_loader:
                noisy_batch = noisy_batch.to(device)
                clean_batch = clean_batch.to(device)
                with torch.cuda.amp.autocast():
                    predictions = model(noisy_batch)
                    loss = mean_squared_error(predictions, clean_batch)
                epoch_val_loss += loss.item()
        epoch_val_loss /= len(val_loader)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

    return model, train_losses, val_losses


def plot_noisy_clean_predicted(model, dataset, index, title, device):
    model.eval()
    noisy_img, clean_img = dataset[index]
    with torch.no_grad():
        predicted = model(noisy_img.unsqueeze(0).to(device)).squeeze(0).cpu()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(noisy_img.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Noisy Input')
    axes[1].imshow(clean_img.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Clean Ground Truth')
    axes[2].imshow(predicted.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Predicted (ResNet)')
    for ax in axes: ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()

    safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title).strip()
    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, f"Deployed_{safe_title}_idx{index}.png")
    plt.savefig(save_path, dpi=200)


# ==========================================
# 3. Main Execution Block
# ==========================================
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware allocated: {device}")

    _dir = os.path.dirname(os.path.abspath(__file__))
    noisy_data = np.load(os.path.join(_dir, "noisy_train_19k_harder.npy")).astype(np.float32)
    clean_data = np.load(os.path.join(_dir, "clean_train_19k_harder.npy")).astype(np.float32)

    noisy_tensor = torch.tensor(normalize_data(noisy_data)).unsqueeze(1)
    clean_tensor = torch.tensor(normalize_data(clean_data)).unsqueeze(1)
    dataset = TensorDataset(noisy_tensor, clean_tensor)

    # --- PHASE 1: FAST OPTUNA SWEEP ---
    print("\n--- Phase 1: Commencing Optuna Hyperparameter Sweep ---")

    sweep_size = int(0.1 * len(dataset))
    sweep_train_size = int(0.8 * sweep_size)
    sweep_val_size = sweep_size - sweep_train_size
    sweep_subset, _ = random_split(dataset, [sweep_size, len(dataset) - sweep_size])
    sweep_train, sweep_val = random_split(sweep_subset, [sweep_train_size, sweep_val_size])

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)

        _, _, val_losses = train_model(
            sweep_train, sweep_val,
            base_channels=64, dropout_rate=dropout_rate,
            lr=lr, batch_size=batch_size, num_epochs=15
        )
        return val_losses[-1]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    print("\n[Sweep Complete] Best Parameters Found:")
    best_params = study.best_trial.params
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # --- PHASE 2: 5-FOLD CV DEPLOYMENT TRAINING ---
    print("\n--- Phase 2: Commencing Full 5-Fold Training with Optimized Params ---")

    K = 5
    kf = KFold(n_splits=K, shuffle=True, random_state=42)

    all_train_losses, all_val_losses = [], []
    best_val_loss = float('inf')
    best_model, best_train_subset, best_val_subset = None, None, None

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)))):
        print(f"\n--- Fold {fold + 1}/{K} ---")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        model, train_losses, val_losses = train_model(
            train_subset, val_subset,
            base_channels=64,
            dropout_rate=best_params["dropout_rate"],
            lr=best_params["lr"],
            batch_size=best_params["batch_size"],
            num_epochs=100,
        )

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_model = model
            best_train_subset, best_val_subset = train_subset, val_subset

    # --- PHASE 3: METRICS & EXPORT ---
    print("\n--- Phase 3: Exporting Deployment Assets ---")

    plt.figure(figsize=(8, 4))
    for tl, vl in zip(all_train_losses, all_val_losses):
        plt.plot(tl, alpha=0.25, color='crimson')
        plt.plot(vl, alpha=0.25, color='darkgreen')
    plt.plot(np.mean(all_train_losses, axis=0), label='Avg Train Loss', color='crimson', linewidth=2)
    plt.plot(np.mean(all_val_losses, axis=0), label='Avg Val Loss', color='darkgreen', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Optimized ResNet {K}-Fold Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(_dir, "Deployed_5-fold_CV_loss.png"), dpi=200)

    best_model.eval()
    plot_noisy_clean_predicted(best_model, best_train_subset, 0, "Best Fold Training Sample", device)
    plot_noisy_clean_predicted(best_model, best_val_subset, 0, "Best Fold Validation Sample", device)

    weights_path = os.path.join(_dir, "ResNet_Best_Weights.pth")
    weights_to_save = best_model.module.state_dict() if isinstance(best_model, nn.DataParallel) else best_model.state_dict()
    torch.save(weights_to_save, weights_path)
    print(f"SUCCESS: Saved deployment-ready weights to: {weights_path}")
