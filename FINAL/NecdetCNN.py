import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset, random_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import optuna


# ==========================================
# 1. Architecture: Heavyweight ResNet (128 Channels, 8 Blocks)
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
    def __init__(self, in_channels=1, base_channels=128, num_blocks=8, dropout_rate=0.2):
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
            nn.Sigmoid()  # STRICTLY SIGMOID FOR PERFECT CRATER TOPOGRAPHY
        )

    def forward(self, x):
        x = self.start_conv(x)
        x = self.res_blocks(x)
        return self.final_conv(x)


# ==========================================
# 2. Pipeline Helpers
# ==========================================
def normalize_data(data, data_min=0.0, data_max=255.0):
    return (data - data_min) / (data_max - data_min)


def mean_squared_error(predictions, targets):
    return torch.mean((predictions - targets) ** 2)


def train_model(train_dataset, val_dataset,
                in_channels=1, base_channels=128, num_blocks=8,
                dropout_rate=0.2, lr=1e-3, batch_size=64, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=10, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=10, pin_memory=True, persistent_workers=True, drop_last=True)

    model = ResNetDenoiser(in_channels, base_channels, num_blocks, dropout_rate)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler()

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_weights = None

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

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_weights = model.state_dict().copy()

    model.load_state_dict(best_model_weights)
    return model, train_losses, val_losses


# ==========================================
# 3. Rubric Visualization Helpers
# ==========================================
def plot_rubric_train_samples(model, dataset, raw_noisy_tensor, indices, device):
    model.eval()
    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4 * len(indices)))

    for i, idx in enumerate(indices):
        filtered_img, clean_img = dataset[idx]
        raw_noisy_img = raw_noisy_tensor[idx]
        with torch.no_grad():
            predicted = model(filtered_img.unsqueeze(0).to(device)).squeeze(0).cpu()

        axes[i, 0].imshow(raw_noisy_img.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Train Noisy {idx}')
        axes[i, 1].imshow(clean_img.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Train Clean Truth {idx}')
        axes[i, 2].imshow(predicted.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Train Denoised {idx}')
        for ax in axes[i]: ax.axis('off')

    plt.tight_layout()
    plt.savefig("Rubric_Train_Visualizations.png", dpi=200)


def plot_rubric_test_samples(model, filtered_tensor, raw_noisy_tensor, indices, device):
    model.eval()
    fig, axes = plt.subplots(len(indices), 2, figsize=(8, 4 * len(indices)))

    for i, idx in enumerate(indices):
        filtered_img = filtered_tensor[idx]
        raw_noisy_img = raw_noisy_tensor[idx]
        with torch.no_grad():
            predicted = model(filtered_img.unsqueeze(0).to(device)).squeeze(0).cpu()

        axes[i, 0].imshow(raw_noisy_img.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Test Noisy {idx}')
        axes[i, 1].imshow(predicted.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Test Denoised {idx}')
        for ax in axes[i]: ax.axis('off')

    plt.tight_layout()
    plt.savefig("Rubric_Test_Visualizations.png", dpi=200)


# ==========================================
# 4. Main Execution Block
# ==========================================
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware allocated: {device}")

    _dir = os.path.dirname(os.path.abspath(__file__))

    # --- DATA LOADING ---
    print("\nLoading Training Data (19k)...")
    noisy_train_data = np.load(os.path.join(_dir, "noisy_train_19k_harder.npy")).astype(np.float32)
    clean_train_data = np.load(os.path.join(_dir, "clean_train_19k_harder.npy")).astype(np.float32)

    print("\nLoading Blind Test Set (500)...")
    blind_test_data = np.load(os.path.join(_dir, "noisy_val_500_harder.npy")).astype(np.float32)

    # --- NOISE FILTER (PRE-PROCESSING) ---
    print("\nApplying Fixed-Pattern Sensor Noise Filter...")
    noise_filter = np.load(os.path.join(_dir, "master_noise_filter_19k.npy")).astype(np.float32)

    # FIX: Create new variables for filtered data so we DON'T overwrite the raw arrays!
    filtered_noisy_train_data = np.clip(noisy_train_data - noise_filter, 0.0, 255.0)
    filtered_blind_test_data = np.clip(blind_test_data - noise_filter, 0.0, 255.0)

    # Proceed to Normalize and Convert FILTERED Data to Tensors (for training)
    full_noisy_tensor = torch.tensor(normalize_data(filtered_noisy_train_data)).unsqueeze(1)
    full_clean_tensor = torch.tensor(normalize_data(clean_train_data)).unsqueeze(1)
    dataset = TensorDataset(full_noisy_tensor, full_clean_tensor)

    blind_test_tensor = torch.tensor(normalize_data(filtered_blind_test_data)).unsqueeze(1)

    # FIX: Convert RAW data to tensors specifically for the "Before" pictures in the plot
    raw_noisy_train_tensor = torch.tensor(normalize_data(noisy_train_data)).unsqueeze(1)
    raw_blind_test_tensor = torch.tensor(normalize_data(blind_test_data)).unsqueeze(1)

    # --- PHASE 1: AUTOMATED OPTUNA SWEEP ---
    print("\n--- Phase 1: Commencing Scaled Optuna Sweep (Rubric: Config Analysis) ---")

    sweep_size = int(0.1 * len(dataset))
    sweep_train_size = int(0.8 * sweep_size)
    sweep_val_size = sweep_size - sweep_train_size
    sweep_subset, _ = random_split(dataset, [sweep_size, len(dataset) - sweep_size])
    sweep_train, sweep_val = random_split(sweep_subset, [sweep_train_size, sweep_val_size])


    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch_size = 64
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)

        _, _, val_losses = train_model(
            sweep_train, sweep_val,
            base_channels=128, num_blocks=8, dropout_rate=dropout_rate,
            lr=lr, batch_size=batch_size, num_epochs=15
        )
        return val_losses[-1]


    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15)

    best_params = study.best_trial.params
    best_params["batch_size"] = 64
    print(f"\n[Sweep Complete] Best Parameters Autonomously Found: {best_params}")

    # --- PHASE 2: MANDATORY K-FOLD CROSS Validation ---
    print("\n--- Phase 2: Commencing Heavyweight 5-Fold Cross Validation ---")

    K = 5
    kf = KFold(n_splits=K, shuffle=True, random_state=42)

    all_train_losses, all_val_losses = [], []
    best_overall_val_loss = float('inf')
    master_model = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)))):
        print(f"\n--- Fold {fold + 1}/{K} ---")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        model, train_losses, val_losses = train_model(
            train_subset, val_subset,
            base_channels=128,
            num_blocks=8,
            dropout_rate=best_params["dropout_rate"],
            lr=best_params["lr"],
            batch_size=best_params["batch_size"],
            num_epochs=100,
        )

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        if val_losses[-1] < best_overall_val_loss:
            best_overall_val_loss = val_losses[-1]
            master_model = model
            print(f"*** New Master Model saved from Fold {fold + 1} ***")

    # --- PHASE 3: METRICS & EXPORT FOR REPORT ---
    print("\n--- Phase 3: Generating Rubric Visualizations ---")

    print("\n--- Final K-Fold Numerical Results ---")
    for i, (tl, vl) in enumerate(zip(all_train_losses, all_val_losses)):
        print(f"Fold {i + 1} | Final Train MSE: {tl[-1]:.6f} | Final Val MSE: {vl[-1]:.6f}")

    # UPDATED GRAPH FIX: Explicitly labelling every fold and formatting the legend
    plt.figure(figsize=(12, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # distinct colors for 5 folds
    for i, (tl, vl) in enumerate(zip(all_train_losses, all_val_losses)):
        plt.plot(tl, color=colors[i], alpha=0.5, label=f'Fold {i + 1} Train')
        plt.plot(vl, color=colors[i], alpha=0.8, linestyle='--', label=f'Fold {i + 1} Val')

    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Heavyweight ResNet K-Fold Cross Validation (K={K})')
    # Put legend outside the plot so it doesn't block the lines
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(_dir, "Rubric_KFold_Losses.png"), dpi=200)

    # FIX: Pass the raw tensors into the plotting functions!
    plot_rubric_train_samples(master_model, dataset, raw_noisy_train_tensor, [0, 42], device)
    plot_rubric_test_samples(master_model, blind_test_tensor, raw_blind_test_tensor, [0, 42], device)

    # --- PHASE 4: FINAL BLIND TEST SET PREDICTION ---
    print("\n--- Phase 4: Generating Final .npz Submission File ---")

    master_model.eval()
    all_predictions = []

    test_loader = DataLoader(TensorDataset(blind_test_tensor), batch_size=64, shuffle=False)

    with torch.no_grad():
        for batch in test_loader:
            noisy_batch = batch[0].to(device)
            predicted_batch = master_model(noisy_batch).cpu().numpy()
            all_predictions.append(predicted_batch)

    final_output_array = np.concatenate(all_predictions, axis=0)
    final_output_array = final_output_array.squeeze(1)

    submission_file = os.path.join(_dir, "Canim_Group_prediction.npz")
    np.savez(submission_file, prediction=final_output_array)

    # FIX: Save the PyTorch weights so they don't vanish!
    weights_file = os.path.join(_dir, "ResNet_Best_Weights.pth")
    torch.save(master_model.state_dict(), weights_file)

    print(f"\nSUCCESS: Pipeline Complete.")
    print(f"Your prediction file is ready: {submission_file}")
