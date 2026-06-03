"""
Local smoke test for CNN_L1.py
Generates tiny fake .npy data and runs the full pipeline with
reduced settings (5 epochs, 50 images) so it finishes in seconds.
"""
import os
import sys
import numpy as np
import tempfile
import shutil

# ── Config ────────────────────────────────────────────────────────────────────
N_TRAIN = 50       # number of fake training images
N_TEST  = 10       # number of fake test images
H, W    = 32, 32   # small image size to keep it fast
EPOCHS  = 5

_dir    = os.path.dirname(os.path.abspath(__file__))
tmp_dir = tempfile.mkdtemp(prefix="cnn_l1_test_")

print("=" * 60)
print(f"Smoke test — tmp output: {tmp_dir}")
print("=" * 60)

# ── Generate fake .npy data ───────────────────────────────────────────────────
rng = np.random.default_rng(42)

clean_train = rng.uniform(0, 255, (N_TRAIN, H, W)).astype(np.float32)
noisy_train = (clean_train + rng.normal(0, 30, clean_train.shape)).clip(0, 255).astype(np.float32)
noisy_test  = rng.uniform(0, 255, (N_TEST,  H, W)).astype(np.float32)

np.save(os.path.join(tmp_dir, "clean_train.npy"), clean_train)
np.save(os.path.join(tmp_dir, "noisy_train.npy"), noisy_train)
np.save(os.path.join(tmp_dir, "noisy_test.npy"),  noisy_test)

print(f"[OK] Fake data created  (train: {N_TRAIN}x{H}x{W}, test: {N_TEST}x{H}x{W})")

# ── Patch CNN_L1 to use tmp paths and fewer epochs ────────────────────────────
import importlib, types

# Temporarily redirect imports so CNN_L1 writes to tmp_dir
original_dir = os.getcwd()
os.chdir(tmp_dir)
sys.path.insert(0, _dir)

import CNN_L1 as cnn_module

# ── Run train_model directly with tiny settings ───────────────────────────────
import torch
from torch.utils.data import TensorDataset, Subset
from sklearn.model_selection import KFold

def normalize(data): return (data - 0.0) / 255.0

noisy_t = torch.tensor(normalize(noisy_train)).unsqueeze(1)
clean_t = torch.tensor(normalize(clean_train)).unsqueeze(1)
dataset = TensorDataset(noisy_t, clean_t)

K  = 2   # 2 folds instead of 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)

all_train_losses = []
all_val_losses   = []
best_val_loss    = float('inf')
best_model       = None
best_train_sub   = None
best_val_sub     = None
device           = torch.device('cpu')

print("\n[RUN] Training loop …")
for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)))):
    print(f"  Fold {fold+1}/{K}")
    train_sub = Subset(dataset, train_idx)
    val_sub   = Subset(dataset, val_idx)

    model, tl, vl = cnn_module.train_model(
        train_sub, val_sub,
        in_channels=1, base_channels=8,   # tiny model
        dropout_rate=0.0, lr=1e-3,
        batch_size=16, num_epochs=EPOCHS,
    )
    all_train_losses.append(tl)
    all_val_losses.append(vl)

    if vl[-1] < best_val_loss:
        best_val_loss  = vl[-1]
        best_model     = model
        best_train_sub = train_sub
        best_val_sub   = val_sub

print("[OK] Training done")

# ── CV plot ───────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

avg_train = np.mean(all_train_losses, axis=0)
avg_val   = np.mean(all_val_losses,   axis=0)

fig, ax1 = plt.subplots(figsize=(9, 4))
ax2 = ax1.twinx()
for tl, vl in zip(all_train_losses, all_val_losses):
    ax1.plot(tl, alpha=0.20, color='steelblue')
    ax2.plot(vl, alpha=0.20, color='darkorange')
l1, = ax1.plot(avg_train, label='Avg Train L1 (MAE)', color='steelblue',  linewidth=2)
l2, = ax2.plot(avg_val,   label='Avg Val MSE',        color='darkorange', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train L1 (MAE)', color='steelblue')
ax2.set_ylabel('Val MSE',        color='darkorange')
ax1.tick_params(axis='y', labelcolor='steelblue')
ax2.tick_params(axis='y', labelcolor='darkorange')
fig.legend([l1, l2], ['Avg Train L1 (MAE)', 'Avg Val MSE'],
           loc='upper right', bbox_to_anchor=(0.88, 0.88))
plt.title(f'{K}-Fold CV — CNN (Train: L1 | Val: MSE) [SMOKE TEST]')
plt.tight_layout()
cv_path = os.path.join(tmp_dir, "cv_plot.png")
plt.savefig(cv_path, dpi=100)
plt.close()
print(f"[OK] CV plot saved: {cv_path}")

# ── Save model ────────────────────────────────────────────────────────────────
model_path = os.path.join(tmp_dir, "model_best.pth")
torch.save(best_model.state_dict(), model_path)
print(f"[OK] Model saved: {model_path}")

# ── Reload model ──────────────────────────────────────────────────────────────
loaded = cnn_module.CNN(in_channels=1, base_channels=8, dropout_rate=0.0)
loaded.load_state_dict(torch.load(model_path, map_location='cpu'))
loaded.eval()
print("[OK] Model reloaded successfully")

# ── Train/val sample plots ────────────────────────────────────────────────────
for idx in [0, 1]:
    cnn_module.plot_train_sample(best_model, best_train_sub, idx,
                                 f"Train Sample {idx}", device, tmp_dir)
    cnn_module.plot_train_sample(best_model, best_val_sub,   idx,
                                 f"Val Sample {idx}",   device, tmp_dir)
print("[OK] Train/val plots saved")

# ── Test inference ────────────────────────────────────────────────────────────
from torch.utils.data import DataLoader

test_noisy_norm = normalize(noisy_test)
test_tensor     = torch.tensor(test_noisy_norm).unsqueeze(1)
loader          = DataLoader(TensorDataset(test_tensor), batch_size=16, shuffle=False)

best_model.eval()
denoised_list = []
with torch.no_grad():
    for (batch,) in loader:
        denoised_list.append(best_model(batch).cpu())

denoised_np = torch.cat(denoised_list).squeeze(1).numpy()
denoised_np = (denoised_np * 255.0).clip(0, 255).astype(np.float32)
pred_path   = os.path.join(tmp_dir, "ferrini_prediction.npz")
np.savez(pred_path, denoised_images=denoised_np)
print(f"[OK] Predictions saved: {pred_path}")
print(f"     Shape: {denoised_np.shape}  |  min={denoised_np.min():.1f}  max={denoised_np.max():.1f}")

# ── Test sample plots ─────────────────────────────────────────────────────────
for idx in [0, 1]:
    cnn_module.plot_test_sample(best_model, test_tensor[idx], idx,
                                f"Test Sample {idx}", device, tmp_dir)
print("[OK] Test plots saved")

# ── Summary ───────────────────────────────────────────────────────────────────
os.chdir(original_dir)
print("\n" + "=" * 60)
print("ALL CHECKS PASSED")
print(f"Output files in: {tmp_dir}")
print("=" * 60)
