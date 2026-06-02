import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


# ── Model (must match CNN_L1.py) ──────────────────────────────────────────────
class CNN(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, dropout_rate=0.2):
        super(CNN, self).__init__()
        self.conv1   = nn.Conv2d(in_channels,        base_channels,     kernel_size=7, padding=3)
        self.bn1     = nn.BatchNorm2d(base_channels)
        self.drop1   = nn.Dropout2d(dropout_rate)
        self.conv2   = nn.Conv2d(base_channels,      base_channels * 2, kernel_size=5, padding=2)
        self.bn2     = nn.BatchNorm2d(base_channels * 2)
        self.drop2   = nn.Dropout2d(dropout_rate)
        self.conv3   = nn.Conv2d(base_channels * 2,  base_channels * 2, kernel_size=3, padding=1)
        self.bn3     = nn.BatchNorm2d(base_channels * 2)
        self.conv4   = nn.Conv2d(base_channels * 2,  base_channels,     kernel_size=3, padding=1)
        self.bn4     = nn.BatchNorm2d(base_channels)
        self.conv5   = nn.Conv2d(base_channels,      in_channels,       kernel_size=1)
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


# ── Config ────────────────────────────────────────────────────────────────────
_dir       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_dir, "model_best.pth")
TEST_NOISY = "noisy_val_1k_harder.npy"
BATCH_SIZE = 64
NUM_SAMPLES_TO_PLOT = 5   # how many test images to save as PNG

# ── Load model ────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Loading model from: {MODEL_PATH}")

model = CNN(in_channels=1, base_channels=64, dropout_rate=0.2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model loaded.")

# ── Load test data ────────────────────────────────────────────────────────────
test_path = os.path.join(_dir, TEST_NOISY)
print(f"Loading test data from: {test_path}")

test_noisy_raw  = np.load(test_path).astype(np.float32)
test_noisy_norm = normalize_data(test_noisy_raw)
test_tensor     = torch.tensor(test_noisy_norm).unsqueeze(1)   # (N, 1, H, W)
print(f"Test set: {test_tensor.shape[0]} images")

# ── Run inference ─────────────────────────────────────────────────────────────
pin = device.type == 'cuda'
loader = DataLoader(TensorDataset(test_tensor), batch_size=BATCH_SIZE,
                    shuffle=False, num_workers=4, pin_memory=pin)

denoised_list = []
with torch.no_grad():
    for (batch,) in loader:
        out = model(batch.to(device)).cpu()
        denoised_list.append(out)

denoised_tensor = torch.cat(denoised_list, dim=0)                   # (N, 1, H, W)
denoised_np     = denoised_tensor.squeeze(1).numpy()                # (N, H, W)
denoised_np     = (denoised_np * 255.0).clip(0, 255).astype(np.float32)

# ── Save predictions ──────────────────────────────────────────────────────────
pred_path = os.path.join(_dir, "ferrini_prediction.npz")
np.savez(pred_path, denoised_images=denoised_np)
print(f"Saved predictions to: {pred_path}")

# ── Save sample PNGs ──────────────────────────────────────────────────────────
for idx in range(min(NUM_SAMPLES_TO_PLOT, len(test_tensor))):
    noisy_img   = test_tensor[idx].squeeze().numpy()
    denoised_img = denoised_tensor[idx].squeeze().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(noisy_img,    cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Noisy Input')
    axes[1].imshow(denoised_img, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Denoised Output')
    for ax in axes:
        ax.axis('off')
    plt.suptitle(f'Test Sample {idx}')
    plt.tight_layout()
    path = os.path.join(_dir, f"test_sample_{idx}.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")

print("\nDone.")
