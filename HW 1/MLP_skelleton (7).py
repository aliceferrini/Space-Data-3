import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt


# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        super(MLP, self).__init__()

        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = h
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def normalize_data(data, data_min=0.0, data_max=255.0):
    return (data - data_min) / (data_max - data_min)


def mean_squared_error(predictions, targets):
    return torch.mean((predictions - targets) ** 2)


def train_model(train_dataset, val_dataset,
                input_size=64, hidden_sizes=None, output_size=64,
                dropout_rate=0.2, lr=1e-3, batch_size=32, num_epochs=300):

    if hidden_sizes is None:
        hidden_sizes = [256, 256, 128]

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
    model = MLP(input_size, hidden_sizes, output_size, dropout_rate).to(device)
    # weight_decay adds L2 regularisation to reduce overfitting
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

    # Return the trained model and loss curves
    return model, train_losses, val_losses


def plot_noisy_clean_predicted(model, dataset, index, title, device):
    model.eval()
    noisy_profile, clean_profile = dataset[index]

    with torch.no_grad():
        predicted = model(noisy_profile.unsqueeze(0).to(device)).squeeze(0).cpu()

    x = np.arange(len(noisy_profile))

    plt.figure(figsize=(10, 4))
    plt.plot(x, noisy_profile.numpy(),  label='Noisy Input',         alpha=0.5, color='gray')
    plt.plot(x, clean_profile.numpy(), label='Clean Ground Truth',   color='green', linewidth=2)
    plt.plot(x, predicted.numpy(),     label='Predicted (Denoised)', color='red',
             linestyle='--', linewidth=2)
    plt.title(title)
    plt.xlabel('Pixel Index')
    plt.ylabel('Normalised Brightness')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # (A) Load data (example: shape (N, H, W))
    _dir = os.path.dirname(os.path.abspath(__file__))
    noisy_data = np.load(os.path.join(_dir, "noisy_images_small_1k.npy")).astype(np.float32)
    clean_data = np.load(os.path.join(_dir, "clean_images_small_1k.npy")).astype(np.float32)

    # Convert to row-wise brightness profiles, shape (N, H)
    noisy_profiles = noisy_data.mean(axis=2)
    clean_profiles = clean_data.mean(axis=2)

    # Normalise both sets to [0, 1]
    noisy_profiles = normalize_data(noisy_profiles)
    clean_profiles = normalize_data(clean_profiles)

    # Convert to torch tensors and create the TensorDataset
    noisy_tensor = torch.tensor(noisy_profiles)
    clean_tensor = torch.tensor(clean_profiles)
    dataset = TensorDataset(noisy_tensor, clean_tensor)

    # Split into train (80 %) and validation (20 %) sets
    n_val   = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, train_losses, val_losses = train_model(
        train_dataset, val_dataset,
        input_size=64,
        hidden_sizes=[256, 256, 128],
        output_size=64,
        dropout_rate=0.2,
        lr=1e-3,
        batch_size=32,
        num_epochs=300,
    )

    # Set model to eval mode
    model.eval()

    # Plot training / validation loss curves
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot a sample from the training set and one from the validation set
    plot_noisy_clean_predicted(
        model, train_dataset, index=0,
        title="Training Sample — Noisy vs Clean vs Predicted",
        device=device,
    )
    plot_noisy_clean_predicted(
        model, val_dataset, index=0,
        title="Validation Sample — Noisy vs Clean vs Predicted",
        device=device,
    )
