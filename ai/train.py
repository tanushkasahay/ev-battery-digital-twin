import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from config import AI_CONFIG, MODEL_PATH, MODELS_DIR
from ai.network import BatteryStateEstimatorNN
from ai.dataset_generator import generate_training_data, BatteryTelemetryDataset


def train_model():
    os.makedirs(MODELS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using computing device: {device}")

    # 1. Generate synthetic telemetry sequences
    seq_len = AI_CONFIG["sequence_length"]
    sequences, targets, _ = generate_training_data(
        num_cycles=10,
        steps_per_cycle=350,
        seq_len=seq_len,
        dt=1.0,
    )

    full_dataset = BatteryTelemetryDataset(sequences, targets)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_data, val_data = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_data, batch_size=AI_CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=AI_CONFIG["batch_size"], shuffle=False)

    # 2. Instantiate Model, Loss Function, and Optimizer
    model = BatteryStateEstimatorNN(
        input_dim=AI_CONFIG["input_dim"],
        hidden_dim=AI_CONFIG["hidden_dim"],
        num_layers=AI_CONFIG["num_layers"],
        dropout=AI_CONFIG["dropout"],
    ).to(device)

    criterion = nn.SmoothL1Loss()  # Huber loss is resilient against noisy sensor outliers
    optimizer = torch.optim.AdamW(model.parameters(), lr=AI_CONFIG["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # 3. Training Loop
    epochs = AI_CONFIG["epochs"]
    print(f"\n--- Starting AI Training ({epochs} Epochs) ---")
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred_soc, pred_soh = model(x_batch)

            target_soc = y_batch[:, 0].unsqueeze(1)
            target_soh = y_batch[:, 1].unsqueeze(1)

            loss_soc = criterion(pred_soc, target_soc)
            loss_soh = criterion(pred_soh, target_soh)
            total_loss = loss_soc + 0.5 * loss_soh

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += total_loss.item() * len(x_batch)

        avg_train_loss = total_train_loss / train_size

        # Validation Step
        model.eval()
        total_val_loss = 0.0
        soc_errors = []

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                pred_soc, pred_soh = model(x_val)
                target_soc = y_val[:, 0].unsqueeze(1)
                target_soh = y_val[:, 1].unsqueeze(1)

                v_loss = criterion(pred_soc, target_soc) + 0.5 * criterion(pred_soh, target_soh)
                total_val_loss += v_loss.item() * len(x_val)

                # Track SOC absolute error
                abs_err = torch.abs(pred_soc - target_soc).cpu().numpy()
                soc_errors.extend(abs_err.flatten())

        avg_val_loss = total_val_loss / val_size
        mae_soc = float(torch.tensor(soc_errors).mean()) * 100.0  # as percentage
        scheduler.step(avg_val_loss)

        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {avg_train_loss:.5f} | "
                f"Val Loss: {avg_val_loss:.5f} | "
                f"SOC MAE: {mae_soc:.2f}%"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save model checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "soc_mae_pct": mae_soc,
                    "input_dim": AI_CONFIG["input_dim"],
                    "hidden_dim": AI_CONFIG["hidden_dim"],
                },
                MODEL_PATH,
            )

    print(f"\nModel training successfully finished! Checkpoint saved to: {MODEL_PATH}")
    print(f"Best Validation Loss: {best_val_loss:.6f} | Final SOC Mean Absolute Error: {mae_soc:.2f}%")


if __name__ == "__main__":
    train_model()
