import os
import sys
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import Dataset
from config import AI_CONFIG, DATA_DIR
from simulation.pack_model import BatteryPack
from simulation.drive_cycles import DriveCycleGenerator


class BatteryTelemetryDataset(Dataset):
    """PyTorch Dataset wrapping sliding-window time series."""
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def generate_training_data(
    num_cycles: int = 12,
    steps_per_cycle: int = 400,
    seq_len: int = 30,
    dt: float = 1.0,
):
    """
    Simulates diverse driving profiles and extracts normalized sequences.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    profiles = ["wltp", "urban", "highway", "aggressive_sport", "fast_charge"]
    
    raw_features = []
    raw_targets = []

    print(f"Generating synthetic telemetry data across {num_cycles} simulation runs...")

    for run_idx in range(num_cycles):
        pack = BatteryPack()
        # Randomize initial pack state
        initial_soc = np.random.uniform(0.30, 0.95)
        for cell in pack.cells:
            cell.soc = float(initial_soc + np.random.normal(0, 0.005))
            cell.soh = float(np.random.uniform(0.85, 1.0))
            cell.temp_c = float(np.random.uniform(18.0, 35.0))
        pack.bms_coulomb_soc = initial_soc

        profile = np.random.choice(profiles)

        for step in range(steps_per_cycle):
            current_a = DriveCycleGenerator.get_current(profile, step, dt=dt)
            res = pack.step(current_a, dt=dt)

            v_std = float(np.std(res["cell_voltages"]))
            t_std = float(np.std(res["cell_temps"]))

            feat = [
                res["pack_voltage"],
                res["measured_current"],
                res["avg_cell_temp"],
                v_std,
                t_std,
                res["delta_cell_voltage"],
            ]
            target = [res["true_soc"], res["true_soh"]]

            raw_features.append(feat)
            raw_targets.append(target)

    raw_features = np.array(raw_features, dtype=np.float32)
    raw_targets = np.array(raw_targets, dtype=np.float32)

    # Compute Feature Normalization Statistics (Mean & Std)
    feat_mean = np.mean(raw_features, axis=0)
    feat_std = np.std(raw_features, axis=0) + 1e-6  # prevent division by zero

    scaler_params = {
        "mean": feat_mean.tolist(),
        "std": feat_std.tolist(),
        "feature_names": ["pack_voltage", "measured_current", "avg_temp", "v_std", "temp_std", "v_delta"],
    }
    scaler_path = os.path.join(DATA_DIR, "scaler_params.json")
    with open(scaler_path, "w") as f:
        json.dump(scaler_params, f, indent=2)
    print(f"Feature scaler parameters saved to: {scaler_path}")

    # Standardize features
    norm_features = (raw_features - feat_mean) / feat_std

    # Build Sliding Window Sequences
    sequences = []
    targets = []
    total_samples = len(norm_features)

    for i in range(total_samples - seq_len):
        seq = norm_features[i : i + seq_len]
        target = raw_targets[i + seq_len]
        sequences.append(seq)
        targets.append(target)

    sequences = np.array(sequences, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    print(f"Generated {len(sequences)} sliding window samples of shape {sequences.shape[1:]}")
    return sequences, targets, scaler_params
