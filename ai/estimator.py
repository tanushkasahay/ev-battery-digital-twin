import os
import sys
import json
import logging
from collections import deque
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from config import AI_CONFIG, MODEL_PATH, DATA_DIR
from ai.network import BatteryStateEstimatorNN

logger = logging.getLogger("AIBatteryEstimator")


class BatteryAIEstimator:
    """
    Real-time inference engine executing on the Digital Twin backend.
    """

    def __init__(self, model_path: str = None, scaler_path: str = None):
        self.model_path = model_path or MODEL_PATH
        self.scaler_path = scaler_path or os.path.join(DATA_DIR, "scaler_params.json")
        self.seq_len = AI_CONFIG["sequence_length"]
        self.device = torch.device("cpu")

        # Rolling window buffer for real-time streaming
        self.buffer = deque(maxlen=self.seq_len)
        
        # Load Scaler Parameters
        self.mean = None
        self.std = None
        self._load_scaler()

        # Load Neural Network Model
        self.model = BatteryStateEstimatorNN(
            input_dim=AI_CONFIG["input_dim"],
            hidden_dim=AI_CONFIG["hidden_dim"],
            num_layers=AI_CONFIG["num_layers"],
        ).to(self.device)
        self._load_model()

    def _load_scaler(self):
        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, "r") as f:
                params = json.load(f)
                self.mean = np.array(params["mean"], dtype=np.float32)
                self.std = np.array(params["std"], dtype=np.float32)
            logger.info("Loaded feature scaler parameters.")
        else:
            logger.warning(f"Scaler parameters not found at {self.scaler_path}. Using standard heuristics.")
            self.mean = np.array([60.0, 5.0, 28.0, 0.01, 0.5, 0.03], dtype=np.float32)
            self.std = np.array([5.0, 10.0, 6.0, 0.02, 0.8, 0.04], dtype=np.float32)

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
                self.model.eval()
                logger.info(f"Loaded trained PyTorch battery model from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load PyTorch weights: {e}")
                self.model.eval()
        else:
            logger.warning(f"No checkpoint found at {self.model_path}. Model using initial weights.")
            self.model.eval()

    def update(self, telemetry: dict) -> dict:
        """
        Ingests a single incoming telemetry packet, updates buffer, and runs AI inference.
        """
        # Extract features
        v_std = float(np.std(telemetry.get("cell_voltages", [3.7])))
        t_std = float(np.std(telemetry.get("cell_temps", [25.0])))
        v_delta = float(telemetry.get("v_delta", 0.0))

        feat = np.array(
            [
                telemetry.get("pack_voltage", 60.0),
                telemetry.get("measured_current", telemetry.get("pack_current", 0.0)),
                telemetry.get("avg_temp", 25.0),
                v_std,
                t_std,
                v_delta,
            ],
            dtype=np.float32,
        )

        # Standardize
        norm_feat = (feat - self.mean) / self.std
        self.buffer.append(norm_feat)

        # If buffer is still priming, return heuristic or Coulomb fallback
        if len(self.buffer) < 5:
            coulomb_soc = telemetry.get("bms_coulomb_soc", 0.85)
            return {
                "ai_soc": coulomb_soc,
                "ai_soh": 1.0,
                "estimated_range_km": round(coulomb_soc * 65.0, 1),
                "confidence_score": 0.50,
                "buffer_ready": False,
            }

        # Pad with oldest reading if buffer is between 5 and seq_len
        current_seq = list(self.buffer)
        if len(current_seq) < self.seq_len:
            padding = [current_seq[0]] * (self.seq_len - len(current_seq))
            seq_array = np.array(padding + current_seq, dtype=np.float32)
        else:
            seq_array = np.array(current_seq, dtype=np.float32)

        # Convert to Tensor [1, seq_len, input_dim]
        input_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_soc, pred_soh = self.model(input_tensor)
            ai_soc = float(pred_soc.item())
            ai_soh = float(pred_soh.item())

        # Estimated EV driving range (assuming ~10 Ah * 60V = ~600 Wh pack, ~10 km range for micro-EV or scaled)
        # Scaled to representative 4-wheeler mini EV range (e.g. 50 km nominal at full charge)
        estimated_range_km = round(ai_soc * 55.0 * ai_soh, 1)

        # Confidence metric (higher when buffer full and voltage std is low)
        confidence = round(min(0.99, 0.70 + 0.25 * (len(self.buffer) / self.seq_len)), 2)

        return {
            "ai_soc": round(ai_soc, 4),
            "ai_soh": round(ai_soh, 4),
            "estimated_range_km": estimated_range_km,
            "confidence_score": confidence,
            "buffer_ready": len(self.buffer) >= self.seq_len,
        }
