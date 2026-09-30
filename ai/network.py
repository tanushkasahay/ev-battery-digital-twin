"""
PyTorch Neural Network Architecture for Battery State Estimation (SoC and SoH).
Uses a Multi-Layer Gated Recurrent Unit (GRU) with Temporal Attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Attention mechanism over the sequence time dimension."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_outputs):
        # rnn_outputs shape: [batch_size, seq_len, hidden_dim]
        scores = self.fc(rnn_outputs)  # [batch_size, seq_len, 1]
        weights = F.softmax(scores, dim=1)  # [batch_size, seq_len, 1]
        context = torch.sum(weights * rnn_outputs, dim=1)  # [batch_size, hidden_dim]
        return context, weights


class BatteryStateEstimatorNN(nn.Module):
    """
    Deep Neural Network for real-time EV Battery State Estimation.
    Inputs: Sequence of battery measurements [V_pack, I_measured, T_avg, V_std, T_std, V_delta].
    Outputs:
    1. State of Charge (SoC) in [0, 1]
    2. State of Health (SoH) in [0, 1]
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input feature projection layer
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Recurrent feature extraction
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Attention layer
        self.attention = TemporalAttention(hidden_dim)

        # Separate prediction heads for SoC and SoH
        self.soc_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # SoC is bounded in [0, 1]
        )

        self.soh_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # SoH is bounded in [0, 1]
        )

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, sequence_length, input_dim]
        Returns:
            soc_pred: [batch_size, 1]
            soh_pred: [batch_size, 1]
        """
        # Linear projection
        proj = self.input_proj(x)
        
        # GRU temporal dynamics
        gru_out, _ = self.gru(proj)  # [batch_size, seq_len, hidden_dim]
        
        # Temporal context via attention
        context, _ = self.attention(gru_out)  # [batch_size, hidden_dim]

        # Multi-task heads
        soc_pred = self.soc_head(context)
        soh_pred = self.soh_head(context)

        return soc_pred, soh_pred
