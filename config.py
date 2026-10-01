"""
Configuration parameters for EV Battery Pack Digital Twin, IoT Telemetry, and AI Models.
Aligned with Automotive BMS Standards (ISO 26262, UN 38.3, SAE J2464).
"""
import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(MODELS_DIR, "battery_ai_model.pt")

# Single Cell Specifications (Li-ion NMC 21700 Format)
CELL_CONFIG = {
    "chemistry": "Lithium Nickel Manganese Cobalt Oxide (NMC 811)",
    "form_factor": "21700 Cylindrical",
    "nominal_capacity_ah": 5.0,        # Nominal capacity in Ah
    "nominal_voltage_v": 3.7,          # Nominal voltage in Volts
    "max_voltage_v": 4.20,             # Upper cut-off voltage
    "min_voltage_v": 3.00,             # Lower cut-off voltage
    "r0_nominal_ohm": 0.022,           # Nominal ohmic internal resistance (22 mOhm)
    "r1_nominal_ohm": 0.015,           # Polarization resistance (15 mOhm)
    "c1_nominal_farad": 2200.0,        # Polarization capacitance (RC time constant ~33s)
    "mass_kg": 0.068,                  # Cell mass (68g)
    "specific_heat_j_per_kg_k": 920.0, # Specific heat capacity
    "heat_transfer_coeff_w_per_k": 0.18, # Convective cooling coefficient (h * A)
    "ambient_temp_c": 25.0,            # Standard ambient temperature
    "coulombic_efficiency": 0.995,     # Efficiency during charge
    "cycle_life_to_80_pct": 1500,      # Cycles to 80% SOH (End of Life)
}

# Battery Pack Specifications (16S2P Micro-EV / Two-Wheeler / Light Commercial EV Module)
PACK_CONFIG = {
    "pack_id": "EV-PACK-60V-TWIN-01",
    "standards_compliance": ["ISO 26262 ASIL-C", "UN 38.3", "SAE J2464"],
    "series_cells": 16,                # 16 cells in series (Nominal ~59.2V)
    "parallel_strings": 2,             # 2 parallel cells per group (~10 Ah capacity)
    "nominal_pack_energy_kwh": 0.592,  # 592 Wh nominal energy
    "nominal_voltage_v": 59.2,         # 16 * 3.7V
    "max_voltage_v": 67.2,             # 16 * 4.2V
    "min_voltage_v": 48.0,             # 16 * 3.0V
    "max_continuous_discharge_a": 35.0,# Continuous current limit (SoP)
    "peak_pulse_discharge_a": 50.0,    # 10-second acceleration pulse limit
    "max_charge_current_a": -15.0,     # Fast charge limit
    "peak_regen_current_a": -25.0,     # Regenerative braking pulse limit
    "max_cell_temp_c": 55.0,           # Warning threshold
    "critical_cell_temp_c": 65.0,      # Thermal runaway alarm threshold
    "over_voltage_limit_v": 4.25,      # Cell over-voltage trip (OVP)
    "under_voltage_limit_v": 2.85,     # Cell under-voltage trip (UVP)
    "max_voltage_imbalance_v": 0.080,  # Max allowed delta V across cells before balancing alarm
    "passive_balancing_threshold_v": 4.10, # Passive bleed triggers above this voltage
    "balancing_bleed_current_a": 0.15, # Passive shunting current
    "energy_consumption_wh_per_km": 30.0, # Vehicle efficiency rating
}

# MQTT IoT Settings
MQTT_CONFIG = {
    "broker_host": os.getenv("MQTT_HOST", "broker.hivemq.com"),  # Default HiveMQ Public Broker
    "broker_port": int(os.getenv("MQTT_PORT", 1883)),
    "username": os.getenv("HIVEMQ_USERNAME", None),
    "password": os.getenv("HIVEMQ_PASSWORD", None),
    "use_tls": os.getenv("MQTT_USE_TLS", "false").lower() in ("true", "1", "yes"),
    "keepalive": 60,
    "topic_telemetry": "ev/battery/pack01/telemetry",
    "topic_control": "ev/battery/pack01/control",
    "topic_twin_estimate": "ev/battery/pack01/twin/estimate",
    "client_id_edge": "ev_bms_edge_gateway_01",
    "client_id_twin": "ev_digital_twin_service_01",
    "client_id_dashboard": "ev_dashboard_client_01",
    "telemetry_interval_sec": 0.5,     # Streaming frequency (2 Hz)
    "hivemq_serverless": {
        "cluster_url": os.getenv("HIVEMQ_CLUSTER_URL", ""),
        "port": 8883,
        "use_tls": True,
    },
}

# AI Neural Network Architecture Settings
AI_CONFIG = {
    "sequence_length": 30,             # Number of historical time-steps for RNN/GRU
    "input_dim": 6,                    # [pack_voltage, pack_current, avg_temp, v_std, temp_std, v_delta]
    "hidden_dim": 64,                  # Hidden dimensions in GRU/LSTM
    "num_layers": 2,                   # Number of recurrent layers
    "dropout": 0.15,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 25,
    "device": "cpu",
}
