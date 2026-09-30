"""
Battery Digital Twin Anomaly and Fault Detection Engine.
Detects cell imbalances, rapid thermal rise (early runaway warning), internal short-circuits, and SOH degradation.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import PACK_CONFIG


class BatteryAnomalyDetector:
    """
    Analyzes live battery telemetry and AI state residuals to flag anomalies.
    """

    def __init__(self):
        self.prev_cell_temps = None
        self.prev_timestamp = None

    def analyze(self, telemetry: dict, ai_estimate: dict) -> dict:
        anomalies = []
        severity = "NORMAL"  # "NORMAL", "WARNING", "CRITICAL"

        cell_voltages = telemetry.get("cell_voltages", [])
        cell_temps = telemetry.get("cell_temps", [])
        timestamp = telemetry.get("timestamp", 0.0)
        v_delta = telemetry.get("v_delta", 0.0)
        max_t = telemetry.get("max_temp", 25.0)
        avg_t = telemetry.get("avg_temp", 25.0)

        # 1. Thermal Anomaly & Early Thermal Runaway Detection
        if max_t >= PACK_CONFIG["critical_cell_temp_c"]:
            anomalies.append({
                "type": "CRITICAL_THERMAL_RUNAWAY",
                "message": f"Critical pack temperature reached: {max_t:.1f}°C (Threshold: {PACK_CONFIG['critical_cell_temp_c']}°C)",
                "severity": "CRITICAL",
            })
            severity = "CRITICAL"
        elif max_t >= PACK_CONFIG["max_cell_temp_c"]:
            anomalies.append({
                "type": "OVER_TEMPERATURE_WARNING",
                "message": f"High temperature detected: {max_t:.1f}°C",
                "severity": "WARNING",
            })
            if severity != "CRITICAL":
                severity = "WARNING"

        # Check localized temperature hotspots: cell temp significantly higher than pack mean
        if cell_temps and len(cell_temps) > 0:
            for idx, t in enumerate(cell_temps):
                if (t - avg_t) > 5.0:
                    anomalies.append({
                        "type": "LOCALIZED_HOTSPOT",
                        "message": f"Cell #{idx + 1} hotspot: {t:.1f}°C ({t - avg_t:.1f}°C above mean)",
                        "cell_id": idx + 1,
                        "severity": "WARNING",
                    })
                    if severity != "CRITICAL":
                        severity = "WARNING"

        # Check thermal rate of rise (dT/dt)
        if self.prev_cell_temps and self.prev_timestamp and (timestamp > self.prev_timestamp):
            dt = timestamp - self.prev_timestamp
            if dt > 0.1:
                for idx, (curr_t, prev_t) in enumerate(zip(cell_temps, self.prev_cell_temps)):
                    rate = (curr_t - prev_t) / dt  # °C per second
                    if rate > 0.8:
                        anomalies.append({
                            "type": "RAPID_TEMPERATURE_RISE",
                            "message": f"Cell #{idx + 1} rapid heating rate: +{rate:.2f}°C/s",
                            "cell_id": idx + 1,
                            "severity": "CRITICAL",
                        })
                        severity = "CRITICAL"

        self.prev_cell_temps = list(cell_temps)
        self.prev_timestamp = timestamp

        # 2. Cell Voltage Imbalance
        if v_delta >= PACK_CONFIG["max_voltage_imbalance_v"]:
            anomalies.append({
                "type": "SEVERE_CELL_IMBALANCE",
                "message": f"Cell voltage delta {v_delta*1000:.1f} mV exceeds limit ({PACK_CONFIG['max_voltage_imbalance_v']*1000:.0f} mV)",
                "severity": "WARNING",
            })
            if severity != "CRITICAL":
                severity = "WARNING"

        # 3. Internal Short Circuit (ISC) Indicator
        # Indicated if a cell's voltage is abnormally low while other cells are balanced
        if cell_voltages and len(cell_voltages) > 0:
            min_v = min(cell_voltages)
            avg_v = sum(cell_voltages) / len(cell_voltages)
            if (avg_v - min_v) > 0.09:
                bad_cell = cell_voltages.index(min_v) + 1
                anomalies.append({
                    "type": "SUSPECTED_INTERNAL_SHORT",
                    "message": f"Cell #{bad_cell} voltage lagging by {(avg_v - min_v)*1000:.1f} mV under load",
                    "cell_id": bad_cell,
                    "severity": "CRITICAL",
                })
                severity = "CRITICAL"

        # 4. State of Health (SOH) Degradation Warning
        ai_soh = ai_estimate.get("ai_soh", 1.0)
        if ai_soh < 0.80:
            anomalies.append({
                "type": "BATTERY_END_OF_LIFE_SOH",
                "message": f"AI Twin SOH dropped to {ai_soh*100:.1f}% (Below 80% warranty/EV threshold)",
                "severity": "WARNING",
            })
            if severity != "CRITICAL":
                severity = "WARNING"

        # 5. Naive Coulomb Drift Residual
        coulomb_soc = telemetry.get("bms_coulomb_soc", 0.0)
        ai_soc = ai_estimate.get("ai_soc", 0.0)
        drift = abs(ai_soc - coulomb_soc)
        if drift > 0.08 and ai_estimate.get("buffer_ready", False):
            anomalies.append({
                "type": "BMS_COULOMB_SENSOR_DRIFT",
                "message": f"BMS Coulomb counting diverged from AI Twin by {drift*100:.1f}%",
                "severity": "WARNING",
            })

        return {
            "status": severity,
            "anomaly_count": len(anomalies),
            "anomalies": anomalies,
        }
