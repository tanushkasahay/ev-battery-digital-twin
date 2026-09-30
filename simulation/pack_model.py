"""
Multi-Cell EV Battery Pack Model with Cell-to-Cell Variability, Thermal Gradients, and BMS Protections.
"""
import random
import numpy as np
from config import CELL_CONFIG, PACK_CONFIG
from simulation.cell_model import BatteryCell


class BatteryPack:
    """
    Simulates an EV Battery Pack consisting of N_s series cells with parallel strings.
    Features:
    - Cell-to-cell manufacturing variations.
    - Thermal gradients across pack geometry.
    - BMS safety protection trips and passive balancing.
    - Naive Coulomb counting accumulator with simulated sensor drift.
    """

    def __init__(self, num_series: int = None, num_parallel: int = None):
        self.num_series = num_series or PACK_CONFIG["series_cells"]
        self.num_parallel = num_parallel or PACK_CONFIG["parallel_strings"]
        self.pack_id = PACK_CONFIG["pack_id"]
        
        # Build individual cells with Gaussian manufacturing deviations
        self.cells = []
        random.seed(42)  # Deterministic seed for reproducible baseline
        
        nominal_cap = CELL_CONFIG["nominal_capacity_ah"]
        nominal_r0 = CELL_CONFIG["r0_nominal_ohm"]
        ambient_temp = CELL_CONFIG["ambient_temp_c"]

        for i in range(self.num_series):
            # Inner cells (middle of pack) have slightly worse cooling
            distance_from_edge = min(i, self.num_series - 1 - i)
            cooling_penalty = 1.0 - (0.15 * (distance_from_edge / (self.num_series / 2)))
            
            # Manufacturing variations
            cap_i = nominal_cap * (1.0 + random.gauss(0, 0.012))
            r0_i = nominal_r0 * (1.0 + random.gauss(0, 0.025))
            initial_soc_i = 0.88 + random.gauss(0, 0.005)
            
            cell = BatteryCell(
                cell_id=i + 1,
                nominal_capacity_ah=cap_i,
                nominal_voltage_v=CELL_CONFIG["nominal_voltage_v"],
                initial_soc=initial_soc_i,
                initial_soh=1.0,
                initial_temp_c=ambient_temp,
                r0_ohm=r0_i,
                r1_ohm=CELL_CONFIG["r1_nominal_ohm"],
                c1_farad=CELL_CONFIG["c1_nominal_farad"],
                mass_kg=CELL_CONFIG["mass_kg"],
                ambient_temp_c=ambient_temp,
                ha_w_per_k=CELL_CONFIG["heat_transfer_coeff_w_per_k"] * cooling_penalty,
            )
            self.cells.append(cell)

        # Naive BMS Coulomb Counting State (accumulates sensor error/drift)
        self.bms_coulomb_soc = 0.88
        self.current_sensor_bias_a = 0.15   # Simulated 150mA current sensor offset
        self.total_cycle_count = 0.0

        # Balancing status for each cell (True if bleed resistor is ON)
        self.balancing_active = [False] * self.num_series

    def step(self, pack_current_a: float, dt: float = 1.0) -> dict:
        """
        Steps the pack forward in time with load current pack_current_a (Amperes).
        pack_current_a > 0: Discharging (power delivered to inverter/motor)
        pack_current_a < 0: Charging (regen braking or plug-in charging)
        """
        # Current per parallel cell string
        cell_current_a = pack_current_a / self.num_parallel

        cell_results = []
        cell_voltages = []
        cell_temps = []
        cell_socs = []
        cell_sohs = []

        # 1. Step each individual cell
        for i, cell in enumerate(self.cells):
            eff_current = cell_current_a
            # If passive balancing is active, shunt bleeding current
            if self.balancing_active[i] and pack_current_a < 0:
                eff_current += PACK_CONFIG["balancing_bleed_current_a"]

            res = cell.step(eff_current, dt=dt)
            cell_results.append(res)
            cell_voltages.append(res["voltage"])
            cell_temps.append(res["temp_c"])
            cell_socs.append(res["soc"])
            cell_sohs.append(res["soh"])

        # 2. Compute aggregate pack metrics
        pack_voltage = float(np.sum(cell_voltages))
        max_v = float(np.max(cell_voltages))
        min_v = float(np.min(cell_voltages))
        delta_v = max_v - min_v

        max_t = float(np.max(cell_temps))
        min_t = float(np.min(cell_temps))
        avg_t = float(np.mean(cell_temps))

        true_soc_mean = float(np.mean(cell_socs))
        true_soh_pack = float(np.min(cell_sohs))  # Weakest link principle
        power_kw = (pack_voltage * pack_current_a) / 1000.0

        # 3. Naive BMS Coulomb Counting (Simulates hardware sensor with drift)
        measured_current = pack_current_a + self.current_sensor_bias_a + random.gauss(0, 0.08)
        pack_nominal_ah = CELL_CONFIG["nominal_capacity_ah"] * self.num_parallel
        delta_coulomb_soc = -(measured_current * (dt / 3600.0)) / pack_nominal_ah
        self.bms_coulomb_soc = float(np.clip(self.bms_coulomb_soc + delta_coulomb_soc, 0.0, 1.0))

        # 4. BMS Safety Alarms & Interlocks
        alarms = []
        if max_v >= PACK_CONFIG["over_voltage_limit_v"]:
            alarms.append("OVER_VOLTAGE_ALARM")
        if min_v <= PACK_CONFIG["under_voltage_limit_v"]:
            alarms.append("UNDER_VOLTAGE_ALARM")
        if max_t >= PACK_CONFIG["critical_cell_temp_c"]:
            alarms.append("CRITICAL_THERMAL_RUNAWAY_RISK")
        elif max_t >= PACK_CONFIG["max_cell_temp_c"]:
            alarms.append("OVER_TEMPERATURE_WARNING")
        if delta_v >= PACK_CONFIG["max_voltage_imbalance_v"]:
            alarms.append("CELL_IMBALANCE_WARNING")

        # 5. Passive Balancing Logic
        for i, v in enumerate(cell_voltages):
            if (
                pack_current_a < 0
                and v > PACK_CONFIG["passive_balancing_threshold_v"]
                and (v - min_v) > 0.025
            ):
                self.balancing_active[i] = True
            else:
                self.balancing_active[i] = False

        return {
            "pack_id": self.pack_id,
            "pack_voltage": round(pack_voltage, 2),
            "pack_current": round(pack_current_a, 2),
            "measured_current": round(measured_current, 2),
            "power_kw": round(power_kw, 3),
            "true_soc": round(true_soc_mean, 4),
            "bms_coulomb_soc": round(self.bms_coulomb_soc, 4),
            "true_soh": round(true_soh_pack, 4),
            "cell_voltages": [round(v, 4) for v in cell_voltages],
            "cell_temps": [round(t, 2) for t in cell_temps],
            "cell_socs": [round(s, 4) for s in cell_socs],
            "max_cell_voltage": round(max_v, 4),
            "min_cell_voltage": round(min_v, 4),
            "delta_cell_voltage": round(delta_v, 4),
            "max_cell_temp": round(max_t, 2),
            "min_cell_temp": round(min_t, 2),
            "avg_cell_temp": round(avg_t, 2),
            "balancing_active": self.balancing_active.copy(),
            "alarms": alarms,
            "bms_healthy": len(alarms) == 0,
        }

    def inject_cell_fault(self, cell_index: int, fault_type: str, severity: float = 1.0):
        """Inject anomaly into a specific cell (0-indexed)."""
        if 0 <= cell_index < len(self.cells):
            self.cells[cell_index].inject_fault(fault_type, severity)

    def reset_all_faults(self):
        """Clear all active faults on all cells."""
        for cell in self.cells:
            cell.inject_fault("clear")
