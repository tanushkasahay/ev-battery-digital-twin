"""
Physics-based Equivalent Circuit Model (ECM Thevenin 1RC) and Thermal Dynamics for an EV Battery Cell.
"""
import math
import numpy as np


class BatteryCell:
    """
    Simulates an individual Lithium-ion NMC cell using:
    1. Thevenin 1RC Equivalent Circuit Model (OCV, R0, R1, C1).
    2. Dynamic Lumped-Parameter Thermal Model (Joule heating + Entropic heat + Convection).
    3. Degradation tracking (Capacity fade and Resistance growth -> SOH).
    4. Fault injection capabilities (Internal short, degradation, cooling failure).
    """

    def __init__(
        self,
        cell_id: int,
        nominal_capacity_ah: float = 5.0,
        nominal_voltage_v: float = 3.7,
        initial_soc: float = 0.90,
        initial_soh: float = 1.0,
        initial_temp_c: float = 25.0,
        r0_ohm: float = 0.022,
        r1_ohm: float = 0.015,
        c1_farad: float = 2200.0,
        mass_kg: float = 0.068,
        ambient_temp_c: float = 25.0,
        ha_w_per_k: float = 0.18,
    ):
        self.cell_id = cell_id
        self.nominal_capacity_ah = nominal_capacity_ah
        self.nominal_voltage_v = nominal_voltage_v
        
        # State Variables
        self.soc = float(initial_soc)          # State of Charge (0.0 to 1.0)
        self.soh = float(initial_soh)          # State of Health (0.0 to 1.0)
        self.temp_c = float(initial_temp_c)    # Core temperature in Celsius
        self.v1 = 0.0                          # Polarization capacitor voltage across R1-C1
        self.terminal_voltage = 3.7            # Initialized in compute_terminal_voltage
        
        # Physical Parameters
        self.r0_base = float(r0_ohm)
        self.r1_base = float(r1_ohm)
        self.c1 = float(c1_farad)
        self.mass = float(mass_kg)
        self.cp = 920.0                        # J/(kg*K)
        self.ha = float(ha_w_per_k)            # Heat transfer coeff * area
        self.ambient_temp_c = float(ambient_temp_c)
        self.coulombic_eff = 0.995
        
        # Cumulative throughput for degradation
        self.cumulative_ah = 0.0
        
        # Fault injection flags
        self.internal_short_ohm = 0.0          # 0 = normal; >0 indicates short-circuit leakage
        self.thermal_fault_active = False      # Reduced cooling dissipation
        self.sensor_bias_v = 0.0               # Sensor calibration drift

    def open_circuit_voltage(self, soc: float) -> float:
        """
        NMC 21700 OCV-SOC polynomial characteristic curve.
        Maps SOC [0, 1] to OCV [~3.0V, ~4.2V].
        """
        s = np.clip(soc, 0.001, 0.999)
        # Accurate 6-term empirical representation of NMC open-circuit curve
        ocv = (
            3.435
            + 0.812 * s
            - 0.324 * (s ** 2)
            + 0.418 * (s ** 3)
            + 0.085 * np.log(s + 1e-4)
            - 0.045 * np.log(1.0 - s + 1e-4)
        )
        return float(np.clip(ocv, 2.95, 4.25))

    def get_effective_r0(self) -> float:
        """
        Computes dynamic internal resistance accounting for:
        1. Temperature dependency (Arrhenius-like exponential increase at low T).
        2. SOC dependency (higher resistance near empty).
        3. Degradation (SOH aging causes R0 increase).
        """
        # Temperature effect: reference temp 25 C (298.15 K)
        t_kelvin = self.temp_c + 273.15
        t_ref = 298.15
        arrhenius_factor = math.exp(2200.0 * (1.0 / t_kelvin - 1.0 / t_ref))
        
        # SOC effect: rises sharply below 10% SOC
        soc_factor = 1.0 + 0.35 * math.exp(-12.0 * max(self.soc, 0.01))
        
        # SOH aging: 100% SOH -> 1.0x; 80% SOH -> ~1.4x
        aging_factor = 1.0 + 2.0 * (1.0 - self.soh)
        
        return self.r0_base * arrhenius_factor * soc_factor * aging_factor

    def get_effective_r1(self) -> float:
        """Polarization resistance scaled with temperature."""
        t_kelvin = self.temp_c + 273.15
        t_ref = 298.15
        arrhenius_factor = math.exp(1800.0 * (1.0 / t_kelvin - 1.0 / t_ref))
        return self.r1_base * arrhenius_factor

    def step(self, current_a: float, dt: float = 1.0) -> dict:
        """
        Executes one physics integration step:
        - current_a > 0: Discharge (cell supplies current)
        - current_a < 0: Charge (current flows into cell)
        - dt: Time step in seconds
        """
        # 1. Account for internal short leakage if fault active
        leakage_current = 0.0
        if self.internal_short_ohm > 0.01:
            leakage_current = self.terminal_voltage / self.internal_short_ohm
        total_cell_current = current_a + leakage_current

        # 2. Update SOC via Coulomb Counting
        eff = self.coulombic_eff if total_cell_current < 0 else 1.0
        effective_capacity_ah = self.nominal_capacity_ah * self.soh
        delta_soc = -(total_cell_current * eff * (dt / 3600.0)) / effective_capacity_ah
        self.soc = float(np.clip(self.soc + delta_soc, 0.0, 1.0))
        
        # 3. Update polarization voltage across RC pair (Thevenin 1RC)
        # Continuous: dV1/dt = -V1 / (R1*C1) + I / C1
        # Discrete exact exponential integration:
        r1 = self.get_effective_r1()
        tau = r1 * self.c1
        exp_decay = math.exp(-dt / tau)
        self.v1 = float(self.v1 * exp_decay + total_cell_current * r1 * (1.0 - exp_decay))
        
        # 4. Compute Terminal Voltage
        ocv = self.open_circuit_voltage(self.soc)
        r0 = self.get_effective_r0()
        ir_drop = total_cell_current * r0
        self.terminal_voltage = float(ocv - ir_drop - self.v1 + self.sensor_bias_v)
        
        # 5. Thermal Dynamics (Joule Heating + Entropic Reaction + Cooling)
        # Joule heat = I^2 * R0 + V1^2 / R1
        # Internal short produces additional V^2 / R_sh heat
        q_joule = (total_cell_current ** 2) * r0 + (self.v1 ** 2) / max(r1, 1e-4)
        if self.internal_short_ohm > 0.01:
            q_joule += (self.terminal_voltage ** 2) / self.internal_short_ohm
            
        # Entropic reversible heat: I * T * (dVoc/dT)
        # dVoc/dT for NMC is approximately -0.25 mV/K at middle SOC
        d_voc_dt = -0.00025
        t_kelvin = self.temp_c + 273.15
        q_entropic = total_cell_current * t_kelvin * d_voc_dt
        
        # Total generated heat
        q_gen = max(0.0, q_joule + q_entropic)
        
        # Heat dissipation to ambient
        eff_ha = self.ha * 0.2 if self.thermal_fault_active else self.ha
        q_loss = eff_ha * (self.temp_c - self.ambient_temp_c)
        
        # Thermal derivative dT/dt = (Q_gen - Q_loss) / (m * Cp)
        dt_temp = (q_gen - q_loss) / (self.mass * self.cp)
        self.temp_c = float(self.temp_c + dt_temp * dt)
        
        # 6. Degradation Tracking (Capacity & SOH fade)
        self.cumulative_ah += abs(total_cell_current) * (dt / 3600.0)
        # Mild degradation equation based on throughput, elevated temperature, and high SOC
        temp_stress = max(1.0, math.exp(0.04 * max(0.0, self.temp_c - 35.0)))
        soc_stress = 1.0 + 0.5 * max(0.0, self.soc - 0.8)
        fade_rate = 1.2e-5 * temp_stress * soc_stress
        self.soh = float(max(0.60, 1.0 - fade_rate * math.sqrt(self.cumulative_ah + 1.0)))

        return {
            "cell_id": self.cell_id,
            "voltage": round(self.terminal_voltage, 4),
            "current": round(total_cell_current, 3),
            "temp_c": round(self.temp_c, 2),
            "soc": round(self.soc, 4),
            "soh": round(self.soh, 4),
            "v1": round(self.v1, 4),
            "ocv": round(ocv, 4),
            "r0_ohm": round(r0, 5),
            "heat_gen_w": round(q_gen, 2),
        }

    def inject_fault(self, fault_type: str, severity: float = 1.0):
        """
        Inject anomalies to test Digital Twin detection:
        - 'internal_short': introduces low resistance leakage path
        - 'high_resistance': increases R0 significantly (aged cell)
        - 'thermal_failure': reduces cooling efficiency
        - 'sensor_bias': offsets voltage sensor reading
        - 'clear': resets all faults
        """
        if fault_type == "internal_short":
            # 10 to 50 ohms causes significant leakage & heating
            self.internal_short_ohm = max(1.0, 50.0 / severity)
        elif fault_type == "high_resistance":
            self.r0_base *= (1.0 + 2.5 * severity)
        elif fault_type == "thermal_failure":
            self.thermal_fault_active = True
        elif fault_type == "sensor_bias":
            self.sensor_bias_v = 0.12 * severity
        elif fault_type == "clear":
            self.internal_short_ohm = 0.0
            self.thermal_fault_active = False
            self.sensor_bias_v = 0.0
