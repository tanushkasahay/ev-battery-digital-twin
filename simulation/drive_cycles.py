"""
Standardized and Dynamic EV Drive Cycles for Battery Telemetry Generation.
Includes WLTP profile, Urban Stop-and-Go, Highway, Aggressive Sport, and Fast Charge.
"""
import math
import numpy as np


class DriveCycleGenerator:
    """
    Generates realistic EV pack load current profiles (in Amperes).
    Positive current (>0) = Discharge (driving).
    Negative current (<0) = Charge (regenerative braking or plug-in charging).
    """

    @staticmethod
    def get_current(cycle_type: str, time_step: int, dt: float = 1.0) -> float:
        """
        Returns load current for current time_step (seconds).
        """
        t = float(time_step * dt)

        if cycle_type.lower() == "wltp":
            # WLTP cycle simulation: combines low, medium, high, and extra-high phases
            # Period of 1800 seconds, synthesized using multi-harmonic velocity-derivative profile
            base_surge = (
                8.0 * math.sin(2 * math.pi * t / 45.0)
                + 12.0 * math.sin(2 * math.pi * t / 130.0)
                + 6.0 * math.sin(2 * math.pi * t / 25.0)
            )
            # Add acceleration transients and regenerative braking
            noise = np.random.normal(0, 1.2)
            current = 5.0 + base_surge + noise
            # Occasional stop & idle (traffic lights)
            if (t % 120) < 15:
                current = 0.0
            # Occasional strong regen braking
            elif (t % 70) > 60:
                current = -8.0 - abs(noise)
            return float(np.clip(current, -18.0, 32.0))

        elif cycle_type.lower() == "urban":
            # Urban Stop-and-Go: lots of 0 current, moderate accelerations, quick regen
            phase = t % 60
            if phase < 15:
                # Stopped at red light / idle
                return 0.2  # tiny parasitic draw (auxiliaries/AC)
            elif phase < 25:
                # Moderate acceleration
                return float(12.0 + np.random.normal(0, 1.5))
            elif phase < 45:
                # Constant speed cruising
                return float(4.5 + np.random.normal(0, 0.5))
            else:
                # Braking into regen
                return float(-7.0 + np.random.normal(0, 0.8))

        elif cycle_type.lower() == "highway":
            # Steady high discharge with slight wind/incline variations
            base_draw = 14.0 + 3.0 * math.sin(2 * math.pi * t / 90.0)
            overtake = 8.0 if (t % 80 > 65) else 0.0
            return float(base_draw + overtake + np.random.normal(0, 0.6))

        elif cycle_type.lower() == "aggressive_sport":
            # Aggressive throttle punches up to 35A, aggressive regen up to -20A
            cycle = t % 30
            if cycle < 8:
                return float(28.0 + np.random.normal(0, 3.0))
            elif cycle < 15:
                return float(10.0 + np.random.normal(0, 1.0))
            elif cycle < 22:
                return float(-16.0 + np.random.normal(0, 2.0))  # Heavy regen
            else:
                return float(32.0 + np.random.normal(0, 2.5))

        elif cycle_type.lower() == "fast_charge":
            # High-rate DC fast charging: -15A CC charging
            return float(-15.0 + np.random.normal(0, 0.2))

        elif cycle_type.lower() == "idle":
            # No load, measuring open-circuit relaxation
            return 0.0

        else:
            # Default fallback: gentle cruising
            return float(6.0 + np.random.normal(0, 0.5))
