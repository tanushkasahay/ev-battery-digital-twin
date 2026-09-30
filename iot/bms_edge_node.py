"""
Embedded IoT BMS Edge Node Emulator.
Simulates a physical Telematics Control Unit (TCU) on the EV Battery Pack.
Publishes real-time sensor telemetry to MQTT and listens for remote control commands.
"""
import time
import threading
import logging
from config import MQTT_CONFIG, PACK_CONFIG
from simulation.pack_model import BatteryPack
from simulation.drive_cycles import DriveCycleGenerator
from iot.mqtt_client import BatteryMQTTClient

logger = logging.getLogger("BMSEdgeNode")


class BMSEdgeGateway:
    """
    Embedded IoT Gateway running alongside the physical BMS hardware.
    Periodically samples cell sensors, detects electrical thresholds,
    and publishes structured JSON telemetry to the cloud MQTT broker.
    """

    def __init__(
        self,
        broker_host: str = None,
        broker_port: int = None,
        username: str = None,
        password: str = None,
        use_tls: bool = False,
    ):
        self.pack = BatteryPack()
        self.broker_host = broker_host or MQTT_CONFIG["broker_host"]
        self.broker_port = broker_port or MQTT_CONFIG["broker_port"]
        self.username = username or MQTT_CONFIG.get("username")
        self.password = password or MQTT_CONFIG.get("password")
        self.use_tls = use_tls or MQTT_CONFIG.get("use_tls", False)
        
        self.client = BatteryMQTTClient(
            client_id=f"{MQTT_CONFIG['client_id_edge']}_{int(time.time()) % 10000}",
            broker_host=self.broker_host,
            broker_port=self.broker_port,
            username=self.username,
            password=self.password,
            use_tls=self.use_tls,
        )
        
        self.running = False
        self.active_cycle = "wltp"
        self.time_step = 0
        self.dt = MQTT_CONFIG["telemetry_interval_sec"]
        self.emergency_stop = False
        self._thread = None

    def start(self):
        """Starts the IoT Edge Node and background streaming thread."""
        logger.info("Initializing BMS Embedded IoT Edge Node...")
        connected = self.client.connect(timeout=5)
        if not connected:
            logger.warning(f"MQTT Broker at {self.broker_host}:{self.broker_port} unavailable. Telemetry will run locally.")

        # Subscribe to cloud commands
        self.client.subscribe(MQTT_CONFIG["topic_control"], self._on_control_command)

        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(f"BMS Edge Node started! Streaming on topic '{MQTT_CONFIG['topic_telemetry']}' at {1/self.dt:.1f} Hz")

    def _on_control_command(self, data: dict):
        """Processes remote commands from the Digital Twin or User Dashboard."""
        cmd = data.get("command")
        logger.info(f"Received Cloud Control Command: {cmd} with payload: {data}")

        if cmd == "set_cycle":
            self.active_cycle = data.get("cycle", "wltp")
            logger.info(f"Drive cycle updated to: {self.active_cycle}")
        elif cmd == "inject_fault":
            cell_idx = int(data.get("cell_id", 1)) - 1
            fault_type = data.get("fault_type", "internal_short")
            severity = float(data.get("severity", 1.0))
            self.pack.inject_cell_fault(cell_idx, fault_type, severity)
            logger.warning(f"Injected fault '{fault_type}' into Cell #{cell_idx + 1} with severity {severity}")
        elif cmd == "clear_faults":
            self.pack.reset_all_faults()
            self.emergency_stop = False
            logger.info("Cleared all cell faults and reset emergency interlocks.")
        elif cmd == "emergency_stop":
            self.emergency_stop = True
            logger.critical("EMERGENCY STOP COMMAND RECEIVED! Contactors open, current set to 0A.")

    def step_once(self) -> dict:
        """Executes a single simulation step and generates telemetry payload."""
        if self.emergency_stop:
            current_a = 0.0
        else:
            current_a = DriveCycleGenerator.get_current(self.active_cycle, self.time_step, self.dt)

        res = self.pack.step(current_a, dt=self.dt)
        self.time_step += 1

        payload = {
            "timestamp": time.time(),
            "time_step": self.time_step,
            "cycle_profile": self.active_cycle,
            "pack_id": res["pack_id"],
            "pack_voltage": res["pack_voltage"],
            "pack_current": res["pack_current"],
            "measured_current": res["measured_current"],
            "power_kw": res["power_kw"],
            "true_soc": res["true_soc"],           # Ground truth (hidden from vehicle in real life)
            "bms_coulomb_soc": res["bms_coulomb_soc"], # Traditional BMS estimate (suffers drift)
            "true_soh": res["true_soh"],
            "cell_voltages": res["cell_voltages"],
            "cell_temps": res["cell_temps"],
            "cell_socs": res["cell_socs"],
            "max_cell_v": res["max_cell_voltage"],
            "min_cell_v": res["min_cell_voltage"],
            "v_delta": res["delta_cell_voltage"],
            "avg_temp": res["avg_cell_temp"],
            "max_temp": res["max_cell_temp"],
            "min_temp": res["min_cell_temp"],
            "balancing_active": res["balancing_active"],
            "alarms": res["alarms"],
            "emergency_stop": self.emergency_stop,
        }
        return payload

    def _loop(self):
        """Continuous sampling and transmission loop."""
        while self.running:
            start_t = time.time()
            telemetry = self.step_once()
            
            # Publish via MQTT
            self.client.publish(MQTT_CONFIG["topic_telemetry"], telemetry)

            elapsed = time.time() - start_t
            sleep_time = max(0.01, self.dt - elapsed)
            time.sleep(sleep_time)

    def stop(self):
        """Stops the edge node."""
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.client.disconnect()
        logger.info("BMS Edge Node stopped.")
