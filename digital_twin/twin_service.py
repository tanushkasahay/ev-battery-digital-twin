"""
Digital Twin Cloud Service.
Subscribes to battery MQTT telemetry, performs real-time PyTorch AI estimation,
monitors pack health/anomalies, and publishes synchronized twin states.
"""
import os
import sys
import time
import logging
from collections import deque
import threading

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import MQTT_CONFIG
from iot.mqtt_client import BatteryMQTTClient
from ai.estimator import BatteryAIEstimator
from digital_twin.anomaly_detector import BatteryAnomalyDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DigitalTwinService")


class BatteryDigitalTwinService:
    """
    Central Digital Twin Engine:
    - Maintains the synchronized digital shadow of the physical EV battery pack.
    - Executes AI-based state estimation (SoC, SoH, RUL).
    - Detects impending thermal runaway or electrical faults.
    - Publishes twin updates and triggers automated safety interlocks.
    """

    def __init__(
        self,
        broker_host: str = None,
        broker_port: int = None,
        username: str = None,
        password: str = None,
        use_tls: bool = False,
        history_len: int = 150,
    ):
        self.broker_host = broker_host or MQTT_CONFIG["broker_host"]
        self.broker_port = broker_port or MQTT_CONFIG["broker_port"]
        self.username = username or MQTT_CONFIG.get("username")
        self.password = password or MQTT_CONFIG.get("password")
        self.use_tls = use_tls or MQTT_CONFIG.get("use_tls", False)
        
        self.ai_estimator = BatteryAIEstimator()
        self.anomaly_detector = BatteryAnomalyDetector()

        self.client = BatteryMQTTClient(
            client_id=f"{MQTT_CONFIG['client_id_twin']}_{int(time.time()) % 10000}",
            broker_host=self.broker_host,
            broker_port=self.broker_port,
            username=self.username,
            password=self.password,
            use_tls=self.use_tls,
        )

        # Thread-safe in-memory state store
        self.lock = threading.Lock()
        self.latest_telemetry = None
        self.latest_ai_estimate = None
        self.latest_anomaly_report = None
        self.telemetry_history = deque(maxlen=history_len)
        self.ai_history = deque(maxlen=history_len)
        
        self.packets_received = 0
        self.last_update_time = None
        self.running = False

    def start(self):
        """Starts the Digital Twin MQTT listener."""
        logger.info("Starting EV Battery Digital Twin Service...")
        connected = self.client.connect(timeout=5)
        if not connected:
            logger.warning(f"MQTT Broker at {self.broker_host}:{self.broker_port} unavailable. Running in local/direct mode.")

        # Subscribe to physical telemetry
        self.client.subscribe(MQTT_CONFIG["topic_telemetry"], self.on_telemetry_received)
        self.running = True
        logger.info(f"Digital Twin subscribed to '{MQTT_CONFIG['topic_telemetry']}'")

    def on_telemetry_received(self, telemetry: dict):
        """Processes incoming real-time telemetry from the BMS edge gateway."""
        now = time.time()
        self.packets_received += 1
        self.last_update_time = now

        # 1. Run AI Inference (SoC & SoH Estimation)
        ai_res = self.ai_estimator.update(telemetry)

        # 2. Run Anomaly & Health Diagnostics
        diag_res = self.anomaly_detector.analyze(telemetry, ai_res)

        # 3. Formulate Digital Twin Sync Payload
        twin_payload = {
            "timestamp": now,
            "pack_id": telemetry.get("pack_id", "UNKNOWN"),
            "ai_soc": ai_res["ai_soc"],
            "ai_soh": ai_res["ai_soh"],
            "estimated_range_km": ai_res["estimated_range_km"],
            "confidence_score": ai_res["confidence_score"],
            "bms_coulomb_soc": telemetry.get("bms_coulomb_soc"),
            "true_soc": telemetry.get("true_soc"),
            "health_status": diag_res["status"],
            "anomalies": diag_res["anomalies"],
            "packets_processed": self.packets_received,
        }

        # 4. Thread-safe internal snapshot storage
        with self.lock:
            self.latest_telemetry = telemetry
            self.latest_ai_estimate = ai_res
            self.latest_anomaly_report = diag_res
            self.telemetry_history.append(telemetry)
            self.ai_history.append(twin_payload)

        # 5. Broadcast Digital Twin state to cloud subscribers
        self.client.publish(MQTT_CONFIG["topic_twin_estimate"], twin_payload)

        # 6. Automated Digital Twin Closed-Loop Safety Control
        if diag_res["status"] == "CRITICAL":
            logger.critical("Digital Twin detected CRITICAL safety condition! Dispatching Emergency Contactor Trip to BMS.")
            self.send_control_command({"command": "emergency_stop", "reason": "DIGITAL_TWIN_AUTO_PROTECTION"})

    def send_control_command(self, command_dict: dict):
        """Sends remote commands back to the BMS edge gateway."""
        self.client.publish(MQTT_CONFIG["topic_control"], command_dict)

    def get_snapshot(self) -> dict:
        """Returns the latest thread-safe snapshot of the Digital Twin."""
        with self.lock:
            return {
                "telemetry": self.latest_telemetry,
                "ai_estimate": self.latest_ai_estimate,
                "anomaly_report": self.latest_anomaly_report,
                "packets_received": self.packets_received,
                "last_update_time": self.last_update_time,
                "telemetry_history": list(self.telemetry_history),
                "ai_history": list(self.ai_history),
            }

    def stop(self):
        """Stops the Digital Twin service."""
        self.running = False
        self.client.disconnect()
        logger.info("Digital Twin Service stopped.")
