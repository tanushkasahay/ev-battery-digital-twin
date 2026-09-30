"""
Resilient MQTT Publisher and Subscriber Wrapper supporting Paho-MQTT v2 and v1.
Includes auto-reconnect, JSON serializing, and thread-safe callbacks.
"""
import json
import logging
import ssl
import time
import paho.mqtt.client as mqtt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MQTTClient")


class BatteryMQTTClient:
    """
    Robust MQTT Client handling both telemetry publishing and digital twin subscriptions.
    Supports local unencrypted brokers as well as HiveMQ Cloud Serverless (TLS + Auth).
    """

    def __init__(
        self,
        client_id: str,
        broker_host: str = "127.0.0.1",
        broker_port: int = 1883,
        username: str = None,
        password: str = None,
        use_tls: bool = False,
    ):
        self.client_id = client_id
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.username = username
        self.password = password
        # Automatically enable TLS if port is 8883 or if explicitly requested
        self.use_tls = use_tls or (broker_port == 8883)
        self.connected = False
        self.subscriptions = {}  # topic -> callback function

        # Initialize Paho MQTT Client (supports both v1.x and v2.x callback API)
        try:
            # Paho MQTT 2.0+ API
            self.client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id=self.client_id,
                clean_session=True,
            )
        except AttributeError:
            # Fallback to older Paho MQTT 1.x
            self.client = mqtt.Client(client_id=self.client_id, clean_session=True)

        # Configure Authentication for HiveMQ Cloud Serverless
        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)

        # Configure TLS / SSL for HiveMQ Cloud Serverless (Port 8883)
        if self.use_tls:
            context = ssl.create_default_context()
            self.client.tls_set_context(context)
            logger.info("Configured TLS/SSL encryption for MQTT connection.")

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            self.connected = True
            logger.info(f"Connected to MQTT Broker [{self.broker_host}:{self.broker_port}] as '{self.client_id}'")
            # Resubscribe to existing registered topics
            for topic in self.subscriptions:
                self.client.subscribe(topic, qos=0)
                logger.info(f"Subscribed to topic: {topic}")
        else:
            logger.error(f"Failed to connect to MQTT broker, return code: {rc}")

    def _on_disconnect(self, client, userdata, rc=None, properties=None, reason_code=None):
        self.connected = False
        logger.warning(f"Disconnected from MQTT Broker (code: {rc or reason_code})")

    def _on_message(self, client, userdata, msg):
        try:
            payload_str = msg.payload.decode("utf-8")
            data = json.loads(payload_str)
            if msg.topic in self.subscriptions:
                self.subscriptions[msg.topic](data)
        except Exception as e:
            logger.error(f"Error handling message on {msg.topic}: {e}")

    def connect(self, timeout: int = 10) -> bool:
        """Connects to the broker and starts the background network loop."""
        try:
            logger.info(f"Connecting to MQTT broker at {self.broker_host}:{self.broker_port}...")
            self.client.connect(self.broker_host, self.broker_port, keepalive=60)
            self.client.loop_start()
            
            # Wait for connection confirmation
            start_t = time.time()
            while not self.connected and (time.time() - start_t) < timeout:
                time.sleep(0.1)
                
            return self.connected
        except Exception as e:
            logger.error(f"Exception connecting to broker: {e}")
            return False

    def subscribe(self, topic: str, callback):
        """Register a topic and its handler callback."""
        self.subscriptions[topic] = callback
        if self.connected:
            self.client.subscribe(topic, qos=0)
            logger.info(f"Subscribed to {topic}")

    def publish(self, topic: str, data: dict, qos: int = 0) -> bool:
        """Publishes a Python dictionary as a serialized JSON message."""
        if not self.connected:
            return False
        try:
            payload = json.dumps(data)
            res = self.client.publish(topic, payload, qos=qos)
            return res.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            logger.error(f"Failed to publish to {topic}: {e}")
            return False

    def disconnect(self):
        """Gracefully disconnects client and stops loop."""
        try:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            logger.info(f"MQTT Client '{self.client_id}' disconnected.")
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
