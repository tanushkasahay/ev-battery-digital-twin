"""
Standalone CLI Runner for the Simulated EV Battery IoT Edge Gateway.
"""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
from config import MQTT_CONFIG
from iot.bms_edge_node import BMSEdgeGateway


def main():
    parser = argparse.ArgumentParser(description="EV Battery BMS Embedded IoT Edge Node")
    parser.add_argument("--host", default=MQTT_CONFIG["broker_host"], help="MQTT Broker Host / HiveMQ Cluster URL")
    parser.add_argument("--port", type=int, default=MQTT_CONFIG["broker_port"], help="MQTT Broker Port (8883 for TLS, 1883 default)")
    parser.add_argument("--username", default=MQTT_CONFIG.get("username"), help="MQTT Username (Required for HiveMQ Cloud)")
    parser.add_argument("--password", default=MQTT_CONFIG.get("password"), help="MQTT Password (Required for HiveMQ Cloud)")
    parser.add_argument("--tls", action="store_true", help="Enable TLS (Required for HiveMQ Cloud on 8883)")
    args = parser.parse_args()

    use_tls = args.tls or (args.port == 8883)

    print("=" * 65)
    print("⚡ EV Battery BMS Embedded IoT Edge Node")
    print("=" * 65)
    print(f"Connecting to MQTT Broker: {args.host}:{args.port} (TLS: {use_tls})")
    print(f"Streaming on topic: {MQTT_CONFIG['topic_telemetry']}")
    print("Press Ctrl+C to terminate.\n")

    gateway = BMSEdgeGateway(
        broker_host=args.host,
        broker_port=args.port,
        username=args.username,
        password=args.password,
        use_tls=use_tls,
    )
    gateway.start()

    try:
        while True:
            time.sleep(1)
            telem = gateway.step_once()
            print(
                f"[Step {telem['time_step']:04d}] "
                f"V: {telem['pack_voltage']:.2f}V | "
                f"I: {telem['pack_current']:+5.1f}A | "
                f"P: {telem['power_kw']:+5.2f}kW | "
                f"True SOC: {telem['true_soc']*100:.1f}% | "
                f"BMS SOC: {telem['bms_coulomb_soc']*100:.1f}% | "
                f"Max T: {telem['max_temp']:.1f}°C | "
                f"Alarms: {telem['alarms']}"
            )
    except KeyboardInterrupt:
        print("\nStopping BMS Edge Node...")
        gateway.stop()
        print("Done.")


if __name__ == "__main__":
    main()
