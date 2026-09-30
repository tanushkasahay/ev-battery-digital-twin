"""
Standalone CLI Runner for the Battery Digital Twin Cloud Service.
"""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
from config import MQTT_CONFIG
from digital_twin.twin_service import BatteryDigitalTwinService


def main():
    parser = argparse.ArgumentParser(description="EV Battery Pack AI Digital Twin Service")
    parser.add_argument("--host", default=MQTT_CONFIG["broker_host"], help="MQTT Broker Host / HiveMQ Cluster URL")
    parser.add_argument("--port", type=int, default=MQTT_CONFIG["broker_port"], help="MQTT Broker Port (8883 for TLS, 1883 default)")
    parser.add_argument("--username", default=MQTT_CONFIG.get("username"), help="MQTT Username (Required for HiveMQ Cloud)")
    parser.add_argument("--password", default=MQTT_CONFIG.get("password"), help="MQTT Password (Required for HiveMQ Cloud)")
    parser.add_argument("--tls", action="store_true", help="Enable TLS (Required for HiveMQ Cloud on 8883)")
    args = parser.parse_args()

    use_tls = args.tls or (args.port == 8883)

    print("=" * 65)
    print("🔋 EV Battery Pack AI Digital Twin Service")
    print("=" * 65)
    print(f"Connecting to MQTT Broker: {args.host}:{args.port} (TLS: {use_tls})")
    print(f"Subscribed to topic: {MQTT_CONFIG['topic_telemetry']}")
    print(f"Publishing AI estimates to: {MQTT_CONFIG['topic_twin_estimate']}")
    print("Press Ctrl+C to terminate.\n")

    twin = BatteryDigitalTwinService(
        broker_host=args.host,
        broker_port=args.port,
        username=args.username,
        password=args.password,
        use_tls=use_tls,
    )
    twin.start()

    try:
        while True:
            time.sleep(1)
            snap = twin.get_snapshot()
            telem = snap["telemetry"]
            ai_est = snap["ai_estimate"]
            diag = snap["anomaly_report"]

            if telem and ai_est:
                coulomb_soc = telem.get("bms_coulomb_soc", 0.0) * 100.0
                ai_soc = ai_est.get("ai_soc", 0.0) * 100.0
                true_soc = telem.get("true_soc", 0.0) * 100.0
                ai_soh = ai_est.get("ai_soh", 1.0) * 100.0
                status = diag.get("status", "NORMAL") if diag else "NORMAL"

                print(
                    f"[Packets: {snap['packets_received']:04d}] "
                    f"True SOC: {true_soc:.1f}% | "
                    f"BMS Coulomb: {coulomb_soc:.1f}% | "
                    f"AI Twin SOC: {ai_soc:.1f}% | "
                    f"AI SOH: {ai_soh:.1f}% | "
                    f"Health: [{status}]"
                )
    except KeyboardInterrupt:
        print("\nStopping Digital Twin Service...")
        twin.stop()
        print("Done.")


if __name__ == "__main__":
    main()
