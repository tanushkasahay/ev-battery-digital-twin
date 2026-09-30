"""
Lightweight Zero-Dependency Pure-Python MQTT 3.1.1 Broker for Offline/Local Testing.
Enables running the full IoT Digital Twin system without external Mosquitto or internet access.
"""
import socket
import threading
import logging

logger = logging.getLogger("LocalMQTTBroker")


class LocalMQTTBroker:
    """
    Lightweight embedded MQTT broker supporting CONNECT, PUBLISH (QoS 0), SUBSCRIBE, and PING.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 1883):
        self.host = host
        self.port = port
        self.clients = []  # list of (socket, subscribed_topics)
        self.running = False
        self.server_sock = None
        self.lock = threading.Lock()

    def start(self):
        try:
            self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_sock.bind((self.host, self.port))
            self.server_sock.listen(15)
            self.running = True
            logger.info(f"Local MQTT Broker started on {self.host}:{self.port}")

            threading.Thread(target=self._accept_loop, daemon=True).start()
            return True
        except Exception as e:
            logger.warning(f"Could not bind Local MQTT Broker on port {self.port}: {e}")
            return False

    def _accept_loop(self):
        while self.running:
            try:
                client_sock, addr = self.server_sock.accept()
                threading.Thread(target=self._client_handler, args=(client_sock,), daemon=True).start()
            except Exception:
                break

    def _client_handler(self, sock):
        client_info = {"sock": sock, "subs": set()}
        with self.lock:
            self.clients.append(client_info)

        buffer = bytearray()
        try:
            while self.running:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buffer.extend(chunk)

                while len(buffer) >= 2:
                    pkt_type = buffer[0] >> 4
                    # Decode remaining length (variable length)
                    multiplier = 1
                    value = 0
                    idx = 1
                    while True:
                        if idx >= len(buffer):
                            break
                        encoded_byte = buffer[idx]
                        value += (encoded_byte & 127) * multiplier
                        multiplier *= 128
                        idx += 1
                        if (encoded_byte & 128) == 0:
                            break

                    total_len = idx + value
                    if len(buffer) < total_len:
                        break  # Packet incomplete, wait for more bytes

                    packet = buffer[:total_len]
                    buffer = buffer[total_len:]

                    self._handle_packet(client_info, pkt_type, packet[idx:])
        except Exception:
            pass
        finally:
            with self.lock:
                if client_info in self.clients:
                    self.clients.remove(client_info)
            try:
                sock.close()
            except Exception:
                pass

    def _handle_packet(self, client_info, pkt_type, payload):
        sock = client_info["sock"]
        # CONNECT (1) -> send CONNACK (2)
        if pkt_type == 1:
            sock.sendall(bytes([0x20, 0x02, 0x00, 0x00]))
        # PUBLISH (3) -> route to matching subscribers
        elif pkt_type == 3:
            if len(payload) >= 2:
                topic_len = (payload[0] << 8) | payload[1]
                topic = payload[2 : 2 + topic_len].decode("utf-8", errors="ignore")
                msg_bytes = payload[2 + topic_len :]

                # Forward to subscribed clients
                self._broadcast(topic, msg_bytes)
        # SUBSCRIBE (8) -> send SUBACK (9) and register topic
        elif pkt_type == 8:
            if len(payload) >= 2:
                msg_id = payload[:2]
                idx = 2
                while idx < len(payload):
                    t_len = (payload[idx] << 8) | payload[idx + 1]
                    idx += 2
                    topic = payload[idx : idx + t_len].decode("utf-8", errors="ignore")
                    idx += t_len + 1  # include requested QoS byte
                    client_info["subs"].add(topic)
                # SUBACK with granted QoS 0
                sock.sendall(bytes([0x90, 0x03, msg_id[0], msg_id[1], 0x00]))
        # PINGREQ (12) -> send PINGRESP (13)
        elif pkt_type == 12:
            sock.sendall(bytes([0xD0, 0x00]))
        # DISCONNECT (14)
        elif pkt_type == 14:
            sock.close()

    def _broadcast(self, topic: str, msg_bytes: bytes):
        topic_encoded = topic.encode("utf-8")
        topic_len = len(topic_encoded)
        rem_len = 2 + topic_len + len(msg_bytes)

        # Build PUBLISH packet (QoS 0)
        pkt = bytearray([0x30])
        # Variable length encoding for remaining length
        val = rem_len
        while True:
            byte = val % 128
            val = val // 128
            if val > 0:
                byte |= 128
            pkt.append(byte)
            if val == 0:
                break
        pkt.extend([(topic_len >> 8) & 0xFF, topic_len & 0xFF])
        pkt.extend(topic_encoded)
        pkt.extend(msg_bytes)

        with self.lock:
            for c in list(self.clients):
                # Simple wildcard matching or direct matching
                for sub in c["subs"]:
                    if sub == topic or sub == "#" or (sub.endswith("/#") and topic.startswith(sub[:-2])):
                        try:
                            c["sock"].sendall(pkt)
                        except Exception:
                            pass
                        break

    def stop(self):
        self.running = False
        if self.server_sock:
            try:
                self.server_sock.close()
            except Exception:
                pass


if __name__ == "__main__":
    import time
    broker = LocalMQTTBroker(port=1883)
    if broker.start():
        print("Local broker running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            broker.stop()
