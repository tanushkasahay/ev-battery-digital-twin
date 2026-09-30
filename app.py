"""
Real-Time Streamlit Interactive Dashboard for EV Battery Pack Digital Twin.
Visualizes multi-cell telemetry, PyTorch AI estimations, thermal heatmaps, and MQTT streaming.
"""
import os
import sys
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Configure sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import PACK_CONFIG, MQTT_CONFIG
from iot.bms_edge_node import BMSEdgeGateway
from digital_twin.twin_service import BatteryDigitalTwinService

st.set_page_config(
    page_title="EV Battery Digital Twin | AI & IoT",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for high-tech dark automotive telematics aesthetic
st.markdown(
    """
    <style>
    .metric-card {
        background: #131722;
        border: 1px solid #2a2e39;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 13px;
        color: #787b86;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #d1d4dc;
    }
    .metric-sub {
        font-size: 12px;
        color: #2962ff;
    }
    .cell-box {
        border-radius: 6px;
        padding: 8px;
        text-align: center;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_system_instances(broker_host, broker_port, username=None, password=None, use_tls=False):
    """Initializes and caches the IoT Edge Gateway and Digital Twin service."""
    gateway = BMSEdgeGateway(
        broker_host=broker_host,
        broker_port=broker_port,
        username=username,
        password=password,
        use_tls=use_tls,
    )
    twin = BatteryDigitalTwinService(
        broker_host=broker_host,
        broker_port=broker_port,
        username=username,
        password=password,
        use_tls=use_tls,
    )
    
    # Start both services
    gateway.start()
    twin.start()
    return gateway, twin


# ==================== SIDEBAR CONTROLS ====================
st.sidebar.title("⚡ EV Pack Telematics")
st.sidebar.caption("Embedded IoT + PyTorch Digital Twin")

broker_choice = st.sidebar.selectbox(
    "MQTT Broker Destination",
    options=[
        "broker.hivemq.com (Public HiveMQ - 1883)",
        "HiveMQ Cloud (Free #1 Serverless - TLS 8883)",
        "127.0.0.1 (Local Mosquitto / Offline)",
        "broker.emqx.io (Cloud Public - 1883)",
    ],
    index=0,
)

# Extract broker credentials & TLS parameters
broker_user = None
broker_pass = None
use_tls = False

if "HiveMQ Cloud" in broker_choice:
    st.sidebar.markdown("☁️ **HiveMQ Cloud Free Serverless Settings**")
    default_cluster = MQTT_CONFIG["hivemq_serverless"]["cluster_url"] or ""
    broker_host = st.sidebar.text_input(
        "Cluster Host URL",
        value=default_cluster,
        placeholder="e.g. your-cluster-id.s1.eu.hivemq.cloud",
        help="Found in your HiveMQ Cloud Console Overview page.",
    )
    broker_port = 8883
    use_tls = True
    broker_user = st.sidebar.text_input("MQTT Username", value=MQTT_CONFIG.get("username") or "", placeholder="e.g. bms_iot_client")
    broker_pass = st.sidebar.text_input("MQTT Password", value=MQTT_CONFIG.get("password") or "", type="password")
    if not broker_host or not broker_user:
        st.sidebar.warning("⚠️ Enter your HiveMQ Cloud cluster URL and credentials to stream.")
elif "broker.hivemq.com" in broker_choice:
    broker_host = "broker.hivemq.com"
    broker_port = 1883
    use_tls = False
elif "emqx" in broker_choice:
    broker_host = "broker.emqx.io"
    broker_port = 1883
    use_tls = False
else:
    broker_host = "127.0.0.1"
    broker_port = 1883
    use_tls = False

gateway, twin = get_system_instances(
    broker_host=broker_host,
    broker_port=broker_port,
    username=broker_user,
    password=broker_pass,
    use_tls=use_tls,
)

st.sidebar.divider()
st.sidebar.subheader("🚗 Dynamic Drive Cycle")
selected_cycle = st.sidebar.selectbox(
    "Active Profile",
    options=["WLTP", "Urban", "Highway", "Aggressive_Sport", "Fast_Charge", "Idle"],
    index=0,
)

if st.sidebar.button("Apply Drive Cycle", use_container_width=True):
    twin.send_control_command({"command": "set_cycle", "cycle": selected_cycle.lower()})
    st.sidebar.success(f"Dispatched cycle: {selected_cycle}")

st.sidebar.divider()
st.sidebar.subheader("⚠️ Fault Injection Testbed")
st.sidebar.caption("Simulate hardware anomalies to test Digital Twin AI & safety alarms")

target_cell = st.sidebar.slider("Target Cell #", min_value=1, max_value=16, value=5)
fault_kind = st.sidebar.selectbox(
    "Anomaly Type",
    options=[
        "Internal Short Circuit (Leakage)",
        "Thermal Runaway / Cooling Loss",
        "Aged Cell (High Internal Resistance)",
        "Voltage Sensor Calibration Bias",
    ],
)

col_f1, col_f2 = st.sidebar.columns(2)
with col_f1:
    if st.button("Inject Fault", type="primary", use_container_width=True):
        kind_map = {
            "Internal Short Circuit (Leakage)": "internal_short",
            "Thermal Runaway / Cooling Loss": "thermal_failure",
            "Aged Cell (High Internal Resistance)": "high_resistance",
            "Voltage Sensor Calibration Bias": "sensor_bias",
        }
        twin.send_control_command({
            "command": "inject_fault",
            "cell_id": target_cell,
            "fault_type": kind_map[fault_kind],
            "severity": 1.5,
        })
        st.sidebar.warning(f"Injected {fault_kind} on Cell {target_cell}")

with col_f2:
    if st.button("Reset All", use_container_width=True):
        twin.send_control_command({"command": "clear_faults"})
        st.sidebar.info("All cell faults cleared.")

if st.sidebar.button("🚨 Emergency Pack Contactor Trip", use_container_width=True):
    twin.send_control_command({"command": "emergency_stop"})
    st.sidebar.error("Emergency stop command dispatched!")

st.sidebar.divider()
st.sidebar.info(
    f"**Pack Specs:** {PACK_CONFIG['series_cells']}S2P Li-ion NMC\n\n"
    f"**Nominal:** 59.2V | 10 Ah | 0.59 kWh\n\n"
    f"**Telemetry Topic:** `{MQTT_CONFIG['topic_telemetry']}`\n\n"
    f"**Twin Topic:** `{MQTT_CONFIG['topic_twin_estimate']}`"
)


# ==================== MAIN DASHBOARD ====================
st.title("🔋 EV Battery Pack — AI Digital Twin & Telematics")
st.caption("Real-Time Synchronization between Simulated Physical Pack (IoT Edge Node) and Deep Learning Digital Twin (PyTorch)")

snapshot = twin.get_snapshot()
telem = snapshot["telemetry"]
ai_est = snapshot["ai_estimate"]
diag = snapshot["anomaly_report"]

if not telem:
    st.info("📡 Waiting for incoming telemetry packets from BMS Edge Gateway via MQTT...")
    time.sleep(1)
    st.rerun()

# Extract real-time variables
pack_v = telem.get("pack_voltage", 0.0)
pack_i = telem.get("pack_current", 0.0)
p_kw = telem.get("power_kw", 0.0)
true_soc = telem.get("true_soc", 0.0) * 100.0
coulomb_soc = telem.get("bms_coulomb_soc", 0.0) * 100.0
ai_soc = (ai_est.get("ai_soc", 0.0) if ai_est else telem.get("bms_coulomb_soc", 0.0)) * 100.0
ai_soh = (ai_est.get("ai_soh", 1.0) if ai_est else 1.0) * 100.0
est_range = ai_est.get("estimated_range_km", 0.0) if ai_est else 0.0
confidence = (ai_est.get("confidence_score", 0.5) if ai_est else 0.5) * 100.0
avg_temp = telem.get("avg_temp", 25.0)
max_temp = telem.get("max_temp", 25.0)
v_delta = telem.get("v_delta", 0.0) * 1000.0  # in mV
health_status = diag["status"] if diag else "NORMAL"
anomalies = diag["anomalies"] if diag else []

# Status Banner
if health_status == "CRITICAL":
    st.error(f"🚨 **DIGITAL TWIN STATUS: CRITICAL SAFETY ALARM** — {len(anomalies)} anomalies detected!")
elif health_status == "WARNING":
    st.warning(f"⚠️ **DIGITAL TWIN STATUS: ATTENTION REQUIRED** — {len(anomalies)} advisory warnings active.")
else:
    st.success("✅ **DIGITAL TWIN STATUS: NOMINAL** — Battery pack operating within optimal safety margins.")

# Top KPIs Row
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Pack Voltage & Current</div>
            <div class="metric-value">{pack_v:.2f} <span style="font-size:16px;">V</span></div>
            <div class="metric-sub">Current: {pack_i:+.1f} A ({p_kw:+.2f} kW)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi2:
    soc_diff = ai_soc - coulomb_soc
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">AI Digital Twin SOC</div>
            <div class="metric-value">{ai_soc:.1f} <span style="font-size:16px;">%</span></div>
            <div class="metric-sub">Coulomb: {coulomb_soc:.1f}% (Δ {soc_diff:+.1f}%)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">State of Health (SOH)</div>
            <div class="metric-value">{ai_soh:.1f} <span style="font-size:16px;">%</span></div>
            <div class="metric-sub">Range: ~{est_range:.1f} km</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Max Cell Temperature</div>
            <div class="metric-value">{max_temp:.1f} <span style="font-size:16px;">°C</span></div>
            <div class="metric-sub">Pack Avg: {avg_temp:.1f} °C</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Cell Imbalance (ΔV)</div>
            <div class="metric-value">{v_delta:.1f} <span style="font-size:16px;">mV</span></div>
            <div class="metric-sub">AI Confidence: {confidence:.0f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Active Anomalies Drawer
if anomalies:
    with st.expander(f"⚠️ Active Diagnostic Insights ({len(anomalies)})", expanded=True):
        for item in anomalies:
            badge_color = "red" if item["severity"] == "CRITICAL" else "orange"
            st.markdown(
                f"- <span style='color:{badge_color}; font-weight:bold;'>[{item['severity']}]</span> "
                f"**{item['type']}**: {item['message']}",
                unsafe_allow_html=True,
            )

st.markdown("---")

# ==================== CELL MATRIX TOPOLOGY ====================
st.subheader("🧩 Multi-Cell Pack Topology (16 Series Cells)")
st.caption("Dynamic individual cell telemetry: voltage, temperature, and passive balancing shunt state.")

cell_vs = telem.get("cell_voltages", [3.7] * 16)
cell_ts = telem.get("cell_temps", [25.0] * 16)
cell_soc_list = telem.get("cell_socs", [0.8] * 16)
balancing = telem.get("balancing_active", [False] * 16)

cols = st.columns(8)
cols2 = st.columns(8)
all_cell_cols = cols + cols2

for i in range(16):
    col = all_cell_cols[i]
    cv = cell_vs[i] if i < len(cell_vs) else 3.7
    ct = cell_ts[i] if i < len(cell_ts) else 25.0
    cs = (cell_soc_list[i] if i < len(cell_soc_list) else 0.8) * 100.0
    is_bal = balancing[i] if i < len(balancing) else False

    # Dynamic styling based on thermal & voltage status
    bg_color = "#1e222d"
    border_color = "#363c4e"
    if ct > PACK_CONFIG["critical_cell_temp_c"] or cv < PACK_CONFIG["under_voltage_limit_v"]:
        bg_color = "#4a151b"
        border_color = "#f23645"
    elif ct > PACK_CONFIG["max_cell_temp_c"] or (cv - min(cell_vs)) > 0.05:
        bg_color = "#3d2e14"
        border_color = "#ff9800"

    bal_badge = "⚡ Bal" if is_bal else ""

    with col:
        st.markdown(
            f"""
            <div class="cell-box" style="background:{bg_color}; border:1px solid {border_color};">
                <div style="font-size:11px; color:#848e9c;">CELL #{i+1} <span style="color:#00e676;">{bal_badge}</span></div>
                <div style="font-size:15px; color:#fff; font-weight:700;">{cv:.3f} V</div>
                <div style="font-size:11px; color:#b2b5be;">{ct:.1f} °C | {cs:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")

# ==================== REAL-TIME TELEMETRY CHARTS ====================
st.subheader("📈 Real-Time Digital Twin Analytics")

telemetry_history = snapshot["telemetry_history"]
ai_history = snapshot["ai_history"]

if len(telemetry_history) > 2:
    df_telem = pd.DataFrame(telemetry_history)
    df_ai = pd.DataFrame(ai_history)

    # Chart 1: Ground Truth vs Coulomb Counting vs AI Digital Twin Estimation
    fig_soc = go.Figure()
    
    # Ground Truth
    fig_soc.add_trace(
        go.Scatter(
            y=df_telem["true_soc"] * 100.0,
            mode="lines",
            name="Ground Truth SOC (Physics)",
            line=dict(color="#00e676", width=2.5),
        )
    )
    # Naive Coulomb Counting (Hardware BMS with sensor drift)
    fig_soc.add_trace(
        go.Scatter(
            y=df_telem["bms_coulomb_soc"] * 100.0,
            mode="lines",
            name="Traditional BMS (Coulomb Counting with Drift)",
            line=dict(color="#ff9800", width=1.8, dash="dot"),
        )
    )
    # AI Digital Twin
    if "ai_soc" in df_ai:
        fig_soc.add_trace(
            go.Scatter(
                y=df_ai["ai_soc"] * 100.0,
                mode="lines",
                name="PyTorch AI Digital Twin Estimate",
                line=dict(color="#2962ff", width=2.5),
            )
        )

    fig_soc.update_layout(
        title="<b>Battery State of Charge (SOC) Tracking: AI Twin vs Traditional BMS Drift</b>",
        xaxis_title="Time Steps (0.5s interval)",
        yaxis_title="State of Charge (%)",
        template="plotly_dark",
        height=340,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_soc, use_container_width=True)

    # 2 Subplots: Voltage/Current and Temperature
    row_c1, row_c2 = st.columns(2)

    with row_c1:
        fig_vi = make_subplots(specs=[[{"secondary_y": True}]])
        fig_vi.add_trace(
            go.Scatter(y=df_telem["pack_voltage"], name="Pack Voltage (V)", line=dict(color="#26a69a", width=2)),
            secondary_y=False,
        )
        fig_vi.add_trace(
            go.Scatter(y=df_telem["pack_current"], name="Pack Current (A)", line=dict(color="#ab47bc", width=1.5)),
            secondary_y=True,
        )
        fig_vi.update_layout(
            title="<b>Pack Electrical Dynamics (V & I)</b>",
            template="plotly_dark",
            height=280,
            margin=dict(l=40, r=40, t=40, b=30),
            legend=dict(orientation="h", y=-0.2),
        )
        fig_vi.update_yaxes(title_text="Voltage (V)", secondary_y=False)
        fig_vi.update_yaxes(title_text="Current (A)", secondary_y=True)
        st.plotly_chart(fig_vi, use_container_width=True)

    with row_c2:
        fig_temp = go.Figure()
        fig_temp.add_trace(
            go.Scatter(y=df_telem["max_temp"], name="Max Cell Temp (°C)", line=dict(color="#f23645", width=2))
        )
        fig_temp.add_trace(
            go.Scatter(y=df_telem["avg_temp"], name="Avg Pack Temp (°C)", line=dict(color="#ffa726", width=1.5))
        )
        fig_temp.add_hline(
            y=PACK_CONFIG["critical_cell_temp_c"],
            line_dash="dash",
            line_color="#f23645",
            annotation_text="Critical Runaway Threshold",
        )
        fig_temp.update_layout(
            title="<b>Thermal Dynamics & Runaway Threshold</b>",
            xaxis_title="Time Steps",
            yaxis_title="Temperature (°C)",
            template="plotly_dark",
            height=280,
            margin=dict(l=40, r=40, t=40, b=30),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_temp, use_container_width=True)

# Footer & Auto-refresh
st.markdown("---")
col_b1, col_b2 = st.columns([3, 1])
with col_b1:
    st.caption(
        f"Synchronized with MQTT Broker: `{broker_host}:{broker_port}` | "
        f"Packets Processed: {snapshot['packets_received']} | "
        f"Telemetry Rate: {1/MQTT_CONFIG['telemetry_interval_sec']} Hz"
    )
with col_b2:
    auto_refresh = st.checkbox("Auto Refresh (1s)", value=True)

if auto_refresh:
    time.sleep(1.0)
    st.rerun()
