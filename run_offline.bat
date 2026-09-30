@echo off
title EV Battery Pack AI Digital Twin (Offline Mode)
echo ===============================================================
echo ⚡ Launching EV Battery Pack AI Digital Twin (Local Mosquitto)
echo ===============================================================
echo Target Broker: 127.0.0.1:1883
echo.

python -m streamlit run app.py
pause
