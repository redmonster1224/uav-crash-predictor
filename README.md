# uav-crash-predictor
LSTM Autoencoder for real-time anomaly detection and crash prediction using UAV flight logs
# 🚁 UAV Crash Predictor Engine
**State-of-the-Art LSTM Autoencoder for Real-Time Flight Anomaly Detection**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

An advanced machine learning pipeline that analyzes raw UAV flight logs (`.ulog`) to predict structural failures and unrecoverable aerodynamic stalls before they happen. By utilizing a Self-Supervised LSTM Autoencoder, this system learns the complex flight dynamics of normal operations and flags catastrophic deviations in real-time.

---

## 🏗️ System Architecture

### 1. Data Pipeline & Inference Engine
The ground control safety engine ingests live telemetry (Pitch, Roll, Motor RPM, Battery Voltage) at 50Hz, buffering it into rolling windows for real-time inference.

```mermaid
graph TD;
    A[Live UAV Telemetry stream] -->|50 Hz| B(Rolling Data Buffer FIFO);
    B --> C{LSTM Autoencoder};
    C -->|Reconstructs Physics| D[Calculates MSE Loss];
    D --> E{Threshold Check};
    E -->|> 0.05| F[🚨 CRITICAL: Trigger RTL/Parachute];
    E -->|< 0.05| G[✅ NOMINAL: Stable Flight];
    
    style F fill:#ff4d4d,stroke:#990000,stroke-width:2px,color:#fff
    style G fill:#4dff4d,stroke:#009900,stroke-width:2px,color:#000
