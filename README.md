# 🏭 CBPM — Condition-Based Predictive Maintenance Platform

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-Time%20Series-green.svg)](https://mongodb.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **production-grade ML pipeline** for industrial IoT predictive maintenance — monitoring multiple equipment units 24/7 across vibration, current, and ultrasonic sensor modalities with real-time anomaly detection, fault classification, and multi-horizon failure forecasting.

> **Note:** This is a sanitised, open-source version of a production system. All proprietary data, credentials, and client-specific configurations have been removed. The architecture, algorithms, and pipeline logic are preserved with synthetic data for demonstration.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CBPM Platform Architecture                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────────┐    ┌─────────────────────┐   │
│  │  IoT Sensors  │───▶│  Data Ingestion   │───▶│  Feature Extraction │   │
│  │  • Vibration  │    │  • API Client     │    │  • PyTorch/CUDA     │   │
│  │  • Current    │    │  • DAQ Reader     │    │  • 198 Features     │   │
│  │  • Ultrasonic │    │  • MongoDB Store  │    │  • Sliding Window   │   │
│  └──────────────┘    └──────────────────┘    └─────────┬───────────┘   │
│                                                         │               │
│                    ┌────────────────────────────────────┘               │
│                    ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     ML Inference Pipeline                        │   │
│  │  ┌────────────────┐  ┌──────────────────┐  ┌────────────────┐  │   │
│  │  │ Anomaly Detect │  │ Fault Classify   │  │ RUL Forecast   │  │   │
│  │  │ Deep Autoencdr │  │ Soft-Vote Ensemb │  │ XGBoost+Weibul │  │   │
│  │  │ <0.3% FP Rate  │  │ RF+ET+HistGBM    │  │ 1min → 5years  │  │   │
│  │  └────────────────┘  └──────────────────┘  └────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                    │                                                     │
│                    ▼                                                     │
│  ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────┐   │
│  │  Alert Engine     │    │  REST API (Flask) │    │  MongoDB Atlas   │   │
│  │  • Threshold      │    │  • 12+ Endpoints  │    │  • Time Series   │   │
│  │  • Severity       │    │  • Health Scores  │    │  • Compression   │   │
│  │  • Remediation    │    │  • Forecasts      │    │  • Indexing      │   │
│  └──────────────────┘    └──────────────────┘    └─────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Multi-Sensor Fusion** | Vibration (8kHz), current, and ultrasonic sensors processed in parallel |
| **GPU-Accelerated Features** | ~198 time-domain + frequency-domain features extracted via PyTorch CUDA |
| **Anomaly Detection** | Deep Autoencoder with per-feature reconstruction error attribution |
| **Fault Classification** | Soft-Voting Ensemble (Random Forest + Extra Trees + HistGBM) |
| **Multi-Horizon Forecasting** | XGBoost + Ridge + Weibull AFT survival analysis (1 min → 5 years) |
| **Real-Time API** | 12+ Flask REST endpoints for inference, health scores, and alerts |
| **MongoDB Time Series** | Optimised storage with zlib compression (60-80% reduction) |
| **Alert Workflow** | Threshold-based triggers with operator acknowledgement lifecycle |
| **Multi-Equipment** | Single codebase supporting multiple equipment types and units |

---

## 📁 Project Structure

```
cbpm-predictive-maintenance/
├── src/
│   ├── feature_extractor.py      # GPU-accelerated feature extraction (PyTorch)
│   ├── anomaly_detector.py       # Deep Autoencoder anomaly detection
│   ├── fault_classifier.py       # Ensemble fault classification
│   ├── forecaster.py             # Multi-horizon RUL forecasting
│   ├── api_server.py             # Flask REST API (12+ endpoints)
│   ├── alert_engine.py           # Intelligent alert workflow
│   ├── mongo_setup.py            # MongoDB Time Series configuration
│   ├── inference_service.py      # Orchestration service
│   └── synthetic_data.py         # Synthetic data generator for demos
├── config/
│   └── settings.py               # Equipment & pipeline configuration
├── docs/
│   └── architecture.md           # Detailed architecture documentation
├── Dockerfile                    # Container-ready deployment
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- MongoDB 6.0+ (or MongoDB Atlas)
- CUDA-capable GPU (optional, falls back to CPU)

### Installation

```bash
git clone https://github.com/aryan9999-crypto/cbpm-predictive-maintenance.git
cd cbpm-predictive-maintenance
pip install -r requirements.txt
```

### Run with Synthetic Data

```bash
# Generate synthetic sensor data
python src/synthetic_data.py

# Start the API server
python src/api_server.py

# Server runs at http://localhost:5000
```

### Docker

```bash
docker build -t cbpm-platform .
docker run -p 5000:5000 cbpm-platform
```

---

## 🔬 Technical Deep-Dive

### Feature Extraction (GPU-Accelerated)

The feature extractor processes raw sensor signals through sliding windows and computes:

**Time-Domain (per sensor):**
- Statistical: mean, variance, std, RMS, peak, peak-to-peak, skewness, kurtosis
- Robust: median, IQR, MAD
- Energy: signal energy, log energy
- Shape: crest factor, clearance factor, shape factor, impulse factor

**Frequency-Domain (vibration sensors):**
- Spectral: centroid, spread, flux, rolloff
- Band energies: 6 frequency bands (0-50Hz, 50-150Hz, 150-300Hz, 300-500Hz, 500-1000Hz, 1000-2000Hz)

### Anomaly Detection

Deep Autoencoder trained on healthy baseline data:
- Reconstruction error threshold learned from training distribution
- Per-feature attribution identifies which sensor channels drive anomalies
- Health score: `100 - anomaly_rate_percent`

### Multi-Horizon Forecasting

| Horizon | Model | Resolution |
|---|---|---|
| 24 hours | XGBoost | 1-minute intervals |
| 7-365 days | Ridge Regression | Daily |
| 2-5 years | Weibull AFT Survival | Monthly |

---

## 📊 Results

| Metric | Value |
|---|---|
| False Positive Rate | < 0.3% |
| Inference Latency (GPU) | ~50ms per window |
| Feature Extraction (GPU vs CPU) | >10× speedup |
| Storage Compression | 60-80% reduction |
| Equipment Types Supported | 4 (fire pump, compressor, gearbox, shaft engine) |
| Concurrent Units | 6 |

---

## 🛠️ Tech Stack

`Python` `PyTorch` `scikit-learn` `XGBoost` `Flask` `MongoDB` `NumPy` `SciPy` `CUDA` `lifelines` `Docker`

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Aryan Gaur** — AI/ML Engineer  
[LinkedIn](https://linkedin.com/in/aryan-gaur-ai) · [GitHub](https://github.com/aryan9999-crypto) · [Email](mailto:aryangaur201@gmail.com)
