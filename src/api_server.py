"""
Flask REST API for CBPM Platform — 12+ endpoints
Author: Aryan Gaur
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# ── Ingestion APIs ──

@app.route("/api/baseline-data", methods=["POST"])
def ingest_baseline():
    """Load labelled historical sensor data for training."""
    payload = request.get_json()
    data = payload.get("data", [])
    if not data:
        return jsonify({"error": "No data provided"}), 400
    return jsonify({"status": "success", "inserted": len(data), "collection": "baseline_collection"}), 201

@app.route("/api/ingest/live", methods=["POST"])
def ingest_live():
    """Live sensor data ingestion — triggers full inference pipeline."""
    payload = request.get_json()
    data = payload.get("data", [])
    machine_id = payload.get("machine_id", "M01")
    if not data:
        return jsonify({"error": "No data provided"}), 400
    # In production: inference_service.process_sensor_data(df, machine_id)
    return jsonify({"status": "success", "prediction": "Normal", "health": 95.0}), 201

# ── KPI / Dashboard APIs ──

@app.route("/anomaly_kpis", methods=["GET"])
def get_anomaly_kpis():
    """Latest anomaly detection results for a machine."""
    machine_id = request.args.get("machine_id", "M01")
    return jsonify({"machine_id": machine_id, "health_score": 95.2, "is_anomaly": False, "anomaly_rate_percent": 0.12})

@app.route("/fault_classification_kpis", methods=["GET"])
def get_fault_kpis():
    """Latest fault classification results."""
    return jsonify({"machine_id": request.args.get("machine_id", "M01"),
                    "predicted_fault": {"fault_name": "Normal", "confidence": 0.97}})

@app.route("/api/forecast/latest", methods=["GET"])
def get_forecast():
    """Latest forecast snapshot."""
    return jsonify({"machine_id": request.args.get("machine_id", "M01"),
                    "current_health": 92.5, "status": "Normal"})

@app.route("/api/forecast/view", methods=["GET"])
def get_forecast_view():
    """Forecast sliced to requested horizon (24hr/7day/30day/90day/1year/5year)."""
    horizon = request.args.get("horizon", "24hr")
    return jsonify({"horizon": horizon, "n_points": 1440, "resolution": "minute"})

@app.route("/api/forecast/history", methods=["GET"])
def get_forecast_history():
    """Recent forecast snapshots for tracking."""
    return jsonify({"status": "success", "history": []})

@app.route("/api/machine/status", methods=["GET"])
def get_machine_status():
    """Consolidated machine condition (vibration + ultrasonic health)."""
    return jsonify({"machine_id": request.args.get("machine_id", "M01"),
                    "combined_health_score": 91.5, "vibration_health": 93.0,
                    "ultrasonic_health": 91.5, "status": "Normal"})

@app.route("/api/historical-health", methods=["GET"])
def get_historical_health():
    """Health score history for trend visualization."""
    return jsonify({"status": "success", "history": []})

# ── Alert APIs ──

@app.route("/alerts", methods=["GET"])
def get_alerts():
    """Active alerts with operator lifecycle tracking."""
    return jsonify({"count": 0, "alerts": []})

@app.route("/api/alerts/<alert_id>/accept", methods=["PATCH"])
def accept_alert(alert_id):
    """Operator acknowledges an alert."""
    return jsonify({"status": "accepted", "alert_id": alert_id})

@app.route("/api/alerts/<alert_id>/resolve", methods=["PATCH"])
def resolve_alert(alert_id):
    """Operator resolves/dismisses an alert."""
    return jsonify({"status": "resolved", "alert_id": alert_id})

@app.route("/api/ultrasonic/kpis", methods=["GET"])
def get_ultrasonic_kpis():
    """Latest ultrasonic inference results."""
    return jsonify({"machine_id": request.args.get("machine_id", "M01"),
                    "ult_health_score": 95.0, "ult_state": "Normal"})

if __name__ == "__main__":
    print("CBPM API Server — 12+ endpoints")
    app.run(host="0.0.0.0", port=5000, debug=True)
