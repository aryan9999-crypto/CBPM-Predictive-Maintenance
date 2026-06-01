"""
Deep Autoencoder Anomaly Detection for Industrial IoT
Author: Aryan Gaur (https://github.com/aryan9999-crypto)
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime


class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AnomalyDetector:
    """Production anomaly detector with per-feature reconstruction error attribution."""

    def __init__(self, model, scaler, threshold, feature_columns):
        self.model = model
        self.scaler = scaler
        self.threshold = threshold
        self.feature_columns = feature_columns
        self.device = next(model.parameters()).device
        self.model.eval()

    def run_inference(self, features, machine_id, timestamp=None):
        if timestamp is None:
            timestamp = datetime.utcnow()

        if hasattr(features, 'columns'):
            available = [c for c in self.feature_columns if c in features.columns]
            features = features[available].values

        scaled = self.scaler.transform(features)
        tensor = torch.tensor(scaled, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            reconstructed = self.model(tensor)
            recon_error = (tensor - reconstructed) ** 2

        mse_per_window = recon_error.mean(dim=1).cpu().numpy()
        mean_error = float(mse_per_window.mean())
        is_anomaly = mean_error > self.threshold
        anomaly_rate = float((mse_per_window > self.threshold).mean() * 100)
        health_score = max(0.0, min(100.0, 100.0 - anomaly_rate))

        # Per-feature attribution for root-cause analysis
        feature_errors = recon_error.mean(dim=0).cpu().numpy()
        top_idx = np.argsort(feature_errors)[::-1][:5]
        attribution = {self.feature_columns[i]: float(feature_errors[i]) for i in top_idx if i < len(self.feature_columns)}

        return {
            "machine_id": machine_id, "timestamp": timestamp.isoformat() + "Z",
            "is_anomaly": bool(is_anomaly), "anomaly_rate_percent": round(anomaly_rate, 4),
            "mean_reconstruction_error": round(mean_error, 6), "threshold": self.threshold,
            "health_score": round(health_score, 2), "windows_analyzed": len(mse_per_window),
            "feature_attribution": attribution,
        }


def train_autoencoder(train_features, latent_dim=16, epochs=100, batch_size=64, lr=1e-3):
    """Train autoencoder on healthy baseline. Returns (model, scaler, threshold)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(train_features)
    model = DeepAutoencoder(scaled.shape[1], latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = torch.tensor(scaled, dtype=torch.float32, device=device)

    model.train()
    for epoch in range(epochs):
        idx = torch.randperm(len(dataset))
        for start in range(0, len(dataset), batch_size):
            batch = dataset[idx[start:start + batch_size]]
            loss = nn.MSELoss()(model(batch), batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs} — Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        errors = ((dataset - model(dataset)) ** 2).mean(dim=1).cpu().numpy()
    threshold = float(np.percentile(errors, 95))
    return model, scaler, threshold


if __name__ == "__main__":
    np.random.seed(42)
    healthy = np.random.randn(5000, 50).astype(np.float32) * 0.5
    print("Training autoencoder...")
    model, scaler, threshold = train_autoencoder(healthy, epochs=50)
    cols = [f"feature_{i}" for i in range(50)]
    detector = AnomalyDetector(model, scaler, threshold, cols)
    print(f"Healthy: {detector.run_inference(healthy[:100], 'DEMO')['health_score']}%")
    anomalous = healthy[:100] + np.random.randn(100, 50) * 3
    print(f"Anomaly: {detector.run_inference(anomalous, 'DEMO')['health_score']}%")
