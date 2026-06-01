"""
GPU-Accelerated Feature Extraction for Industrial IoT Sensor Data
=================================================================
Processes multi-sensor time-series data through sliding windows using PyTorch
for parallel computation on GPU. Dynamically adapts to any sensor type
(vibration, temperature, pressure, current, ultrasonic).

Computes ~198 features per inference window:
- Time-domain: mean, var, std, RMS, peak, P2P, skewness, kurtosis, median, IQR, MAD, energy
- Ratio: crest factor, clearance factor, shape factor, impulse factor  
- Frequency-domain: spectral centroid, spread, flux, rolloff, 6 band energies

Author: Aryan Gaur (https://github.com/aryan9999-crypto)
"""

import torch
import pandas as pd
import numpy as np
import re


class PyTorchVibrationFeatureExtractor:
    """
    GPU-accelerated feature extractor for condition-based predictive maintenance.
    Processes sliding windows in parallel using PyTorch tensors.

    Dynamically adapts to input DataFrame columns formatted as:
        SENSOR_NAME(sensor_type)
        e.g., DEV(vibration), BEARING_A(temp), PUMP(pressure)
    """

    def __init__(self, sampling_rate: int = 8000, device=None):
        self.sampling_rate = sampling_rate
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def _sliding_windows(self, tensor: torch.Tensor, window_size: int, step: int):
        """Create sliding windows using PyTorch unfold — zero-copy on GPU."""
        return tensor.unfold(0, window_size, step)

    # ─────────────────────────────────────────────────────────────
    # Vibration Feature Extraction (High-Frequency Sensors)
    # ─────────────────────────────────────────────────────────────

    def extract_vibration_features(self, windows: torch.Tensor) -> dict:
        """
        Extract ~33 features per vibration sensor from sliding windows.

        Args:
            windows: Tensor of shape (num_windows, window_size)

        Returns:
            Dictionary of feature_name -> tensor(num_windows,)
        """
        f = {}
        W = windows.shape[1]
        num_windows = windows.shape[0]

        # ── Time Domain ──
        f["mean"] = windows.mean(dim=1)
        f["var"] = windows.var(dim=1, unbiased=False)
        f["std"] = windows.std(dim=1, unbiased=False)
        f["rms"] = torch.sqrt((windows ** 2).mean(dim=1))

        abs_windows = windows.abs()
        f["peak"] = abs_windows.max(dim=1).values
        f["peak_to_peak"] = windows.max(dim=1).values - windows.min(dim=1).values

        # Higher-order statistics
        safe_std = torch.clamp(f["std"], min=1e-8)
        diffs = windows - f["mean"].unsqueeze(1)
        f["skewness"] = (diffs ** 3).mean(dim=1) / (safe_std ** 3)
        f["kurtosis"] = (diffs ** 4).mean(dim=1) / (safe_std ** 4) - 3.0  # Excess kurtosis

        # Robust statistics
        f["median"] = windows.median(dim=1).values
        q75 = torch.quantile(windows, 0.75, dim=1)
        q25 = torch.quantile(windows, 0.25, dim=1)
        f["iqr"] = q75 - q25

        diff_med = (windows - f["median"].unsqueeze(1)).abs()
        f["mad"] = diff_med.median(dim=1).values

        # Energy
        f["energy"] = (windows ** 2).sum(dim=1)
        f["log_energy"] = torch.log(f["energy"] + 1e-8)

        # ── Ratio Features (ISO standard indicators) ──
        mean_sqrt = abs_windows.sqrt().mean(dim=1)
        mean_abs = abs_windows.mean(dim=1)
        safe_rms = torch.clamp(f["rms"], min=1e-8)
        safe_mean_sqrt2 = torch.clamp(mean_sqrt ** 2, min=1e-8)
        safe_mean_abs = torch.clamp(mean_abs, min=1e-8)

        f["crest_factor"] = f["peak"] / safe_rms
        f["clearance_factor"] = f["peak"] / safe_mean_sqrt2
        f["shape_factor"] = f["rms"] / safe_mean_abs
        f["impulse_factor"] = f["peak"] / safe_mean_abs

        # Handle zero-signal edge case
        zero_mask = f["rms"] == 0
        f["crest_factor"][zero_mask] = 0
        f["clearance_factor"][zero_mask] = 0
        f["shape_factor"][zero_mask] = 0
        f["impulse_factor"][zero_mask] = 0

        # ── Frequency Domain (FFT-based) ──
        fft_vals = torch.fft.rfft(windows, dim=1)
        magnitude = fft_vals.abs()
        half_len = W // 2
        magnitude = magnitude[:, :half_len]
        freqs = torch.fft.rfftfreq(W, 1.0 / self.sampling_rate, device=self.device)[:half_len]

        total_mag = magnitude.sum(dim=1)
        safe_total_mag = torch.clamp(total_mag, min=1e-8)

        f["spectral_centroid"] = (magnitude * freqs).sum(dim=1) / safe_total_mag

        spread_diff = freqs.unsqueeze(0) - f["spectral_centroid"].unsqueeze(1)
        f["spectral_spread"] = torch.sqrt(
            ((spread_diff ** 2) * magnitude).sum(dim=1) / safe_total_mag
        )
        f["spectral_flux"] = (torch.diff(magnitude, dim=1) ** 2).sum(dim=1)

        # Spectral rolloff (85th percentile)
        cum_mag = torch.cumsum(magnitude, dim=1)
        rolloff_threshold = 0.85 * total_mag.unsqueeze(1)
        rolloff_idx = (cum_mag >= rolloff_threshold).long().argmax(dim=1)
        f["spectral_rolloff"] = freqs[rolloff_idx]

        # Band energies (6 frequency bands)
        freq_bands = [
            (0, 50), (50, 150), (150, 300),
            (300, 500), (500, 1000), (1000, 2000)
        ]
        for i, (low, high) in enumerate(freq_bands):
            mask = (freqs >= low) & (freqs < high)
            if not mask.any():
                f[f"freq_band_{i}_energy"] = torch.zeros(num_windows, device=self.device)
                f[f"freq_band_{i}_mean_freq"] = torch.zeros(num_windows, device=self.device)
            else:
                band_mag = magnitude[:, mask]
                f[f"freq_band_{i}_energy"] = (band_mag ** 2).sum(dim=1)
                f[f"freq_band_{i}_mean_freq"] = torch.full(
                    (num_windows,), freqs[mask].mean().item(), device=self.device
                )

        return f

    def extract_slow_varying_features(self, windows: torch.Tensor) -> dict:
        """Extract features for slow-varying sensors (temp, pressure, current)."""
        f = {}
        f["mean"] = windows.mean(dim=1)
        f["max"] = windows.max(dim=1).values
        f["min"] = windows.min(dim=1).values
        f["var"] = windows.var(dim=1, unbiased=False)
        f["trend"] = windows[:, -1] - windows[:, 0]
        return f

    # ─────────────────────────────────────────────────────────────
    # Main Pipeline
    # ─────────────────────────────────────────────────────────────

    def create_feature_matrix(
        self, df: pd.DataFrame, window_size: int = 2000, overlap: float = 0.8
    ) -> pd.DataFrame:
        """
        Process a multi-sensor DataFrame into a feature matrix.

        Args:
            df: DataFrame with columns like 'DEV(vibration)', 'PUMP(pressure)', etc.
            window_size: Samples per sliding window (default: 2000)
            overlap: Window overlap fraction (default: 0.8 = 80%)

        Returns:
            DataFrame with ~198 features per window
        """
        step = int(window_size * (1 - overlap))
        num_rows = len(df)

        if num_rows < window_size:
            return pd.DataFrame()

        num_windows = (num_rows - window_size) // step + 1
        starts = np.arange(0, num_windows * step, step)
        ends = starts + window_size

        results = {
            "window_id": np.arange(num_windows),
            "start_idx": starts,
            "end_idx": ends,
        }

        if self.device.type == "cpu":
            print("[INFO] Running feature extraction on CPU (no CUDA GPU detected)")

        # Dynamically parse sensor types from column names
        for col in df.columns:
            match = re.match(r'^(.*?)\((.*?)\)$', col)
            if match:
                sensor_name, sensor_type = match.groups()
                sensor_type = sensor_type.lower()

                col_values = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=np.float32)
                if np.isnan(col_values).any():
                    nan_count = np.isnan(col_values).sum()
                    print(f"[WARN] Column '{col}' had {nan_count} NaN values. Filling with 0.")
                    col_values = np.nan_to_num(col_values, nan=0.0)

                tensor = torch.tensor(col_values, dtype=torch.float32, device=self.device)
                windows = self._sliding_windows(tensor, window_size, step)

                with torch.no_grad():
                    if sensor_type == "vibration":
                        feats = self.extract_vibration_features(windows)
                    else:
                        feats = self.extract_slow_varying_features(windows)

                for k, v_tensor in feats.items():
                    results[f"{sensor_name}_{k}"] = v_tensor.cpu().numpy()

        return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    # Generate synthetic multi-sensor data
    N = 100_000
    dummy = pd.DataFrame({
        "DEV(vibration)": np.random.randn(N).astype(np.float32),
        "BEARING_A(vibration)": np.random.randn(N).astype(np.float32),
        "PUMP(pressure)": np.random.uniform(2.0, 5.0, N).astype(np.float32),
        "MOTOR(temp)": np.random.uniform(60, 90, N).astype(np.float32),
        "DRIVE(current)": np.random.uniform(10, 25, N).astype(np.float32),
    })

    extractor = PyTorchVibrationFeatureExtractor(sampling_rate=8000)

    import time
    start = time.perf_counter()
    feature_df = extractor.create_feature_matrix(dummy, window_size=2000, overlap=0.8)
    elapsed = time.perf_counter() - start

    print(f"\nResults:")
    print(f"  Input samples: {N:,}")
    print(f"  Feature windows: {len(feature_df)}")
    print(f"  Features per window: {len(feature_df.columns) - 3}")
    print(f"  Total columns: {list(feature_df.columns)[:15]}...")
    print(f"  Extraction time: {elapsed:.3f}s")
    print(f"  Device: {extractor.device}")
