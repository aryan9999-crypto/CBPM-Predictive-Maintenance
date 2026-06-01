"""
Synthetic Data Generator for CBPM Demo
Generates realistic multi-sensor industrial IoT data.
Author: Aryan Gaur
"""
import numpy as np
import pandas as pd

def generate_vibration_signal(n_samples, fs=8000, fault_type="normal"):
    t = np.arange(n_samples) / fs
    signal = np.random.randn(n_samples) * 0.1  # Base noise
    signal += 0.5 * np.sin(2 * np.pi * 29.6 * t)  # Shaft frequency
    signal += 0.3 * np.sin(2 * np.pi * 120 * t)   # Bearing frequency

    if fault_type == "bearing_fault":
        signal += 1.5 * np.sin(2 * np.pi * 236.8 * t) + np.random.randn(n_samples) * 0.5
    elif fault_type == "misalignment":
        signal += 1.0 * np.sin(2 * np.pi * 59.2 * t) + 0.8 * np.sin(2 * np.pi * 88.8 * t)
    elif fault_type == "imbalance":
        signal += 2.0 * np.sin(2 * np.pi * 29.6 * t)
    return signal.astype(np.float32)

def generate_dataset(n_samples=100000, fault_type="normal"):
    return pd.DataFrame({
        "DEV(vibration)": generate_vibration_signal(n_samples, fault_type=fault_type),
        "BEARING_A(vibration)": generate_vibration_signal(n_samples, fault_type=fault_type) * 0.8,
        "PUMP(pressure)": np.random.uniform(2.5, 4.5, n_samples).astype(np.float32),
        "MOTOR(temp)": np.random.uniform(65, 85, n_samples).astype(np.float32),
        "DRIVE(current)": np.random.uniform(12, 22, n_samples).astype(np.float32),
    })

if __name__ == "__main__":
    for fault in ["normal", "bearing_fault", "misalignment", "imbalance"]:
        df = generate_dataset(50000, fault)
        print(f"{fault}: {df.shape} — vibration range [{df['DEV(vibration)'].min():.2f}, {df['DEV(vibration)'].max():.2f}]")
