"""
Ensemble Fault Classifier for Predictive Maintenance
Soft-Voting Ensemble: Random Forest + Extra Trees + HistGradientBoosting
Author: Aryan Gaur
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from datetime import datetime


class FaultClassifier:
    """Soft-voting ensemble for multi-class fault diagnosis."""

    def __init__(self, model, scaler, label_encoder, feature_columns):
        self.model = model
        self.scaler = scaler
        self.le = label_encoder
        self.feature_columns = feature_columns

    def classify(self, features, machine_id, timestamp=None):
        if timestamp is None:
            timestamp = datetime.utcnow()
        if hasattr(features, 'columns'):
            available = [c for c in self.feature_columns if c in features.columns]
            features = features[available].values
        scaled = self.scaler.transform(features)
        predictions = self.model.predict(scaled)
        probabilities = self.model.predict_proba(scaled)
        # Majority vote across windows
        from collections import Counter
        vote = Counter(predictions).most_common(1)[0][0]
        fault_name = self.le.inverse_transform([vote])[0]
        avg_proba = probabilities.mean(axis=0)
        confidence = float(avg_proba.max())
        return {
            "machine_id": machine_id, "timestamp": timestamp.isoformat() + "Z",
            "predicted_fault": {"fault_name": fault_name, "confidence": round(confidence, 4)},
            "class_probabilities": {self.le.inverse_transform([i])[0]: round(float(p), 4) for i, p in enumerate(avg_proba)},
            "windows_analyzed": len(predictions),
        }


def train_fault_classifier(features, labels):
    """Train soft-voting ensemble. Returns (model, scaler, label_encoder)."""
    le = LabelEncoder()
    y = le.fit_transform(labels)
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    ensemble = VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)),
        ('et', ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=42)),
        ('hgb', HistGradientBoostingClassifier(max_iter=200, max_depth=8, random_state=42)),
    ], voting='soft')
    scores = cross_val_score(ensemble, X, y, cv=5, scoring='accuracy')
    print(f"  CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    ensemble.fit(X, y)
    return ensemble, scaler, le

if __name__ == "__main__":
    np.random.seed(42)
    n, f = 2000, 50
    X = np.vstack([np.random.randn(n//4, f) * s + m for s, m in [(0.5, 0), (1.0, 2), (0.8, -1), (1.2, 1)]])
    y = np.array(["Normal"]*500 + ["Bearing_Fault"]*500 + ["Misalignment"]*500 + ["Imbalance"]*500)
    model, scaler, le = train_fault_classifier(X, y)
    clf = FaultClassifier(model, scaler, le, [f"f_{i}" for i in range(f)])
    print(clf.classify(X[:10], "DEMO"))
