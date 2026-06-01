"""
MongoDB Time Series Setup for CBPM Platform
Configures Time Series collections with compression and indexing.
Author: Aryan Gaur
"""
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import CollectionInvalid

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "cbpm_database"

def setup_database(uri=MONGO_URI, db_name=DB_NAME):
    client = MongoClient(uri)
    db = client[db_name]

    # Time Series collections for optimised time-range queries
    ts_collections = {
        "anomaly_collection":    {"timeField": "timestamp", "metaField": "machine_id"},
        "fault_collection":      {"timeField": "timestamp", "metaField": "machine_id"},
        "forecast_collection":   {"timeField": "timestamp", "metaField": "machine_id"},
        "machine_forecasts":     {"timeField": "timestamp", "metaField": "machine_id"},
        "ultrasonic_collection": {"timeField": "timestamp", "metaField": "machine_id"},
    }

    for col_name, ts_opts in ts_collections.items():
        try:
            db.create_collection(col_name, timeseries=ts_opts)
            print(f"  Created Time Series: {col_name}")
        except CollectionInvalid:
            print(f"  Exists: {col_name}")

    # Standard collections
    col_alerts = db["alerts_collection"]
    col_raw = db["raw_data_bin"]

    # Indexes
    col_alerts.create_index([("alert_id", ASCENDING)], unique=True, sparse=True)
    col_alerts.create_index([("status", ASCENDING)])
    col_raw.create_index([("machine_id", ASCENDING), ("timestamp", DESCENDING)])

    print("MongoDB setup complete.")
    return db

if __name__ == "__main__":
    setup_database()
