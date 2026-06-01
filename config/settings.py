# Equipment & Pipeline Configuration
MACHINE_ID = "M01"
SITE_ID = "Plant_A"
POLL_INTERVAL_SEC = 60
FS = 8000                   # Sampling rate (Hz)
TOTAL_SAMPLES = 32768       # Samples per acquisition window

# MongoDB (swap URI for Atlas cloud deployment)
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "cbpm_database"

# Health State Thresholds
STATE_THRESHOLDS = {"Critical": 50.0, "Early Degradation": 75.0, "Normal": 90.0}

# Multi-Equipment Configuration
EQUIPMENT_CONFIG = {
    "fire_pump":    {"fs": 8000, "total_samples": 32768, "sensors": ["vibration", "current", "ultrasonic"]},
    "compressor":   {"fs": 8000, "total_samples": 32768, "sensors": ["vibration", "current", "ultrasonic"]},
    "gearbox":      {"fs": 8000, "total_samples": 32768, "sensors": ["vibration", "ultrasonic"]},
    "shaft_engine": {"fs": 8000, "total_samples": 32768, "sensors": ["vibration", "ultrasonic"]},
}

UNIT_REGISTRY = {
    "FP_01":   {"type": "fire_pump",    "compartment": "Engine_Room", "site": "Plant_A"},
    "COMP_01": {"type": "compressor",   "compartment": "Engine_Room", "site": "Plant_A"},
    "GB_01":   {"type": "gearbox",      "compartment": "Engine_Room", "site": "Plant_A"},
    "GB_02":   {"type": "gearbox",      "compartment": "Engine_Room", "site": "Plant_A"},
    "SE_01":   {"type": "shaft_engine", "compartment": "Engine_Room", "site": "Plant_A"},
    "SE_02":   {"type": "shaft_engine", "compartment": "Engine_Room", "site": "Plant_A"},
}
