# Football Analytics Configuration

import os
from pathlib import Path
from typing import Dict, Any

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Database configuration
DATABASE_CONFIG = {
    "url": os.getenv("DATABASE_URL", "sqlite:///football_analytics.db"),
    "echo": os.getenv("DEBUG", "False").lower() == "true"
}

# Redis configuration
REDIS_CONFIG = {
    "url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    "decode_responses": True
}

# API configurations
API_CONFIGS = {
    "football_data": {
        "base_url": "https://api.football-data.org/v4",
        "api_key": os.getenv("FOOTBALL_DATA_API_KEY"),
        "rate_limit": 10  # requests per minute
    },
    "sportmonks": {
        "base_url": "https://api.sportmonks.com/v3",
        "api_key": os.getenv("SPORTMONKS_API_KEY"),
        "rate_limit": 60  # requests per minute
    }
}

# Model configurations
MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    },
    "xgboost": {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    },
    "neural_network": {
        "hidden_layers": [128, 64, 32],
        "dropout_rate": 0.3,
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.001
    },
    "ensemble": {
        "models": ["random_forest", "xgboost", "neural_network"],
        "voting": "soft"
    }
}

# Feature engineering settings
FEATURE_CONFIG = {
    "rolling_windows": [3, 5, 10],
    "position_weights": {
        "GK": {
            "saves": 1.0,
            "clean_sheets": 1.0,
            "distribution_accuracy": 0.8,
            "goals_conceded": -1.0
        },
        "DEF": {
            "tackles": 1.0,
            "interceptions": 1.0,
            "aerial_duels_won": 0.9,
            "pass_accuracy": 0.7,
            "clean_sheets": 0.8
        },
        "MID": {
            "key_passes": 1.0,
            "dribbles_successful": 0.8,
            "pass_accuracy": 1.0,
            "distance_covered": 0.6,
            "possession_won": 0.7
        },
        "ATT": {
            "goals": 1.0,
            "assists": 0.8,
            "shots_on_target": 0.9,
            "expected_goals": 1.0,
            "conversion_rate": 0.9
        }
    }
}

# Data collection settings
DATA_COLLECTION = {
    "leagues": ["PL", "CL", "BL1", "SA", "FL1"],  # Premier League, Champions League, etc.
    "seasons": ["2022", "2023", "2024"],
    "update_frequency": 24,  # hours
    "batch_size": 100
}

# Prediction settings
PREDICTION_CONFIG = {
    "confidence_threshold": float(os.getenv("PREDICTION_CONFIDENCE_THRESHOLD", "0.7")),
    "update_interval": 6,  # hours
    "lookback_days": 30
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(LOGS_DIR / "football_analytics.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
    },
    "loggers": {
        "": {
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "handlers": ["console", "file"]
        }
    }
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    "title": "Football Analytics Platform",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "page_icon": "⚽"
}

# API server configuration
API_CONFIG = {
    "host": os.getenv("FASTAPI_HOST", "0.0.0.0"),
    "port": int(os.getenv("FASTAPI_PORT", "8000")),
    "debug": os.getenv("DEBUG", "False").lower() == "true",
    "reload": os.getenv("DEBUG", "False").lower() == "true"
}

# Security configuration
SECURITY_CONFIG = {
    "secret_key": os.getenv("SECRET_KEY", "your-secret-key-change-in-production"),
    "jwt_secret": os.getenv("JWT_SECRET", "your-jwt-secret-change-in-production"),
    "jwt_algorithm": os.getenv("JWT_ALGORITHM", "HS256"),
    "jwt_expiration_hours": int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
}

# MLflow configuration
MLFLOW_CONFIG = {
    "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
    "experiment_name": os.getenv("MLFLOW_EXPERIMENT_NAME", "football_analytics"),
    "artifact_location": str(MODEL_DIR / "mlflow_artifacts")
}