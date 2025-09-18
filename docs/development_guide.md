# Football Analytics Platform - Development Guide

## Overview

This document provides detailed information for developers working on the Football Analytics Platform.

## Architecture

### System Components

1. **Data Layer**
   - Raw data ingestion from APIs and web scraping
   - Data validation and cleaning pipelines
   - Feature engineering and transformation
   - Storage in PostgreSQL with Redis caching

2. **Model Layer**
   - Multiple ML models (RF, XGBoost, Neural Networks)
   - Time series forecasting with Prophet/LSTM
   - Ensemble methods for improved accuracy
   - MLflow for experiment tracking and model versioning

3. **API Layer**
   - FastAPI for REST endpoints
   - Real-time prediction services
   - Authentication and rate limiting
   - Automated documentation with OpenAPI

4. **Presentation Layer**
   - Streamlit dashboard for interactive analysis
   - Real-time charts and visualizations
   - Player comparison tools
   - Match prediction interfaces

## Development Workflow

### Setting Up Development Environment

```bash
# 1. Clone and setup
git clone <repository-url>
cd football-analytics-2025
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# 3. Setup pre-commit hooks
pre-commit install

# 4. Configure environment
cp config/.env.example .env
# Edit .env with your API keys and settings
```

### Project Structure Details

```
src/
├── data/
│   ├── __init__.py
│   ├── collectors/
│   │   ├── __init__.py
│   │   ├── football_data_api.py      # Football-Data.org API client
│   │   ├── sportmonks_api.py         # Sportmonks API client
│   │   └── scraper.py                # Web scraping utilities
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── cleaner.py                # Data cleaning utilities
│   │   ├── validator.py              # Data validation rules
│   │   └── transformer.py            # Data transformation
│   └── database/
│       ├── __init__.py
│       ├── models.py                 # SQLAlchemy ORM models
│       ├── connection.py             # Database connection
│       └── migrations/               # Database migration scripts
├── features/
│   ├── __init__.py
│   ├── position_features.py          # Position-specific features
│   ├── form_features.py              # Form and rolling statistics
│   ├── context_features.py           # Match context features
│   └── feature_selector.py           # Feature selection utilities
├── models/
│   ├── __init__.py
│   ├── base_model.py                 # Base model interface
│   ├── traditional_models.py         # RF, XGBoost implementations
│   ├── neural_networks.py            # Deep learning models
│   ├── time_series.py                # Prophet, LSTM models
│   ├── ensemble.py                   # Ensemble methods
│   └── predictor.py                  # Main prediction service
├── api/
│   ├── __init__.py
│   ├── main.py                       # FastAPI application
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── predictions.py            # Prediction endpoints
│   │   ├── players.py                # Player data endpoints
│   │   └── matches.py                # Match data endpoints
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── auth.py                   # Authentication middleware
│   │   └── rate_limiting.py          # Rate limiting
│   └── schemas/
│       ├── __init__.py
│       ├── player.py                 # Player data schemas
│       └── prediction.py             # Prediction schemas
└── visualization/
    ├── __init__.py
    ├── dashboard.py                  # Main Streamlit dashboard
    ├── components/
    │   ├── __init__.py
    │   ├── player_comparison.py      # Player comparison widgets
    │   ├── match_prediction.py       # Match prediction interface
    │   └── performance_charts.py     # Performance visualization
    └── utils/
        ├── __init__.py
        ├── plotly_utils.py           # Plotly visualization helpers
        └── data_formatting.py       # Data formatting for charts
```

## Data Flow

### 1. Data Collection Pipeline

```python
# Example data collection flow
from src.data.collectors import FootballDataAPI, SportmonksAPI
from src.data.processors import DataCleaner, DataValidator

# Initialize collectors
fd_api = FootballDataAPI(api_key="your_key")
sm_api = SportmonksAPI(api_key="your_key")

# Collect data
matches = fd_api.get_matches(league_id="PL", season="2024")
player_stats = sm_api.get_player_statistics(season="2024")

# Process data
cleaner = DataCleaner()
validator = DataValidator()

clean_matches = cleaner.clean_match_data(matches)
validated_stats = validator.validate_player_data(player_stats)
```

### 2. Feature Engineering Pipeline

```python
from src.features import PositionFeatureEngineer, FormFeatureEngineer

# Create position-specific features
pos_engineer = PositionFeatureEngineer()
gk_features = pos_engineer.create_goalkeeper_features(gk_data)
att_features = pos_engineer.create_attacker_features(att_data)

# Create form features
form_engineer = FormFeatureEngineer()
form_features = form_engineer.create_rolling_features(player_data, windows=[3, 5, 10])
```

### 3. Model Training Pipeline

```python
from src.models import TraditionalModels, NeuralNetworkModel, EnsembleModel

# Train individual models
rf_model = TraditionalModels().train_random_forest(X_train, y_train)
xgb_model = TraditionalModels().train_xgboost(X_train, y_train)
nn_model = NeuralNetworkModel(input_dim=X_train.shape[1])
nn_model.train(X_train, y_train, X_val, y_val)

# Create ensemble
ensemble = EnsembleModel([rf_model, xgb_model, nn_model])
ensemble.optimize_weights(X_val, y_val)
```

## Testing Strategy

### Unit Tests

```python
# tests/test_feature_engineering.py
import pytest
from src.features import PositionFeatureEngineer

def test_goalkeeper_features():
    engineer = PositionFeatureEngineer()
    sample_data = {
        'saves': [10, 5, 8],
        'goals_conceded': [1, 2, 0],
        'matches_played': [1, 1, 1]
    }
    
    result = engineer.create_goalkeeper_features(pd.DataFrame(sample_data))
    
    assert 'save_percentage' in result.columns
    assert result['save_percentage'].iloc[0] == 10/11  # 10/(10+1)
```

### Integration Tests

```python
# tests/test_api_integration.py
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_prediction_endpoint():
    response = client.post("/predict/player_performance", json={
        "player_id": "123",
        "match_context": {...}
    })
    
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "confidence" in response.json()
```

### Performance Tests

```python
# tests/test_performance.py
import time
from src.models import EnsembleModel

def test_prediction_latency():
    model = EnsembleModel.load("models/saved/ensemble_v1.pkl")
    
    start_time = time.time()
    prediction = model.predict(sample_features)
    end_time = time.time()
    
    latency = end_time - start_time
    assert latency < 0.1  # Less than 100ms
```

## API Documentation

### Prediction Endpoints

#### Player Performance Prediction
```
POST /api/v1/predict/player_performance
```

Request Body:
```json
{
    "player_id": "string",
    "match_context": {
        "opposition_team": "string",
        "venue": "home|away",
        "competition": "string",
        "date": "2024-01-01"
    },
    "features": {
        "recent_form": "object",
        "fatigue_level": "number",
        "opposition_difficulty": "number"
    }
}
```

Response:
```json
{
    "prediction": {
        "rating": 7.5,
        "goals_probability": 0.15,
        "assists_probability": 0.25,
        "standout_probability": 0.72
    },
    "confidence": 0.84,
    "model_version": "ensemble_v1.2",
    "timestamp": "2024-01-01T10:00:00Z"
}
```

#### Match Outcome Prediction
```
POST /api/v1/predict/match_outcome
```

Request Body:
```json
{
    "home_team": "string",
    "away_team": "string",
    "date": "2024-01-01",
    "competition": "string"
}
```

Response:
```json
{
    "predictions": {
        "home_win": 0.45,
        "draw": 0.25,
        "away_win": 0.30
    },
    "expected_goals": {
        "home": 1.8,
        "away": 1.2
    },
    "confidence": 0.78
}
```

## Database Schema

### Core Tables

```sql
-- Players table
CREATE TABLE players (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    position VARCHAR(10) NOT NULL,
    team_id INTEGER REFERENCES teams(id),
    birth_date DATE,
    nationality VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Matches table
CREATE TABLE matches (
    id SERIAL PRIMARY KEY,
    home_team_id INTEGER REFERENCES teams(id),
    away_team_id INTEGER REFERENCES teams(id),
    competition VARCHAR(100),
    match_date TIMESTAMP,
    home_score INTEGER,
    away_score INTEGER,
    status VARCHAR(50)
);

-- Player performances table
CREATE TABLE player_performances (
    id SERIAL PRIMARY KEY,
    player_id INTEGER REFERENCES players(id),
    match_id INTEGER REFERENCES matches(id),
    rating DECIMAL(3,1),
    goals INTEGER DEFAULT 0,
    assists INTEGER DEFAULT 0,
    minutes_played INTEGER,
    -- Position-specific metrics
    saves INTEGER,
    tackles INTEGER,
    key_passes INTEGER,
    shots INTEGER
);
```

## Deployment

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/football_analytics
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: football_analytics
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine

  dashboard:
    build: .
    command: streamlit run src/visualization/dashboard.py --server.port=8501 --server.address=0.0.0.0
    ports:
      - "8501:8501"
    depends_on:
      - api

volumes:
  postgres_data:
```

## Monitoring and Observability

### Logging

```python
import logging
from config.config import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

def predict_player_performance(player_id, features):
    logger.info(f"Starting prediction for player {player_id}")
    
    try:
        prediction = model.predict(features)
        logger.info(f"Prediction completed: {prediction}")
        return prediction
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise
```

### Metrics Collection

```python
from prometheus_client import Counter, Histogram, start_http_server

prediction_counter = Counter('predictions_total', 'Total predictions made')
prediction_latency = Histogram('prediction_duration_seconds', 'Prediction latency')

@prediction_latency.time()
def make_prediction(features):
    prediction_counter.inc()
    return model.predict(features)
```

## Performance Optimization

### Caching Strategy

```python
import redis
import pickle
from functools import wraps

redis_client = redis.from_url(REDIS_CONFIG["url"])

def cache_result(expiration=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return pickle.loads(cached_result)
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            redis_client.setex(
                cache_key, 
                expiration, 
                pickle.dumps(result)
            )
            
            return result
        return wrapper
    return decorator

@cache_result(expiration=1800)  # 30 minutes
def get_player_features(player_id, date):
    # Expensive feature computation
    return compute_features(player_id, date)
```

### Database Optimization

```python
# Use database connection pooling
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_CONFIG["url"],
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True
)

# Use batch operations for bulk inserts
def bulk_insert_performances(performances):
    with engine.begin() as conn:
        conn.execute(
            PlayerPerformance.__table__.insert(),
            performances
        )
```

## Security Considerations

### API Authentication

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
import jwt

security = HTTPBearer()

def verify_token(token: str = Depends(security)):
    try:
        payload = jwt.decode(
            token.credentials,
            SECURITY_CONFIG["jwt_secret"],
            algorithms=[SECURITY_CONFIG["jwt_algorithm"]]
        )
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
```

### Input Validation

```python
from pydantic import BaseModel, validator

class PredictionRequest(BaseModel):
    player_id: str
    match_date: str
    features: dict
    
    @validator('player_id')
    def validate_player_id(cls, v):
        if not v or len(v) < 3:
            raise ValueError('Invalid player ID')
        return v
    
    @validator('features')
    def validate_features(cls, v):
        required_features = ['form_rating', 'opposition_strength']
        for feature in required_features:
            if feature not in v:
                raise ValueError(f'Missing required feature: {feature}')
        return v
```

This development guide provides the foundation for building and maintaining the Football Analytics Platform. For specific implementation details, refer to the inline documentation in each module.