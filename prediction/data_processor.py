#!/usr/bin/env python3
"""
Data Processing Module
Handles all data loading, cleaning, and preprocessing for FPL predictions
"""

import pandas as pd
import numpy as np
import requests
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles all FPL data processing operations"""

    def __init__(self):
        self.api_base = "https://fantasy.premierleague.com/api"
        self.cache = {}
        self.data_dir = Path("data")

    def load_bootstrap_data(self) -> Dict[str, Any]:
        """Load FPL bootstrap data"""
        # This method will be extracted from FPLPredictor
        logger.info("Loading bootstrap data...")

        # Find latest bootstrap file
        bootstrap_files = list(self.data_dir.glob("bootstrap_data_*.json"))
        if not bootstrap_files:
            raise FileNotFoundError("No bootstrap data files found")

        latest_file = max(bootstrap_files, key=lambda x: x.stat().st_mtime)

        with open(latest_file, 'r') as f:
            data = json.load(f)

        logger.info(f"✅ Loaded bootstrap data from {latest_file.name}")
        return data

    def load_player_data(self) -> pd.DataFrame:
        """Load and process player data"""
        # This method will be extracted from FPLPredictor
        logger.info("Loading player data...")

        # Placeholder - will be replaced with extracted method
        bootstrap_data = self.load_bootstrap_data()
        players_df = pd.DataFrame(bootstrap_data.get('elements', []))

        logger.info(f"✅ Loaded {len(players_df)} players")
        return players_df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data"""
        # This method will be extracted from FPLPredictor
        logger.info("Cleaning data...")

        # Basic cleaning
        df = df.dropna(subset=['id', 'web_name'])
        df = df.fillna(0)

        logger.info(f"✅ Cleaned data: {len(df)} records")
        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models"""
        # This method will be extracted from FPLPredictor
        logger.info("Preparing features...")

        # Placeholder feature engineering
        # This will be replaced with actual extracted method

        logger.info("✅ Features prepared")
        return df


    # TODO: Extract these methods from FPLPredictor:
# - _load_current_data
# - _load_models
# - _load_time_series_models
# - fetch_bootstrap_data
# - process_api_response
# - _prepare_features
# - clean_player_data
# - load_fixture_data
# - cache_data
