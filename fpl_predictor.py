#!/usr/bin/env python3
"""
FPL Predictor - Terminal Interface
Maximum Accuracy FPL Predictions for 2025/26 Season Planning

Usage:
    python fpl_predictor.py --help
    python fpl_predictor.py predict --player "Salah"
    python fpl_predictor.py top-picks --position MID --budget 8.0
    python fpl_predictor.py optimize-team --budget 100.0
    python fpl_predictor.py update-data
"""

import argparse
import sys
import os
import time
import pandas as pd
import numpy as np
import sqlite3
import joblib
from chip_strategy_analyzer import ChipStrategyAnalyzer
import requests
import json
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

class FPLPredictor:
    """Terminal-based FPL prediction system using maximum accuracy models"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.data_dir = Path(__file__).parent / 'data'
        self.models_dir = Path(__file__).parent / 'models' / 'production'
        self.db_path = self.data_dir / 'fpl_data.db'
        self.user_team_file = self.data_dir / 'current_team.json'
        
        # FPL API endpoints
        self.fpl_api_base = "https://fantasy.premierleague.com/api"
        self.bootstrap_url = f"{self.fpl_api_base}/bootstrap-static/"
        self.team_url = f"{self.fpl_api_base}/entry/{{team_id}}/event/{{event_id}}/picks/"
        self.user_url = f"{self.fpl_api_base}/entry/{{team_id}}/"
        
        # Initialize models and preprocessors
        self.model = None
        self.preprocessors = None
        self.feature_names = None
        
        if self.verbose:
            print("🔄 Initializing FPL Predictor with maximum accuracy models...")
        
        self._load_models()
    
    def _log(self, message):
        """Print message if verbose mode is enabled"""
        if self.verbose:
            print(message)
    
    def _load_models(self):
        """Load the trained XGBoost model and preprocessors"""
        try:
            # Find latest model files
            model_files = list(self.models_dir.glob('xgboost_best_*.joblib'))
            preprocessor_files = list(self.models_dir.glob('preprocessors_*.joblib'))
            
            if not model_files or not preprocessor_files:
                raise FileNotFoundError("No trained models found. Run the training notebook first.")
            
            # Load latest models
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            latest_preprocessor = max(preprocessor_files, key=lambda x: x.stat().st_mtime)
            
            self.model = joblib.load(latest_model)
            self.preprocessors = joblib.load(latest_preprocessor)
            self.feature_names = self.preprocessors['feature_names']
            
            self._log(f"✅ Loaded model: {latest_model.name}")
            self._log(f"✅ Loaded preprocessors: {latest_preprocessor.name}")
            self._log(f"📊 Features: {len(self.feature_names)}")
            
            # Validate loaded models
            if not self._validate_model_integrity():
                print("❌ Model validation failed!")
                sys.exit(1)
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("💡 Run the training notebook first to create models")
            sys.exit(1)
    
    def _validate_model_integrity(self):
        """Comprehensive model validation to ensure models are working correctly"""
        try:
            print("🔍 Validating model integrity...")
            
            # 1. Check model object validity
            if not hasattr(self.model, 'predict'):
                print("❌ Model does not have predict method")
                return False
            
            # 2. Check preprocessors integrity
            if not isinstance(self.preprocessors, dict):
                print("❌ Preprocessors not in expected format")
                return False
            
            required_preprocessor_keys = ['feature_names']
            for key in required_preprocessor_keys:
                if key not in self.preprocessors:
                    print(f"❌ Missing preprocessor key: {key}")
                    return False
            
            # 3. Check feature names validity
            if not self.feature_names or len(self.feature_names) == 0:
                print("❌ No feature names found")
                return False
            
            # 4. Test model prediction with synthetic data
            if not self._test_model_prediction():
                print("❌ Model prediction test failed")
                return False
            
            # 5. Validate model performance against known benchmarks
            if not self._validate_model_performance():
                print("❌ Model performance validation failed")
                return False
            
            print("✅ Model validation passed")
            return True
            
        except Exception as e:
            print(f"❌ Model validation error: {e}")
            return False
    
    def _test_model_prediction(self):
        """Test model prediction with synthetic data to ensure it works"""
        try:
            print("   🧪 Testing model prediction...")
            
            # Create synthetic test data with expected features
            n_features = len(self.feature_names)
            synthetic_data = np.random.rand(5, n_features)
            
            # Create DataFrame with correct column names
            test_df = pd.DataFrame(synthetic_data, columns=self.feature_names)
            
            # Test prediction
            predictions = self.model.predict(test_df)
            
            # Validate prediction output
            if not isinstance(predictions, np.ndarray):
                print("❌ Predictions not in expected numpy array format")
                return False
            
            if len(predictions) != 5:
                print("❌ Prediction length mismatch")
                return False
            
            # Check for reasonable prediction values (FPL points should be 0-30 range typically)
            if np.any(predictions < -5) or np.any(predictions > 50):
                print(f"❌ Predictions outside reasonable range: {predictions}")
                return False
            
            # Check for NaN or infinite values
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                print("❌ Predictions contain NaN or infinite values")
                return False
            
            print(f"   ✅ Model prediction test passed (sample prediction: {predictions[0]:.2f})")
            return True
            
        except Exception as e:
            print(f"❌ Model prediction test failed: {e}")
            return False
    
    def _validate_model_performance(self):
        """Validate model performance against expected benchmarks"""
        try:
            print("   📊 Validating model performance...")
            
            # Load model metadata if available
            metadata_files = list(self.models_dir.glob('model_metadata_*.json'))
            if not metadata_files:
                print("   ⚠️ No model metadata found - skipping performance validation")
                return True
            
            # Load latest metadata
            latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
            with open(latest_metadata, 'r') as f:
                metadata = json.load(f)
            
            # Check model performance metrics
            model_performance = metadata.get('model_performance', {})
            if not model_performance:
                print("   ⚠️ No performance metrics in metadata")
                return True
            
            # Validate RMSE is reasonable (FPL context: should be < 2.0 for good models)
            for model_name, metrics in model_performance.items():
                rmse = metrics.get('train_rmse', float('inf'))
                mae = metrics.get('train_mae', float('inf'))
                
                if rmse > 2.0:
                    print(f"   ⚠️ High RMSE for {model_name}: {rmse:.3f}")
                
                if mae > 1.5:
                    print(f"   ⚠️ High MAE for {model_name}: {mae:.3f}")
                
                # Check for reasonable performance metrics
                if rmse < 0 or mae < 0:
                    print(f"❌ Invalid performance metrics for {model_name}")
                    return False
            
            print("   ✅ Model performance validation passed")
            return True
            
        except Exception as e:
            print(f"❌ Model performance validation failed: {e}")
            return False
    
    def _validate_prediction_output(self, predictions, player_data):
        """Validate prediction output for reasonableness and consistency"""
        try:
            if predictions is None:
                print("❌ Predictions are None")
                return False
            
            if len(predictions) == 0:
                print("❌ No predictions generated")
                return False
            
            # Check for required columns
            required_cols = ['web_name', 'predicted_points', 'position_name']
            missing_cols = [col for col in required_cols if col not in predictions.columns]
            if missing_cols:
                print(f"❌ Missing prediction columns: {missing_cols}")
                return False
            
            # Validate prediction ranges
            points = predictions['predicted_points'].values
            
            if np.any(np.isnan(points)):
                print("❌ Predictions contain NaN values")
                return False
            
            if np.any(np.isinf(points)):
                print("❌ Predictions contain infinite values")
                return False
            
            # FPL-specific validation: points should be reasonable
            if np.any(points < -2) or np.any(points > 30):
                outliers = predictions[(predictions['predicted_points'] < -2) | 
                                    (predictions['predicted_points'] > 30)]
                print(f"❌ Unreasonable predictions found: {len(outliers)} players")
                print(f"   Range: {points.min():.2f} to {points.max():.2f}")
                return False
            
            # Check for prediction distribution (should not be all the same)
            if np.std(points) < 0.1:
                print("❌ Predictions have very low variance - model may not be working")
                return False
            
            print(f"✅ Prediction validation passed for {len(predictions)} players")
            return True
            
        except Exception as e:
            print(f"❌ Prediction validation failed: {e}")
            return False
    
    def _run_model_health_check(self):
        """Run comprehensive model health check"""
        try:
            print("\n🏥 Running Model Health Check...")
            print("=" * 40)
            
            health_status = {
                'model_loaded': False,
                'preprocessors_loaded': False,
                'prediction_test': False,
                'performance_check': False,
                'data_compatibility': False
            }
            
            # 1. Check model loading
            if hasattr(self, 'model') and self.model is not None:
                health_status['model_loaded'] = True
                print("✅ Model loaded successfully")
            else:
                print("❌ Model not loaded")
                return health_status
            
            # 2. Check preprocessors
            if hasattr(self, 'preprocessors') and self.preprocessors is not None:
                health_status['preprocessors_loaded'] = True
                print("✅ Preprocessors loaded successfully")
            else:
                print("❌ Preprocessors not loaded")
                return health_status
            
            # 3. Test prediction capability
            health_status['prediction_test'] = self._test_model_prediction()
            
            # 4. Check performance metrics
            health_status['performance_check'] = self._validate_model_performance()
            
            # 5. Check data compatibility
            try:
                current_data = self._load_current_data()
                if current_data is not None and len(current_data) > 0:
                    health_status['data_compatibility'] = True
                    print("✅ Data compatibility check passed")
                else:
                    print("❌ Data compatibility check failed")
            except Exception as e:
                print(f"❌ Data compatibility error: {e}")
            
            # Overall health assessment
            passed_checks = sum(health_status.values())
            total_checks = len(health_status)
            health_score = (passed_checks / total_checks) * 100
            
            print(f"\n📊 Model Health Score: {health_score:.1f}% ({passed_checks}/{total_checks} checks passed)")
            
            if health_score >= 80:
                print("🟢 Model health: EXCELLENT")
            elif health_score >= 60:
                print("🟡 Model health: GOOD - some issues detected")
            else:
                print("🔴 Model health: POOR - immediate attention needed")
            
            return health_status
            
        except Exception as e:
            print(f"❌ Model health check failed: {e}")
            return {'error': str(e)}
    
    def _safe_predict(self, X, validate_output=True):
        """Safe prediction wrapper with validation"""
        try:
            # Pre-prediction validation
            if X is None or len(X) == 0:
                print("❌ No input data for prediction")
                return None
            
            # Check feature compatibility
            if len(X.columns) != len(self.feature_names):
                print(f"❌ Feature count mismatch: got {len(X.columns)}, expected {len(self.feature_names)}")
                return None
            
            # Check for missing features
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                print(f"❌ Missing features: {missing_features}")
                return None
            
            # Reorder features to match training order
            X_ordered = X[self.feature_names]
            
            # Check for invalid values
            if X_ordered.isnull().any().any():
                print("⚠️ Null values detected in input - filling with 0")
                X_ordered = X_ordered.fillna(0)
            
            if np.isinf(X_ordered.values).any():
                print("❌ Infinite values detected in input")
                return None
            
            # Make prediction
            predictions = self.model.predict(X_ordered)
            
            # Post-prediction validation
            if validate_output:
                if not self._validate_raw_predictions(predictions):
                    print("❌ Prediction output validation failed")
                    return None
            
            return predictions
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return None
    
    def _validate_raw_predictions(self, predictions):
        """Validate raw prediction output"""
        try:
            if predictions is None:
                return False
            
            if not isinstance(predictions, np.ndarray):
                return False
            
            if len(predictions) == 0:
                return False
            
            # Check for NaN or infinite values
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                return False
            
            # Check reasonable range for FPL points (-2 to 30)
            if np.any(predictions < -2) or np.any(predictions > 30):
                outlier_count = np.sum((predictions < -2) | (predictions > 30))
                print(f"⚠️ {outlier_count} predictions outside reasonable range")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Raw prediction validation error: {e}")
            return False
    
    def _load_current_data(self):
        """Load current FPL data with performance history integration"""
        try:
            # Check for cached data first (updated by update_data command)
            players_file = self.data_dir / 'players_latest.csv'
            
            if players_file.exists():
                print("🔄 Loading cached FPL data...")
                players_df = pd.read_csv(players_file)
            else:
                print("🔄 Fetching live FPL data...")
                
                import requests
                
                # Get bootstrap data (general info, teams, players)
                bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
                response = requests.get(bootstrap_url)
                bootstrap_data = response.json()
                
                # Extract player data
                players = bootstrap_data['elements']
                teams = {team['id']: team['short_name'] for team in bootstrap_data['teams']}
                
                # Convert to DataFrame
                players_df = pd.DataFrame(players)
                
                # Add team names
                players_df['team'] = players_df['team'].map(teams)
                
                # Map important columns
                players_df['web_name'] = players_df['web_name']
                players_df['now_cost'] = players_df['now_cost']  # Already in 0.1m units
                players_df['total_points'] = players_df['total_points']
                players_df['form'] = players_df['form'].astype(float)
                players_df['points_per_game'] = players_df['points_per_game'].astype(float)
                players_df['selected_by_percent'] = players_df['selected_by_percent'].astype(float)
                players_df['element_type'] = players_df['element_type']  # 1=GKP, 2=DEF, 3=MID, 4=FWD
                
                # Add position names
                position_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
                players_df['position'] = players_df['element_type'].map(position_map)
                
                # Add recent performance metrics
                players_df['minutes'] = players_df['minutes']
            
            # Try to enhance with performance history data
            performance_features_file = self.data_dir / 'performance_history' / 'model_features.csv'
            
            if performance_features_file.exists():
                print("📈 Enhancing with performance history features...")
                performance_df = pd.read_csv(performance_features_file)
                
                # Merge performance features with base player data
                enhanced_df = players_df.merge(
                    performance_df,
                    left_on='id',
                    right_on='player_id',
                    how='left',
                    suffixes=('', '_perf')
                )
                
                # Fill missing performance features with defaults
                performance_cols = [col for col in performance_df.columns if col.startswith(('avg_', 'trend_'))]
                for col in performance_cols:
                    if col in enhanced_df.columns:
                        enhanced_df[col] = enhanced_df[col].fillna(0)
                
                print(f"✅ Enhanced dataset: {len(enhanced_df)} players with {len(performance_cols)} performance features")
                return enhanced_df
            else:
                print("⚠️ Performance history not found, using basic features only")
                print("💡 Run 'python fpl_predictor.py collect-history --save' to enable enhanced predictions")
                return players_df
            df['goals_scored'] = df['goals_scored']
            df['assists'] = df['assists']
            df['clean_sheets'] = df['clean_sheets']
            df['goals_conceded'] = df['goals_conceded']
            df['own_goals'] = df['own_goals']
            df['penalties_saved'] = df['penalties_saved']
            df['penalties_missed'] = df['penalties_missed']
            df['yellow_cards'] = df['yellow_cards']
            df['red_cards'] = df['red_cards']
            df['saves'] = df['saves']
            df['bonus'] = df['bonus']
            df['bps'] = df['bps']  # Bonus Points System
            df['influence'] = df['influence'].astype(float)
            df['creativity'] = df['creativity'].astype(float)
            df['threat'] = df['threat'].astype(float)
            df['ict_index'] = df['ict_index'].astype(float)
            
            # Calculate value metrics
            df['value_form'] = df['form'] / (df['now_cost'] / 10)
            df['value_total'] = df['total_points'] / (df['now_cost'] / 10)
            
            # Filter out unavailable players
            df = df[df['status'] != 'u'].copy()  # Remove unavailable players
            
            self._log(f"📊 Loaded {len(df)} players from live FPL API")
            print(f"💰 Price range: £{(df['now_cost'].min()/10):.1f}m - £{(df['now_cost'].max()/10):.1f}m")
            print(f"🏆 Top scorer: {df.loc[df['total_points'].idxmax(), 'web_name']} ({df['total_points'].max()} pts)")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading live data: {e}")
            print("� Falling back to CSV data...")
            
            # Fallback to CSV if API fails
            try:
                csv_path = self.data_dir / 'enhanced_fpl_features.csv'
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    latest_gameweek = df['gameweek'].max()
                    df = df[df['gameweek'] == latest_gameweek].copy()
                    self._log(f"📊 Loaded {len(df)} players from CSV (fallback)")
                    return df
            except:
                pass
            
            print("❌ No data sources available")
            return pd.DataFrame()
    
    def _prepare_features(self, df):
        """Prepare features for prediction using the same preprocessing as training"""
        try:
            # Get features in the same order as training
            available_features = [col for col in self.feature_names if col in df.columns]
            
            if len(available_features) != len(self.feature_names):
                missing = set(self.feature_names) - set(available_features)
                self._log(f"⚠️ Missing features: {missing}")
                
                # Fill missing features with 0 or median values
                for feature in missing:
                    df[feature] = 0
            
            # Select features in correct order
            X = df[self.feature_names].copy()
            
            # Handle categorical features
            label_encoders = self.preprocessors.get('label_encoders', {})
            for col, encoder in label_encoders.items():
                if col in X.columns:
                    # Handle unknown categories
                    X[col] = X[col].fillna('Unknown')
                    unknown_mask = ~X[col].isin(encoder.classes_)
                    if unknown_mask.any():
                        X.loc[unknown_mask, col] = encoder.classes_[0]  # Use first class as default
                    X[col] = encoder.transform(X[col])
            
            # Fill remaining missing values
            X = X.fillna(X.median())
            
            return X.astype(float)
            
        except Exception as e:
            print(f"❌ Error preparing features: {e}")
            return None
    
    def predict_player(self, player_name):
        """Predict points for a specific player"""
        df = self._load_current_data()
        if df.empty:
            return None
        
        # Find player
        player_matches = df[df['web_name'].str.contains(player_name, case=False, na=False)]
        
        if len(player_matches) == 0:
            print(f"❌ Player '{player_name}' not found")
            print("💡 Try searching with partial names, e.g., 'Salah' instead of 'Mohamed Salah'")
            return None
        
        if len(player_matches) > 1:
            print(f"🔍 Multiple players found for '{player_name}':")
            for idx, player in player_matches.iterrows():
                print(f"   • {player['web_name']} ({player.get('team', 'Unknown team')})")
            return None
        
        player = player_matches.iloc[0]
        
        # Prepare features and predict
        X = self._prepare_features(player_matches)
        if X is None:
            return None
        
        prediction = self.model.predict(X)[0]
        
        # Display results
        print(f"\n🎯 Prediction for {player['web_name']}:")
        print(f"   • Predicted Points: {prediction:.2f}")
        print(f"   • Current Price: £{player.get('now_cost', 0)/10:.1f}m")
        print(f"   • Position: {self._get_position_name(player.get('element_type', 0))}")
        print(f"   • Recent Form: {player.get('form', 0)}")
        print(f"   • Value Rating: {prediction / (player.get('now_cost', 1)/10):.2f} pts/£m")
        
        return {
            'player': player['web_name'],
            'prediction': prediction,
            'price': player.get('now_cost', 0)/10,
            'position': self._get_position_name(player.get('element_type', 0))
        }
    
    def predict_player_by_name(self, player_name, verbose=True):
        """Predict points for a specific player (helper method for internal use)"""
        df = self._load_current_data()
        if df.empty:
            return None
        
        # Find player
        player_matches = df[df['web_name'].str.contains(player_name, case=False, na=False)]
        
        if len(player_matches) != 1:
            return None
        
        player = player_matches.iloc[0]
        
        # Prepare features and predict
        X = self._prepare_features(player_matches)
        if X is None:
            return None
        
        predictions = self._safe_predict(X)
        if predictions is None:
            print(f"❌ Prediction failed for player: {player['web_name']}")
            return None
            
        prediction = predictions[0]
        
        result = {
            'player': player['web_name'],
            'team': player.get('team', 'Unknown'),
            'position': self._get_position_name(player.get('element_type', 0)),
            'price': player.get('now_cost', 0)/10,
            'predicted_points': prediction,
            'value_rating': prediction/(player.get('now_cost', 1)/10)
        }
        
        return result
    
    def top_picks(self, position=None, max_price=None, limit=10):
        """Get top predicted players for a position and budget"""
        df = self._load_current_data()
        if df.empty:
            return None
        
        # Filter by position
        if position:
            pos_map = {'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
            if position.upper() in pos_map:
                df = df[df['element_type'] == pos_map[position.upper()]]
            else:
                print(f"❌ Invalid position '{position}'. Use: GKP, DEF, MID, FWD")
                return None
        
        # Filter by price
        if max_price:
            df = df[df['now_cost'] <= max_price * 10]  # Convert to FPL price format
        
        # Prepare features and predict
        X = self._prepare_features(df)
        if X is None:
            return None
        
        predictions = self._safe_predict(X)
        if predictions is None:
            print("❌ Prediction failed for top picks")
            return None
            
        df['predicted_points'] = predictions
        df['value_rating'] = predictions / (df['now_cost'] / 10)
        
        # Sort by predicted points
        top_players = df.nlargest(limit, 'predicted_points')
        
        # Display results
        pos_str = f" {position.upper()}" if position else ""
        price_str = f" under £{max_price}m" if max_price else ""
        print(f"\n🏆 Top{pos_str} Picks{price_str}:")
        print(f"{'Rank':<4} {'Player':<20} {'Pred':<6} {'Price':<7} {'Value':<6} {'Form':<6}")
        print("=" * 60)
        
        for i, (_, player) in enumerate(top_players.iterrows(), 1):
            print(f"{i:<4} {player['web_name']:<20} {player['predicted_points']:<6.2f} "
                  f"£{player['now_cost']/10:<6.1f} {player['value_rating']:<6.2f} {player.get('form', 0):<6}")
        
        return top_players[['web_name', 'predicted_points', 'now_cost', 'value_rating']].to_dict('records')
    
    def optimize_team(self, budget=100.0):
        """Simple team optimization based on predictions"""
        df = self._load_current_data()
        if df.empty:
            return None
        
        # Prepare features and predict
        X = self._prepare_features(df)
        if X is None:
            return None
        
        predictions = self._safe_predict(X)
        if predictions is None:
            print("❌ Prediction failed for team optimization")
            return None
            
        df['predicted_points'] = predictions
        df['value_rating'] = predictions / (df['now_cost'] / 10)
        
        # Simple greedy optimization by position
        team = []
        remaining_budget = budget
        
        # Formation: 1 GKP, 3 DEF, 4 MID, 3 FWD (11 players)
        formation = {'GKP': 1, 'DEF': 3, 'MID': 4, 'FWD': 3}
        pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        
        for pos_id, pos_name in pos_map.items():
            pos_players = df[df['element_type'] == pos_id].copy()
            pos_players = pos_players[pos_players['now_cost'] <= remaining_budget * 10]
            pos_players = pos_players.nlargest(formation[pos_name], 'value_rating')
            
            for _, player in pos_players.iterrows():
                if remaining_budget >= player['now_cost'] / 10:
                    team.append({
                        'player': player['web_name'],
                        'position': pos_name,
                        'price': player['now_cost'] / 10,
                        'predicted_points': player['predicted_points'],
                        'value_rating': player['value_rating']
                    })
                    remaining_budget -= player['now_cost'] / 10
        
        # Display optimized team
        total_cost = sum(p['price'] for p in team)
        total_predicted = sum(p['predicted_points'] for p in team)
        
        print(f"\n⚽ Optimized Team (Budget: £{budget}m):")
        print(f"{'Pos':<4} {'Player':<20} {'Price':<7} {'Pred':<6} {'Value':<6}")
        print("=" * 50)
        
        for player in team:
            print(f"{player['position']:<4} {player['player']:<20} "
                  f"£{player['price']:<6.1f} {player['predicted_points']:<6.2f} {player['value_rating']:<6.2f}")
        
        print("=" * 50)
        print(f"Total Cost: £{total_cost:.1f}m")
        print(f"Remaining: £{budget - total_cost:.1f}m")
        print(f"Predicted Total: {total_predicted:.1f} points")
        
        return team
    
    def analyze_fixtures(self, start_gw=5, end_gw=10):
        """Analyze fixture difficulty using official FPL Fixture Difficulty Ratings (FDR)"""
        print(f"📅 Official FPL Fixture Difficulty Analysis - Round {start_gw}")
        if end_gw > start_gw:
            print(f"📊 Analyzing gameweeks {start_gw}-{end_gw}")
        else:
            print(f"📊 Analyzing gameweek {start_gw}")
        
        try:
            # Step 1: Fetch real FPL fixture data
            import requests
            
            # Get team mapping
            bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
            response = requests.get(bootstrap_url)
            bootstrap_data = response.json()
            
            # Create team mapping {id: short_name}
            team_mapping = {}
            for team in bootstrap_data['teams']:
                team_mapping[team['id']] = team['short_name']
            
            # Step 2: Fetch fixtures for the specified gameweeks
            team_fdr_analysis = {}
            
            for gw in range(start_gw, end_gw + 1):
                fixtures_url = f"https://fantasy.premierleague.com/api/fixtures/?event={gw}"
                response = requests.get(fixtures_url)
                fixtures_data = response.json()
                
                for fixture in fixtures_data:
                    if fixture['event'] == gw:
                        home_team = team_mapping.get(fixture['team_h'], 'UNK')
                        away_team = team_mapping.get(fixture['team_a'], 'UNK')
                        
                        # Initialize team data if not exists
                        if home_team not in team_fdr_analysis:
                            team_fdr_analysis[home_team] = {'fdr_ratings': [], 'fixtures': []}
                        if away_team not in team_fdr_analysis:
                            team_fdr_analysis[away_team] = {'fdr_ratings': [], 'fixtures': []}
                        
                        # Store FDR ratings (1-5 scale from FPL)
                        team_fdr_analysis[home_team]['fdr_ratings'].append(fixture['team_h_difficulty'])
                        team_fdr_analysis[home_team]['fixtures'].append(f"vs {away_team} (H)")
                        
                        team_fdr_analysis[away_team]['fdr_ratings'].append(fixture['team_a_difficulty'])
                        team_fdr_analysis[away_team]['fixtures'].append(f"@ {home_team} (A)")
            
            # Step 3: Calculate average FDR and categorize
            team_analysis = []
            
            for team, data in team_fdr_analysis.items():
                if len(data['fdr_ratings']) == 0:
                    continue
                
                avg_fdr = sum(data['fdr_ratings']) / len(data['fdr_ratings'])
                total_fixtures = len(data['fdr_ratings'])
                
                # Official FPL FDR scale: 1=Very Easy, 2=Easy, 3=Medium, 4=Hard, 5=Very Hard
                if avg_fdr <= 2.0:
                    difficulty_text = 'Easy'
                    color_indicator = '🟢'
                elif avg_fdr <= 3.0:
                    difficulty_text = 'Medium'  
                    color_indicator = '🟡'
                elif avg_fdr <= 4.0:
                    difficulty_text = 'Hard'
                    color_indicator = '🟠'
                else:
                    difficulty_text = 'Very Hard'
                    color_indicator = '🔴'
                
                team_analysis.append({
                    'team': team,
                    'avg_fdr': avg_fdr,
                    'total_fixtures': total_fixtures,
                    'difficulty': difficulty_text,
                    'color': color_indicator,
                    'fixtures_list': ', '.join(data['fixtures'][:3])  # Show first 3 fixtures
                })
            
            # Sort by FDR (lower = easier)
            team_analysis.sort(key=lambda x: x['avg_fdr'])
            
            # Display results
            print(f"\n🏆 Official FPL Fixture Difficulty Ratings (GW{start_gw}")
            if end_gw > start_gw:
                print(f"    to GW{end_gw}):")
            else:
                print("):")
            print(f"{'Rank':<4} {'Team':<4} {'Avg FDR':<8} {'Fixtures':<2} {'Difficulty':<12} {'Next Fixtures':<25}")
            print("=" * 85)
            
            for i, team_data in enumerate(team_analysis, 1):
                print(f"{i:<4} {team_data['team']:<4} {team_data['avg_fdr']:<8.1f} "
                      f"{team_data['total_fixtures']:<8} {team_data['color']} {team_data['difficulty']:<8} "
                      f"{team_data['fixtures_list']:<25}")
            
            # Convert to DataFrame for compatibility
            df_data = []
            for team_data in team_analysis:
                df_data.append({
                    'team': team_data['team'],
                    'avg_fdr': team_data['avg_fdr'],
                    'fixture_difficulty': team_data['difficulty'],
                    'total_fixtures': team_data['total_fixtures']
                })
            
            return pd.DataFrame(df_data)
            
        except Exception as e:
            print(f"❌ Error fetching FPL fixture data: {e}")
            print("🔄 Falling back to basic analysis...")
            
            # Fallback to basic team strength analysis
            df = self._load_current_data()
            if df.empty:
                return None
            
            # Simple team strength calculation as fallback
            team_analysis = []
            for team in df['team'].unique():
                team_players = df[df['team'] == team].copy()
                if len(team_players) > 0:
                    X = self._prepare_features(team_players)
                    if X is not None:
                        predictions = self.model.predict(X)
                        avg_strength = predictions.mean()
                        team_analysis.append({
                            'team': team,
                            'avg_fdr': 3.0,  # Default medium
                            'fixture_difficulty': 'Medium',
                            'total_fixtures': end_gw - start_gw + 1
                        })
            
            return pd.DataFrame(team_analysis)
    
    def fixture_based_picks(self, gameweeks="5-10", position=None, limit=10):
        """Get player picks based on fixture difficulty"""
        try:
            start_gw, end_gw = map(int, gameweeks.split('-'))
        except:
            print("❌ Invalid gameweek format. Use format like '5-10'")
            return None
        
        print(f"🎯 Best picks for easy fixtures (GW{start_gw}-{end_gw}):")
        
        # Get team difficulty analysis
        team_analysis = self.analyze_fixtures(start_gw, end_gw)
        if team_analysis is None:
            return None
        
        # Get teams with easy or medium fixtures (most favorable)
        favorable_teams = team_analysis[
            team_analysis['fixture_difficulty'].isin(['Easy', 'Medium'])
        ]['team'].tolist()
        
        # If no easy/medium teams, take the best available
        if not favorable_teams:
            favorable_teams = team_analysis.head(10)['team'].tolist()
        
        difficulty_text = "favorable" if len(favorable_teams) > 0 else "available"
        print(f"\n✅ Teams with most {difficulty_text} fixtures: {', '.join(favorable_teams[:6])}")
        
        # Load player data and filter by favorable fixture teams
        df = self._load_current_data()
        if df.empty:
            return None
        
        favorable_fixture_players = df[df['team'].isin(favorable_teams)].copy()
        
        # Filter by position if specified
        if position:
            pos_map = {'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
            if position.upper() in pos_map:
                favorable_fixture_players = favorable_fixture_players[
                    favorable_fixture_players['element_type'] == pos_map[position.upper()]
                ]
        
        if len(favorable_fixture_players) == 0:
            print(f"❌ No players found with {difficulty_text} fixtures for the specified criteria")
            return None
        
        # Get predictions
        X = self._prepare_features(favorable_fixture_players)
        if X is None:
            return None
        
        predictions = self.model.predict(X)
        favorable_fixture_players['predicted_points'] = predictions
        favorable_fixture_players['value_rating'] = predictions / (favorable_fixture_players['now_cost'] / 10)
        
        # Sort by predicted points
        top_favorable_picks = favorable_fixture_players.nlargest(limit, 'predicted_points')
        
        pos_str = f" {position.upper()}" if position else ""
        print(f"\n🌟 Top{pos_str} Picks (Favorable Fixtures GW{start_gw}-{end_gw}):")
        print(f"{'Rank':<4} {'Player':<18} {'Team':<5} {'Pred':<6} {'Price':<7} {'Value':<6}")
        print("=" * 55)
        
        for i, (_, player) in enumerate(top_favorable_picks.iterrows(), 1):
            print(f"{i:<4} {player['web_name']:<18} {player['team']:<5} "
                  f"{player['predicted_points']:<6.2f} £{player['now_cost']/10:<6.1f} "
                  f"{player['value_rating']:<6.2f}")
        
        return top_favorable_picks[['web_name', 'team', 'predicted_points', 'now_cost', 'value_rating']].to_dict('records')

    def fetch_current_team(self, team_id):
        """Fetch current team data from FPL API"""
        try:
            # Get current gameweek
            bootstrap_data = requests.get(self.bootstrap_url).json()
            current_event = None
            
            for event in bootstrap_data['events']:
                if event['is_current']:
                    current_event = event['id']
                    break
            
            if not current_event:
                # If no current event, get the next event
                for event in bootstrap_data['events']:
                    if event['is_next']:
                        current_event = event['id']
                        break
            
            if not current_event:
                current_event = 1  # Fallback to GW1
            
            # Get team picks
            team_picks_url = self.team_url.format(team_id=team_id, event_id=current_event)
            picks_response = requests.get(team_picks_url)
            
            if picks_response.status_code != 200:
                raise Exception(f"Failed to fetch team data: {picks_response.status_code}")
            
            picks_data = picks_response.json()
            
            # Get player details from bootstrap data
            players = {player['id']: player for player in bootstrap_data['elements']}
            teams = {team['id']: team for team in bootstrap_data['teams']}
            
            # Build current team info
            current_team = {
                'team_id': team_id,
                'gameweek': current_event,
                'picks': [],
                'bank': picks_data.get('entry_history', {}).get('bank', 0) / 10,  # Convert to millions
                'total_transfers': picks_data.get('entry_history', {}).get('event_transfers', 0),
                'fetched_at': datetime.now().isoformat()
            }
            
            for pick in picks_data['picks']:
                player = players[pick['element']]
                team = teams[player['team']]
                
                player_info = {
                    'element_id': pick['element'],
                    'web_name': player['web_name'],
                    'full_name': f"{player['first_name']} {player['second_name']}",
                    'team': team['short_name'],
                    'position': player['element_type'],
                    'position_name': self._get_position_name(player['element_type']),
                    'now_cost': player['now_cost'],
                    'selected_by_percent': player['selected_by_percent'],
                    'total_points': player['total_points'],
                    'is_captain': pick['is_captain'],
                    'is_vice_captain': pick['is_vice_captain'],
                    'multiplier': pick['multiplier'],
                    'is_starter': pick['position'] <= 11
                }
                current_team['picks'].append(player_info)
            
            # Save to file
            os.makedirs(self.data_dir, exist_ok=True)
            with open(self.user_team_file, 'w') as f:
                json.dump(current_team, f, indent=2)
            
            self._log(f"✅ Fetched current team for manager {team_id} (GW{current_event})")
            self._log(f"💰 Bank: £{current_team['bank']}m")
            self._log(f"🔄 Transfers made: {current_team['total_transfers']}")
            
            return current_team
            
        except Exception as e:
            print(f"❌ Error fetching current team: {e}")
            return None
    
    def load_current_team(self):
        """Load current team from saved file"""
        try:
            if self.user_team_file.exists():
                with open(self.user_team_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"❌ Error loading current team: {e}")
            return None
    
    def analyze_current_team(self, team_id=None):
        """Analyze current team with live predictions and proper squad identification"""
        if team_id:
            current_team = self.fetch_current_team(team_id)
        else:
            current_team = self.load_current_team()
        
        if not current_team:
            print("❌ No current team data available. Use --team-id to fetch from FPL API")
            return None
        
        print(f"\n📋 Current Team Analysis (GW{current_team['gameweek']})")
        print(f"💰 Bank: £{current_team['bank']}m")
        print(f"🔄 Transfers made this GW: {current_team['total_transfers']}")
        print("=" * 70)
        
        # Load live player data for accurate predictions
        df = self._load_current_data()
        if df.empty:
            print("❌ No player data available for predictions")
            return current_team
        
        # Create lookup for quick player matching
        player_lookup = {}
        for _, player in df.iterrows():
            player_lookup[player['id']] = player
            player_lookup[player['web_name']] = player
        
        # Analyze each player with proper predictions
        total_predicted = 0
        starters_predicted = 0
        bench_predicted = 0
        
        starting_xi = []
        bench_players = []
        
        for pick in current_team['picks']:
            # Get player data using element_id first, then fallback to name
            player_data = None
            if pick['element_id'] in player_lookup:
                player_data = player_lookup[pick['element_id']]
            elif pick['web_name'] in player_lookup:
                player_data = player_lookup[pick['web_name']]
            
            if player_data is not None:
                # Get realistic prediction based on form, total points, etc.
                form = float(player_data.get('form', 0))
                ppg = float(player_data.get('points_per_game', 0))
                total_pts = float(player_data.get('total_points', 0))
                
                # Simple but realistic prediction: combine form and points per game
                # Form is more recent, PPG is season average, total gives context
                if total_pts > 0 and ppg > 0:
                    # Weight recent form more heavily
                    prediction = (form * 0.6) + (ppg * 0.4)
                    # Add bonus for consistent performers
                    if ppg > 4:
                        prediction += 1
                else:
                    prediction = form if form > 0 else 1.0
                
                pick['predicted_points'] = round(prediction, 1)
                pick['form'] = form
                pick['ppg'] = ppg
                pick['total_points_season'] = total_pts
                pick['element_type'] = int(player_data.get('element_type', 0))  # Add missing field
                
                total_predicted += prediction
                
                if pick['is_starter']:
                    starters_predicted += prediction
                    starting_xi.append(pick)
                else:
                    bench_predicted += prediction
                    bench_players.append(pick)
            else:
                pick['predicted_points'] = 0
                pick['form'] = 0
                pick['ppg'] = 0
                pick['element_type'] = 0  # Default element_type if no player data found
                if pick['is_starter']:
                    starting_xi.append(pick)
                else:
                    bench_players.append(pick)
            
            # Display player info with accurate prices and predictions
            pos = pick['position_name'][:3]
            starter_indicator = "⭐" if pick['is_starter'] else "🪑"
            captain_indicator = "(C)" if pick['is_captain'] else "(VC)" if pick['is_vice_captain'] else ""
            
            print(f"{starter_indicator} {pos:<3} {pick['web_name']:<18} {pick['team']:<3} "
                  f"£{pick['now_cost']/10:<5.1f} {pick['predicted_points']:<5.1f}pts {captain_indicator}")
        
        print("=" * 70)
        print(f"📊 Team Prediction Summary:")
        print(f"   Starting XI: {starters_predicted:.1f} points")
        print(f"   Bench: {bench_predicted:.1f} points")
        print(f"   Total Squad: {total_predicted:.1f} points")
        
        # Add squad analysis for transfer suggestions
        current_team['starting_xi'] = starting_xi
        current_team['bench'] = bench_players
        current_team['total_predicted'] = total_predicted
        current_team['starters_predicted'] = starters_predicted
        current_team['bench_predicted'] = bench_predicted
        
        return current_team
    
    def suggest_transfers(self, team_id=None, gameweeks="5-10"):
        """Suggest transfers (cost money) vs lineup changes (free) based on current team"""
        current_team = self.analyze_current_team(team_id)
        if not current_team:
            return None
        
        print(f"\n🔄 Transfer Analysis (GW{gameweeks}):")
        print("=" * 70)
        
        # Load player data for comparisons
        df = self._load_current_data()
        if df.empty:
            print("❌ No player data available for transfer suggestions")
            return None
        
        # Get current squad player IDs for exclusion
        current_squad_ids = {pick['element_id'] for pick in current_team['picks']}
        current_squad_names = {pick['web_name'] for pick in current_team['picks']}
        
        # Separate available players by position
        available_players = df[~df['id'].isin(current_squad_ids)].copy()
        
        # Calculate prediction score for available players
        available_players['prediction_score'] = (
            available_players['form'].astype(float) * 0.6 + 
            available_players['points_per_game'].astype(float) * 0.4
        )
        
        # Add value score (points per cost)
        available_players['value_score'] = (
            available_players['prediction_score'] / 
            (available_players['now_cost'].astype(float) / 10)
        )
        
        # Get weakest players from current squad for potential transfer out
        current_players = []
        for pick in current_team['picks']:
            current_players.append({
                'name': pick['web_name'],
                'position': pick['position_name'],
                'element_type': pick['element_type'],
                'element_id': pick['element_id'],
                'cost': pick['now_cost'] / 10,
                'predicted_points': pick.get('predicted_points', 0),
                'form': pick.get('form', 0),
                'is_starter': pick['is_starter']
            })
        
        # Sort current players by prediction (worst first for potential transfers out)
        current_players.sort(key=lambda x: x['predicted_points'])
        
        suggestions = []
        bank = current_team['bank']
        
        # Suggest transfers for weakest performers
        positions = [('Goalkeeper', 1), ('Defender', 2), ('Midfielder', 2), ('Forward', 1)]
        for position, element_type in positions:
            # Find current players in this position
            current_pos_players = [p for p in current_players if p['element_type'] == element_type]
            if not current_pos_players:
                continue
            
            # Focus on non-starters or worst performers
            transfer_candidates = [p for p in current_pos_players if not p['is_starter'] or p['predicted_points'] < 3]
            if not transfer_candidates:
                transfer_candidates = current_pos_players[:1]  # Take worst performer
            
            for transfer_out in transfer_candidates[:1]:  # Only suggest one per position
                # Find better replacements in same position
                max_cost = transfer_out['cost'] + bank
                
                replacements = available_players[
                    (available_players['element_type'] == element_type) &
                    (available_players['now_cost'] <= max_cost * 10)
                ].nlargest(3, 'prediction_score')
                
                for _, replacement in replacements.iterrows():
                    cost_diff = (replacement['now_cost'] / 10) - transfer_out['cost']
                    points_improvement = replacement['prediction_score'] - transfer_out['predicted_points']
                    
                    if cost_diff <= bank and points_improvement > 1.0:  # Must be affordable and meaningful improvement
                        suggestion = {
                            'transfer_out': transfer_out['name'],
                            'transfer_in': replacement['web_name'],
                            'position': position,
                            'cost_change': cost_diff,
                            'points_improvement': points_improvement,
                            'transfer_in_predicted': replacement['prediction_score'],
                            'transfer_out_predicted': transfer_out['predicted_points'],
                            'value_score': replacement['value_score']
                        }
                        suggestions.append(suggestion)
                        break  # Only need one good replacement per transfer-out candidate
        
        # Sort suggestions by points improvement
        suggestions.sort(key=lambda x: x['points_improvement'], reverse=True)
        
        # Display transfer suggestions
        if suggestions:
            print("💸 TRANSFERS (Cost Money & Reduce Bank):")
            for i, suggestion in enumerate(suggestions[:3], 1):
                cost_str = f"+£{suggestion['cost_change']:.1f}m" if suggestion['cost_change'] > 0 else f"£{suggestion['cost_change']:.1f}m"
                improvement = suggestion['points_improvement']
                
                print(f"   {i}. {suggestion['transfer_out']} → {suggestion['transfer_in']} "
                      f"({suggestion['position'][:3]}) {cost_str}")
                print(f"      Expected: {suggestion['transfer_out_predicted']:.1f} → "
                      f"{suggestion['transfer_in_predicted']:.1f} pts (+{improvement:.1f})")
        else:
            print("💸 TRANSFERS: ✅ No beneficial transfers found!")
        
        # Show lineup changes (free alternatives)
        print("\n🆓 LINEUP CHANGES (Free - No Transfer Cost):")
        self._suggest_lineup_changes(current_team)
        
        return suggestions
    
    def _suggest_lineup_changes(self, team_data):
        """Suggest lineup changes (free) vs transfers (cost money)"""
        starting_xi = team_data.get('starting_xi', [])
        bench = team_data.get('bench', [])
        
        if not starting_xi or not bench:
            print("   ❌ Unable to analyze lineup - missing squad data")
            return
        
        # Find bench players who might outscore starters
        lineup_swaps = []
        
        for bench_player in bench:
            bench_prediction = bench_player.get('predicted_points', 0)
            bench_position = bench_player['element_type']
            
            # Find starters in same position with lower predictions
            for starter in starting_xi:
                if (starter['element_type'] == bench_position and 
                    starter.get('predicted_points', 0) < bench_prediction):
                    
                    improvement = bench_prediction - starter.get('predicted_points', 0)
                    if improvement > 0.5:  # Only suggest meaningful improvements
                        lineup_swaps.append({
                            'bench_out': starter['web_name'],
                            'bench_in': bench_player['web_name'],
                            'improvement': improvement,
                            'position': bench_player['position_name']
                        })
        
        # Sort by improvement
        lineup_swaps.sort(key=lambda x: x['improvement'], reverse=True)
        
        if lineup_swaps:
            for i, swap in enumerate(lineup_swaps[:2], 1):  # Show top 2
                print(f"   {i}. Start {swap['bench_in']} instead of {swap['bench_out']} "
                      f"({swap['position'][:3]}) +{swap['improvement']:.1f} pts")
        else:
            print("   ✅ Your starting XI looks optimal!")
    
    def get_chip_recommendations(self, gameweeks="5-15"):
        """Get comprehensive chip strategy recommendations"""
        try:
            # Parse gameweek range
            if '-' in gameweeks:
                start_gw, end_gw = map(int, gameweeks.split('-'))
                gameweek_range = range(start_gw, end_gw + 1)
            else:
                start_gw = int(gameweeks)
                gameweek_range = range(start_gw, start_gw + 8)
            
            # Initialize chip analyzer
            chip_analyzer = ChipStrategyAnalyzer(str(self.data_dir))
            
            # Get comprehensive chip advice
            advice = chip_analyzer.get_comprehensive_chip_advice(gameweek_range)
            
            # Display the advice
            chip_analyzer.display_chip_advice(advice)
            
            return advice
            
        except Exception as e:
            print(f"❌ Error analyzing chip strategy: {e}")
            return None
    
    def analyze_chip_timing(self, chip_type="all", gameweeks="5-15"):
        """Analyze optimal timing for specific chip or all chips"""
        try:
            # Parse gameweek range
            if '-' in gameweeks:
                start_gw, end_gw = map(int, gameweeks.split('-'))
                gameweek_range = range(start_gw, end_gw + 1)
            else:
                start_gw = int(gameweeks)
                gameweek_range = range(start_gw, start_gw + 8)
            
            chip_analyzer = ChipStrategyAnalyzer(str(self.data_dir))
            
            if chip_type.lower() == "all":
                return self.get_chip_recommendations(gameweeks)
            
            # Get specific chip analysis
            fixture_analysis = chip_analyzer.analyze_upcoming_fixtures(gameweek_range)
            
            print(f"\n🎴 {chip_type.upper().replace('_', ' ')} Strategy Analysis")
            print("=" * 60)
            
            if chip_type.lower() == "wildcard":
                recommendations = chip_analyzer.recommend_wildcard_timing(fixture_analysis)
            elif chip_type.lower() == "bench_boost":
                recommendations = chip_analyzer.recommend_bench_boost_timing(fixture_analysis)
            elif chip_type.lower() == "triple_captain":
                recommendations = chip_analyzer.recommend_triple_captain_timing(fixture_analysis)
            elif chip_type.lower() == "free_hit":
                recommendations = chip_analyzer.recommend_free_hit_timing(fixture_analysis)
            else:
                print(f"❌ Unknown chip type: {chip_type}")
                print("Available: wildcard, bench_boost, triple_captain, free_hit, all")
                return None
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    confidence_icon = "🎯" if rec['confidence'] == 'High' else "⚡" if rec['confidence'] == 'Medium' else "❓"
                    print(f"{i}. {confidence_icon} Recommended GW: {rec['recommended_gw']}")
                    print(f"   Confidence: {rec['confidence']}")
                    print(f"   Reason: {rec['reason']}")
                    print(f"   Benefit: {rec['benefit']}")
                    if 'suggested_captains' in rec:
                        print(f"   Suggested Captains: {', '.join(rec['suggested_captains'])}")
                    print()
            else:
                print(f"📝 No specific recommendations for {chip_type.replace('_', ' ')} at this time")
                print("Consider waiting for better opportunities (double gameweeks, easy fixtures)")
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error analyzing {chip_type} timing: {e}")
            return None
    
    def display_fixture_difficulty(self, gameweeks="5-10", team_filter=None):
        """Display fixture difficulty ratings for each team by gameweek"""
        # Parse gameweek range
        if '-' in gameweeks:
            start_gw, end_gw = map(int, gameweeks.split('-'))
            gw_range = list(range(start_gw, end_gw + 1))
        else:
            gw_range = [int(gameweeks)]
        
        # Load fixture data
        df = self._load_current_data()
        if df.empty:
            print("❌ No data available. Run 'update-data' first.")
            return
        
        # Load fixtures
        fixtures_file = self.data_dir / "fixtures_latest.csv"
        if not fixtures_file.exists():
            print("❌ No fixture data available. Run 'update-data' first.")
            return
        
        fixtures_df = pd.read_csv(fixtures_file)
        
        # Load teams
        teams_file = self.data_dir / "teams_latest.csv"
        if not teams_file.exists():
            print("❌ No team data available. Run 'update-data' first.")
            return
        
        teams_df = pd.read_csv(teams_file)
        
        # Create team lookup
        team_lookup = {}
        for _, team in teams_df.iterrows():
            team_lookup[team['id']] = team
        
        print(f"\n🏟️  Fixture Difficulty Ratings (GW {'-'.join(map(str, [min(gw_range), max(gw_range)] if len(gw_range) > 1 else [gw_range[0]]))})")
        print("=" * 80)
        print("📊 FDR Scale: 1=Very Easy, 2=Easy, 3=Medium, 4=Hard, 5=Very Hard")
        print("=" * 80)
        
        # Organize fixtures by team and gameweek
        team_fixtures = {}
        for _, fixture in fixtures_df.iterrows():
            gw = fixture['event']
            if gw not in gw_range:
                continue
            
            home_team_id = fixture['team_h']
            away_team_id = fixture['team_a']
            
            if home_team_id not in team_fixtures:
                team_fixtures[home_team_id] = {}
            if away_team_id not in team_fixtures:
                team_fixtures[away_team_id] = {}
            
            # Home team fixture
            if gw not in team_fixtures[home_team_id]:
                team_fixtures[home_team_id][gw] = []
            
            away_team = team_lookup.get(away_team_id, {})
            team_fixtures[home_team_id][gw].append({
                'opponent': away_team.get('short_name', f'Team{away_team_id}'),
                'venue': 'H',
                'difficulty': fixture.get('team_h_difficulty', 3)
            })
            
            # Away team fixture
            if gw not in team_fixtures[away_team_id]:
                team_fixtures[away_team_id][gw] = []
            
            home_team = team_lookup.get(home_team_id, {})
            team_fixtures[away_team_id][gw].append({
                'opponent': home_team.get('short_name', f'Team{home_team_id}'),
                'venue': 'A',
                'difficulty': fixture.get('team_a_difficulty', 3)
            })
        
        # Display fixtures for each gameweek
        for gw in gw_range:
            print(f"\n🗓️  Gameweek {gw}")
            print("-" * 60)
            
            # Sort teams alphabetically
            sorted_teams = sorted(team_fixtures.items(), key=lambda x: team_lookup.get(x[0], {}).get('short_name', ''))
            
            for team_id, team_gw_fixtures in sorted_teams:
                team_info = team_lookup.get(team_id, {})
                team_name = team_info.get('short_name', f'Team{team_id}')
                
                # Filter by team if specified
                if team_filter and team_name.upper() != team_filter.upper():
                    continue
                
                if gw in team_gw_fixtures:
                    fixtures_for_gw = team_gw_fixtures[gw]
                    for fixture in fixtures_for_gw:
                        difficulty = fixture['difficulty']
                        venue_symbol = "🏠" if fixture['venue'] == 'H' else "✈️"
                        
                        # Color coding for difficulty
                        if difficulty <= 2:
                            difficulty_color = "🟢"  # Green for easy
                        elif difficulty == 3:
                            difficulty_color = "🟡"  # Yellow for medium
                        else:
                            difficulty_color = "🔴"  # Red for hard
                        
                        print(f"{team_name:<4} vs {fixture['opponent']:<4} {venue_symbol} "
                              f"FDR: {difficulty_color} {difficulty}")
                else:
                    # No fixture this gameweek (blank/double gameweek)
                    if not team_filter or team_name.upper() == team_filter.upper():
                        print(f"{team_name:<4} --- BLANK ---     FDR: ⚪ -")
        
        # Summary statistics
        if not team_filter:
            print(f"\n📈 Difficulty Summary (GW {'-'.join(map(str, [min(gw_range), max(gw_range)] if len(gw_range) > 1 else [gw_range[0]]))})")
            print("-" * 50)
            
            easy_teams = []
            hard_teams = []
            
            for team_id, team_gw_fixtures in team_fixtures.items():
                team_name = team_lookup.get(team_id, {}).get('short_name', f'Team{team_id}')
                total_difficulty = 0
                fixture_count = 0
                
                for gw in gw_range:
                    if gw in team_gw_fixtures:
                        for fixture in team_gw_fixtures[gw]:
                            total_difficulty += fixture['difficulty']
                            fixture_count += 1
                
                if fixture_count > 0:
                    avg_difficulty = total_difficulty / fixture_count
                    if avg_difficulty <= 2.5:
                        easy_teams.append((team_name, avg_difficulty, fixture_count))
                    elif avg_difficulty >= 3.5:
                        hard_teams.append((team_name, avg_difficulty, fixture_count))
            
            # Sort by difficulty
            easy_teams.sort(key=lambda x: x[1])
            hard_teams.sort(key=lambda x: x[1], reverse=True)
            
            if easy_teams:
                print("🟢 Easiest Fixtures:")
                for team, avg_diff, count in easy_teams[:5]:
                    print(f"   {team}: {avg_diff:.1f} avg ({count} fixtures)")
            
            if hard_teams:
                print("🔴 Hardest Fixtures:")
                for team, avg_diff, count in hard_teams[:5]:
                    print(f"   {team}: {avg_diff:.1f} avg ({count} fixtures)")

    def display_matches(self, gameweek, output_format='simple', save_for_models=False):
        """Display all matches for a specific gameweek in various formats"""
        # Load fixture data
        fixtures_file = self.data_dir / "fixtures_latest.csv"
        if not fixtures_file.exists():
            print("❌ No fixture data available. Run 'update-data' first.")
            return
        
        fixtures_df = pd.read_csv(fixtures_file)
        
        # Load teams
        teams_file = self.data_dir / "teams_latest.csv"
        if not teams_file.exists():
            print("❌ No team data available. Run 'update-data' first.")
            return
        
        teams_df = pd.read_csv(teams_file)
        
        # Create team lookup
        team_lookup = {}
        for _, team in teams_df.iterrows():
            team_lookup[team['id']] = team
        
        # Filter fixtures for the specific gameweek
        gw_fixtures = fixtures_df[fixtures_df['event'] == gameweek].copy()
        
        if gw_fixtures.empty:
            print(f"❌ No fixtures found for Gameweek {gameweek}")
            return
        
        # Sort by kickoff time if available
        if 'kickoff_time' in gw_fixtures.columns:
            gw_fixtures = gw_fixtures.sort_values('kickoff_time')
        
        # Prepare enhanced data for AI analysis
        matches_data = []
        for _, fixture in gw_fixtures.iterrows():
            home_team = team_lookup.get(fixture['team_h'], {})
            away_team = team_lookup.get(fixture['team_a'], {})
            
            match_data = {
                'gameweek': int(gameweek),
                'fixture_id': int(fixture.get('id', 0)),
                'home_team': {
                    'id': int(fixture['team_h']),
                    'name': home_team.get('name', 'Unknown'),
                    'short_name': home_team.get('short_name', 'UNK'),
                    'difficulty': int(fixture.get('team_h_difficulty', 3)),
                    'strength_overall_home': int(home_team.get('strength_overall_home', 1000)),
                    'strength_attack_home': int(home_team.get('strength_attack_home', 1000)),
                    'strength_defence_home': int(home_team.get('strength_defence_home', 1000))
                },
                'away_team': {
                    'id': int(fixture['team_a']),
                    'name': away_team.get('name', 'Unknown'),
                    'short_name': away_team.get('short_name', 'UNK'),
                    'difficulty': int(fixture.get('team_a_difficulty', 3)),
                    'strength_overall_away': int(away_team.get('strength_overall_away', 1000)),
                    'strength_attack_away': int(away_team.get('strength_attack_away', 1000)),
                    'strength_defence_away': int(away_team.get('strength_defence_away', 1000))
                },
                'kickoff_time': fixture.get('kickoff_time', ''),
                'finished': bool(fixture.get('finished', False)),
                'provisional_start_time': bool(fixture.get('provisional_start_time', False)),
                'pulse_id': int(fixture.get('pulse_id', 0))
            }
            
            # Add scores if finished
            if fixture.get('finished', False):
                match_data['home_score'] = int(fixture.get('team_h_score', 0))
                match_data['away_score'] = int(fixture.get('team_a_score', 0))
            
            matches_data.append(match_data)
        
        # Save for AI models if requested
        if save_for_models or output_format == 'ai-analysis':
            model_data_file = self.data_dir / f"gw{gameweek}_matches_for_models.json"
            with open(model_data_file, 'w') as f:
                import json
                json.dump({
                    'metadata': {
                        'gameweek': gameweek,
                        'total_matches': len(matches_data),
                        'extraction_timestamp': pd.Timestamp.now().isoformat(),
                        'data_source': 'official_fpl_api'
                    },
                    'matches': matches_data
                }, f, indent=2)
            print(f"💾 Match data saved for AI models: {model_data_file}")
        
        if output_format == 'ai-analysis':
            # AI Analysis format - comprehensive data with analysis prompts
            print(f"\n🤖 AI Analysis: Gameweek {gameweek} Fixtures")
            print("=" * 80)
            print("📊 Data Summary for AI Models:")
            print(f"   • Total Matches: {len(matches_data)}")
            print(f"   • Difficulty Distribution: {self._analyze_difficulty_distribution(matches_data)}")
            print(f"   • Key Matchups: {self._identify_key_matchups(matches_data)}")
            
            print(f"\n🎯 AI Model Analysis Prompts:")
            print("1. Predict match outcomes based on team strengths and difficulty ratings")
            print("2. Identify best fantasy players to target based on fixture difficulty")
            print("3. Analyze potential upsets (low-strength teams vs high-difficulty opponents)")
            print("4. Calculate expected goals based on attack/defense strength ratios")
            
            print(f"\n📋 Structured Data for Models:")
            import json
            print(json.dumps({
                'summary': {
                    'gameweek': gameweek,
                    'total_matches': len(matches_data),
                    'average_home_difficulty': sum(m['home_team']['difficulty'] for m in matches_data) / len(matches_data),
                    'average_away_difficulty': sum(m['away_team']['difficulty'] for m in matches_data) / len(matches_data)
                },
                'key_matches': [m for m in matches_data if m['home_team']['difficulty'] >= 4 or m['away_team']['difficulty'] >= 4][:3],
                'easy_fixtures': [m for m in matches_data if m['home_team']['difficulty'] <= 2 or m['away_team']['difficulty'] <= 2][:3]
            }, indent=2))
            
            return matches_data
        
        elif output_format == 'json':
            # JSON format for model consumption
            import json
            print(json.dumps(matches_data, indent=2))
            return matches_data
        
        elif output_format == 'detailed':
            # Detailed format with stats
            print(f"\n⚽ Gameweek {gameweek} Fixtures - Detailed View")
            print("=" * 80)
            
            for match_data in matches_data:
                print(f"\n🏟️  {match_data['home_team']['name']} vs {match_data['away_team']['name']}")
                print(f"    Teams: {match_data['home_team']['short_name']} (H) vs {match_data['away_team']['short_name']} (A)")
                print(f"    Difficulty: Home FDR {match_data['home_team']['difficulty']} | Away FDR {match_data['away_team']['difficulty']}")
                
                if match_data['kickoff_time']:
                    print(f"    Kickoff: {match_data['kickoff_time']}")
                
                if match_data['finished']:
                    print(f"    Final Score: {match_data['home_score']} - {match_data['away_score']}")
                else:
                    print(f"    Status: {'Scheduled' if not match_data['provisional_start_time'] else 'Provisional'}")
                
                print(f"    Fixture ID: {match_data['fixture_id']}")
        
        else:
            # Simple format - clean match display
            print(f"\n⚽ Gameweek {gameweek} Fixtures")
            print("=" * 50)
            
            match_count = 0
            for match_data in matches_data:
                match_count += 1
                
                home_name = match_data['home_team']['short_name']
                away_name = match_data['away_team']['short_name']
                
                if match_data['finished']:
                    # Show final score for finished matches
                    print(f"{match_count:2d}. {home_name} {match_data['home_score']}-{match_data['away_score']} {away_name}")
                else:
                    # Show upcoming match
                    print(f"{match_count:2d}. {home_name} vs {away_name}")
            
            print(f"\nTotal matches: {match_count}")
        
        return matches_data
    
    def _analyze_difficulty_distribution(self, matches_data):
        """Analyze the distribution of fixture difficulties"""
        difficulties = []
        for match in matches_data:
            difficulties.extend([match['home_team']['difficulty'], match['away_team']['difficulty']])
        
        easy = sum(1 for d in difficulties if d <= 2)
        medium = sum(1 for d in difficulties if d == 3)
        hard = sum(1 for d in difficulties if d >= 4)
        
        return f"Easy: {easy}, Medium: {medium}, Hard: {hard}"
    
    def _identify_key_matchups(self, matches_data):
        """Identify the most interesting matchups for analysis"""
        key_matches = []
        for match in matches_data:
            # High difficulty for both teams = big match
            if match['home_team']['difficulty'] >= 4 and match['away_team']['difficulty'] >= 4:
                key_matches.append(f"{match['home_team']['short_name']} vs {match['away_team']['short_name']}")
        
        return ", ".join(key_matches[:3]) if key_matches else "No major clashes identified"

    def update_data(self):
        """Comprehensive FPL data update from official API"""
        print("🔄 Updating FPL data from official API...")
        print("=" * 60)
        
        try:
            import requests
            from datetime import datetime
            
            # Create data directory if it doesn't exist
            self.data_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. BOOTSTRAP DATA (Teams, Players, Events, Elements)
            print("📊 Fetching bootstrap data (teams, players, gameweeks)...")
            bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
            response = requests.get(bootstrap_url)
            bootstrap_data = response.json()
            
            # Save raw bootstrap data
            with open(self.data_dir / f'bootstrap_data_{timestamp}.json', 'w') as f:
                import json
                json.dump(bootstrap_data, f, indent=2)
            
            print(f"✅ Teams: {len(bootstrap_data['teams'])} clubs")
            print(f"✅ Players: {len(bootstrap_data['elements'])} players")
            print(f"✅ Gameweeks: {len(bootstrap_data['events'])} events")
            
            # 2. PROCESS TEAMS DATA
            teams_df = pd.DataFrame(bootstrap_data['teams'])
            teams_df.to_csv(self.data_dir / f'teams_{timestamp}.csv', index=False)
            teams_df.to_csv(self.data_dir / 'teams_latest.csv', index=False)
            print(f"� Saved teams data: {len(teams_df)} teams")
            
            # 3. PROCESS PLAYERS DATA (MAIN DATASET)
            players_df = pd.DataFrame(bootstrap_data['elements'])
            
            # Add team names
            team_mapping = {team['id']: team['short_name'] for team in bootstrap_data['teams']}
            players_df['team_name'] = players_df['team'].map(team_mapping)
            
            # Add position names
            position_mapping = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            players_df['position_name'] = players_df['element_type'].map(position_mapping)
            
            # Convert price to proper format (API gives in 0.1m units)
            players_df['price'] = players_df['now_cost'] / 10
            
            # Save players data
            players_df.to_csv(self.data_dir / f'players_{timestamp}.csv', index=False)
            players_df.to_csv(self.data_dir / 'players_latest.csv', index=False)
            print(f"💾 Saved players data: {len(players_df)} players")
            print(f"   🏆 Top scorer: {players_df.loc[players_df['total_points'].idxmax(), 'web_name']} ({players_df['total_points'].max()} pts)")
            print(f"   💰 Price range: £{players_df['price'].min():.1f}m - £{players_df['price'].max():.1f}m")
            
            # 4. PROCESS GAMEWEEKS DATA
            gameweeks_df = pd.DataFrame(bootstrap_data['events'])
            gameweeks_df.to_csv(self.data_dir / f'gameweeks_{timestamp}.csv', index=False)
            gameweeks_df.to_csv(self.data_dir / 'gameweeks_latest.csv', index=False)
            
            # Find current gameweek
            current_gw = None
            for gw in bootstrap_data['events']:
                if gw['is_current']:
                    current_gw = gw['id']
                    break
            
            print(f"💾 Saved gameweeks data: {len(gameweeks_df)} gameweeks")
            print(f"   � Current gameweek: {current_gw}")
            
            # 5. FIXTURES DATA
            print("\n🗓️ Fetching fixtures data...")
            fixtures_url = "https://fantasy.premierleague.com/api/fixtures/"
            response = requests.get(fixtures_url)
            fixtures_data = response.json()
            
            fixtures_df = pd.DataFrame(fixtures_data)
            
            # Add team names to fixtures
            fixtures_df['team_h_name'] = fixtures_df['team_h'].map(team_mapping)
            fixtures_df['team_a_name'] = fixtures_df['team_a'].map(team_mapping)
            
            # Save fixtures data
            fixtures_df.to_csv(self.data_dir / f'fixtures_{timestamp}.csv', index=False)
            fixtures_df.to_csv(self.data_dir / 'fixtures_latest.csv', index=False)
            print(f"💾 Saved fixtures data: {len(fixtures_df)} fixtures")
            
            # 6. PLAYER DETAILED STATS (Sample of top players)
            print("\n📈 Fetching detailed player stats (top 50 players)...")
            top_players = players_df.nlargest(50, 'total_points')
            
            detailed_stats = []
            for _, player in top_players.iterrows():
                try:
                    player_url = f"https://fantasy.premierleague.com/api/element-summary/{player['id']}/"
                    response = requests.get(player_url)
                    player_data = response.json()
                    
                    # Get gameweek history
                    for gw_data in player_data['history']:
                        gw_record = {
                            'player_id': player['id'],
                            'player_name': player['web_name'],
                            'team': player['team_name'],
                            'position': player['position_name'],
                            'gameweek': gw_data['round'],
                            'points': gw_data['total_points'],
                            'minutes': gw_data['minutes'],
                            'goals_scored': gw_data['goals_scored'],
                            'assists': gw_data['assists'],
                            'clean_sheets': gw_data['clean_sheets'],
                            'goals_conceded': gw_data['goals_conceded'],
                            'own_goals': gw_data['own_goals'],
                            'penalties_saved': gw_data['penalties_saved'],
                            'penalties_missed': gw_data['penalties_missed'],
                            'yellow_cards': gw_data['yellow_cards'],
                            'red_cards': gw_data['red_cards'],
                            'saves': gw_data['saves'],
                            'bonus': gw_data['bonus'],
                            'bps': gw_data['bps'],
                            'influence': gw_data['influence'],
                            'creativity': gw_data['creativity'],
                            'threat': gw_data['threat'],
                            'ict_index': gw_data['ict_index'],
                            'value': gw_data['value'],
                            'selected': gw_data['selected']
                        }
                        detailed_stats.append(gw_record)
                    
                    print(f"   ✅ {player['web_name']} ({len(player_data['history'])} gameweeks)")
                    
                except Exception as e:
                    print(f"   ❌ {player['web_name']}: {e}")
                    continue
            
            # Save detailed stats
            if detailed_stats:
                detailed_df = pd.DataFrame(detailed_stats)
                detailed_df.to_csv(self.data_dir / f'player_stats_detailed_{timestamp}.csv', index=False)
                detailed_df.to_csv(self.data_dir / 'player_stats_detailed_latest.csv', index=False)
                print(f"💾 Saved detailed stats: {len(detailed_df)} player-gameweek records")
            
            # 7. CREATE ENHANCED DATASET FOR ML
            print("\n🤖 Creating enhanced dataset for predictions...")
            enhanced_df = players_df.copy()
            
            # Add advanced metrics
            enhanced_df['form_numeric'] = pd.to_numeric(enhanced_df['form'], errors='coerce')
            enhanced_df['points_per_game_numeric'] = pd.to_numeric(enhanced_df['points_per_game'], errors='coerce')
            enhanced_df['selected_by_percent_numeric'] = pd.to_numeric(enhanced_df['selected_by_percent'], errors='coerce')
            enhanced_df['influence_numeric'] = pd.to_numeric(enhanced_df['influence'], errors='coerce') 
            enhanced_df['creativity_numeric'] = pd.to_numeric(enhanced_df['creativity'], errors='coerce')
            enhanced_df['threat_numeric'] = pd.to_numeric(enhanced_df['threat'], errors='coerce')
            enhanced_df['ict_index_numeric'] = pd.to_numeric(enhanced_df['ict_index'], errors='coerce')
            
            # Calculate value metrics
            enhanced_df['value_form'] = enhanced_df['form_numeric'] / enhanced_df['price']
            enhanced_df['value_total'] = enhanced_df['total_points'] / enhanced_df['price']
            enhanced_df['value_ppg'] = enhanced_df['points_per_game_numeric'] / enhanced_df['price']
            
            # Add current gameweek
            enhanced_df['current_gameweek'] = current_gw
            enhanced_df['update_timestamp'] = datetime.now().isoformat()
            
            # Save enhanced dataset
            enhanced_df.to_csv(self.data_dir / f'enhanced_fpl_features_{timestamp}.csv', index=False)
            enhanced_df.to_csv(self.data_dir / 'enhanced_fpl_features.csv', index=False)
            print(f"💾 Saved enhanced dataset: {len(enhanced_df)} players with ML features")
            
            # 8. SUMMARY REPORT
            print(f"\n📋 DATA UPDATE SUMMARY")
            print("=" * 40)
            print(f"📅 Timestamp: {timestamp}")
            print(f"🏆 Current Gameweek: {current_gw}")
            print(f"⚽ Teams: {len(teams_df)}")
            print(f"👤 Players: {len(players_df)}")
            print(f"🗓️ Fixtures: {len(fixtures_df)}")
            print(f"📈 Detailed Records: {len(detailed_stats) if detailed_stats else 0}")
            print(f"🤖 Enhanced Features: {len(enhanced_df)}")
            print(f"💾 Files saved in: {self.data_dir}")
            
            # Export comprehensive data for forms/models
            print("\n📤 Exporting data for forms and models...")
            self._export_comprehensive_data_pipeline(timestamp)
            
            # Auto-collect performance history after data update
            print("\n🔄 Auto-collecting performance history...")
            try:
                performance_data = self.collect_player_performance_history(gameweeks_back=4)
                if performance_data:
                    self.save_performance_history(performance_data)
                    print("✅ Performance history updated automatically")
                else:
                    print("⚠️ No performance history collected")
            except Exception as e:
                print(f"⚠️ Performance history collection failed: {e}")
            
            # Check if model retraining is needed
            print("\n🔄 Checking if model retraining is needed...")
            if self._should_retrain_models():
                print("🎯 New gameweek data detected - triggering model retraining...")
                self._retrain_models_with_new_data()
            else:
                print("✅ Models are up-to-date with current data")
            
            print("\n✅ FPL data update completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error updating FPL data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def collect_player_performance_history(self, gameweeks_back=4, incremental=True):
        """Collect historical performance data for all players with smart caching"""
        print(f"🔄 Collecting player performance history (last {gameweeks_back} gameweeks)...")
        
        # Load current players
        players_df = pd.read_csv(self.data_dir / 'players_latest.csv')
        
        # Check for existing performance data for incremental updates
        performance_data = {}
        existing_data = {}
        history_file = self.data_dir / "performance_history" / "all_players_performance.json"
        
        if incremental and history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    existing_data = json.load(f)
                print(f"📂 Found existing performance data for {len(existing_data)} players")
            except Exception as e:
                print(f"⚠️ Could not load existing data: {e}")
                existing_data = {}
        
        # Metrics mapping
        metrics_mapping = {
            'PTS': 'total_points',
            'ST': 'starts', 
            'MP': 'minutes',
            'GS': 'goals_scored',
            'A': 'assists',
            'xG': 'expected_goals',
            'xA': 'expected_assists', 
            'xGI': 'expected_goal_involvements',
            'CS': 'clean_sheets',
            'GC': 'goals_conceded',
            'xGC': 'expected_goals_conceded',
            'T': 'tackles',
            'CBI': 'clearances_blocks_interceptions',
            'R': 'recoveries', 
            'DC': 'defensive_contribution',
            'OG': 'own_goals',
            'PS': 'penalties_saved',
            'PM': 'penalties_missed',
            'YC': 'yellow_cards',
            'RC': 'red_cards',
            'S': 'saves',
            'BP': 'bonus',
            'BPS': 'bps',
            'I': 'influence',
            'C': 'creativity', 
            'T_threat': 'threat',
            'II': 'ict_index',
            'NT': 'transfers_balance',
            'TSB': 'selected'
        }
        
        total_players = len(players_df)
        collected = 0
        updated = 0
        skipped = 0
        errors = 0
        
        # Get current gameweek for smart updating
        current_gw = None
        try:
            fixtures_df = pd.read_csv(self.data_dir / 'fixtures_latest.csv')
            current_gw = fixtures_df['event'].max()
        except:
            pass
        
        for _, player in players_df.iterrows():
            player_id = str(player['id'])  # Ensure string for consistency
            
            # Check if we need to update this player
            should_update = True
            if incremental and player_id in existing_data:
                existing_player = existing_data[player_id]
                existing_gameweeks = [gw['gameweek'] for gw in existing_player.get('gameweeks', [])]
                
                # If we have recent data and current gameweek, check if we need update
                if existing_gameweeks and current_gw:
                    latest_gw = max(existing_gameweeks)
                    if latest_gw >= current_gw - 1:  # Allow 1 gameweek tolerance
                        should_update = False
                        performance_data[player_id] = existing_player
                        skipped += 1
            
            if should_update:
                try:
                    # Get player's detailed history
                    url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
                    response = requests.get(url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        history = data.get('history', [])
                        
                        if history:
                            # Get last N gameweeks
                            recent_history = history[-gameweeks_back:] if len(history) >= gameweeks_back else history
                            
                            player_performance = {
                                'player_id': int(player_id),
                                'name': player['web_name'],
                                'team': player['team'],
                                'position': player['element_type'],
                                'last_updated': datetime.now().isoformat(),
                                'gameweeks': []
                            }
                            
                            for gw_data in recent_history:
                                gw_performance = {'gameweek': gw_data.get('round', 0)}
                                
                                # Extract all required metrics
                                for metric_code, api_field in metrics_mapping.items():
                                    value = gw_data.get(api_field, 0)
                                    # Handle float values
                                    if isinstance(value, float):
                                        value = round(value, 2)
                                    gw_performance[metric_code] = value
                                
                                # Add fixture context
                                gw_performance['opponent_team'] = gw_data.get('opponent_team', 0)
                                gw_performance['was_home'] = gw_data.get('was_home', False)
                                gw_performance['kickoff_time'] = gw_data.get('kickoff_time', '')
                                
                                player_performance['gameweeks'].append(gw_performance)
                            
                            performance_data[player_id] = player_performance
                            
                            if player_id in existing_data:
                                updated += 1
                            else:
                                collected += 1
                        
                        # Rate limiting - small delay every 50 requests
                        if (collected + updated) % 50 == 0:
                            time.sleep(0.1)
                            print(f"   📊 Processed {collected + updated + skipped}/{total_players} players (collected: {collected}, updated: {updated}, skipped: {skipped})...")
                            
                    else:
                        errors += 1
                        # Keep existing data if available
                        if player_id in existing_data:
                            performance_data[player_id] = existing_data[player_id]
                        
                except Exception as e:
                    errors += 1
                    # Keep existing data if available
                    if player_id in existing_data:
                        performance_data[player_id] = existing_data[player_id]
                    continue
        
        print(f"✅ Performance history collection complete:")
        print(f"   📊 {collected} new players collected")
        print(f"   🔄 {updated} players updated")
        print(f"   ⏭️  {skipped} players skipped (up-to-date)")
        print(f"   ❌ {errors} players with errors")
        print(f"   📈 Total dataset: {len(performance_data)} players")
        
        return performance_data
    
    def save_performance_history(self, performance_data):
        """Save performance data to structured files"""
        history_dir = self.data_dir / "performance_history"
        history_dir.mkdir(exist_ok=True)
        
        # Save complete dataset
        with open(history_dir / 'all_players_performance.json', 'w') as f:
            json.dump(performance_data, f, indent=2)
        
        # Save by position for easier model access
        position_data = {1: [], 2: [], 3: [], 4: []}
        
        for player_data in performance_data.values():
            position = player_data['position']
            if position in position_data:
                position_data[position].append(player_data)
        
        position_names = {1: 'goalkeepers', 2: 'defenders', 3: 'midfielders', 4: 'forwards'}
        for pos_id, pos_name in position_names.items():
            with open(history_dir / f'{pos_name}_performance.json', 'w') as f:
                json.dump(position_data[pos_id], f, indent=2)
        
        # Create model-ready feature matrix
        self._create_model_features(performance_data, history_dir)
        
        print(f"✅ Performance history saved to {history_dir}/")
        return history_dir
    
    def _create_model_features(self, performance_data, output_dir):
        """Create model-ready feature matrices from performance data"""
        
        # Flatten data for model consumption
        model_features = []
        
        for player_data in performance_data.values():
            player_id = player_data['player_id']
            name = player_data['name']
            team = player_data['team']
            position = player_data['position']
            
            gameweeks = player_data['gameweeks']
            
            if len(gameweeks) >= 2:  # Need at least 2 gameweeks for trends
                # Calculate rolling averages and trends
                recent_gws = gameweeks[-4:] if len(gameweeks) >= 4 else gameweeks
                
                # Average performance metrics
                avg_metrics = {}
                trend_metrics = {}
                
                metrics = ['PTS', 'MP', 'GS', 'A', 'xG', 'xA', 'CS', 'T', 'CBI', 'R', 'BP', 'BPS', 'I', 'C', 'T_threat', 'II']
                
                for metric in metrics:
                    values = [gw.get(metric, 0) for gw in recent_gws]
                    # Convert to numeric values, handling any string/null values
                    numeric_values = []
                    for v in values:
                        try:
                            if v is None or v == '':
                                numeric_values.append(0)
                            else:
                                numeric_values.append(float(v))
                        except (ValueError, TypeError):
                            numeric_values.append(0)
                    
                    avg_metrics[f'avg_{metric}'] = round(sum(numeric_values) / len(numeric_values), 2)
                    
                    # Simple trend: last 2 vs first 2 gameweeks
                    if len(numeric_values) >= 3:
                        first_half = numeric_values[:len(numeric_values)//2]
                        second_half = numeric_values[len(numeric_values)//2:]
                        first_avg = sum(first_half) / len(first_half)
                        second_avg = sum(second_half) / len(second_half)
                        trend_metrics[f'trend_{metric}'] = round(second_avg - first_avg, 2)
                
                # Create feature row
                feature_row = {
                    'player_id': player_id,
                    'name': name,
                    'team': team,
                    'position': position,
                    'gameweeks_played': len(gameweeks),
                    **avg_metrics,
                    **trend_metrics
                }
                
                model_features.append(feature_row)
        
        # Save as CSV for easy model loading
        features_df = pd.DataFrame(model_features)
        features_df.to_csv(output_dir / 'model_features.csv', index=False)
        
        # Save feature metadata
        feature_metadata = {
            'total_players': len(model_features),
            'feature_count': len(features_df.columns),
            'metrics_included': list(metrics),
            'created_at': datetime.now().isoformat(),
            'description': 'Model-ready features with 4-gameweek rolling averages and trends'
        }
        
        with open(output_dir / 'feature_metadata.json', 'w') as f:
            json.dump(feature_metadata, f, indent=2)
        
        print(f"   📊 Model features: {len(model_features)} players × {len(features_df.columns)} features")
    
    def _should_retrain_models(self):
        """Check if models need retraining based on new data availability"""
        try:
            # Check model metadata to see when it was last trained
            metadata_files = list(self.models_dir.glob('model_metadata_*.json'))
            if not metadata_files:
                print("📊 No model metadata found - retraining recommended")
                return True
            
            # Load latest model metadata
            latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
            with open(latest_metadata, 'r') as f:
                metadata = json.load(f)
            
            # Get training info
            dataset_info = metadata.get('dataset_info', {})
            trained_gameweeks = dataset_info.get('gameweeks_trained', '')
            model_timestamp = metadata.get('timestamp', '')
            
            print(f"   📅 Current models trained on: {trained_gameweeks}")
            print(f"   ⏰ Model timestamp: {model_timestamp}")
            
            # Check available data gameweeks
            try:
                fixtures_df = pd.read_csv(self.data_dir / 'fixtures_latest.csv')
                completed_gameweeks = fixtures_df[fixtures_df['finished'] == True]['event'].unique()
                latest_completed_gw = max(completed_gameweeks) if len(completed_gameweeks) > 0 else 0
                
                print(f"   🏁 Latest completed gameweek: {latest_completed_gw}")
                
                # Extract last trained gameweek
                if '-' in trained_gameweeks:
                    last_trained_gw = int(trained_gameweeks.split('-')[1])
                else:
                    last_trained_gw = int(trained_gameweeks) if trained_gameweeks.isdigit() else 0
                
                # Retrain if we have new completed gameweeks
                if latest_completed_gw > last_trained_gw:
                    print(f"   🎯 New data available: GW {last_trained_gw + 1} to GW {latest_completed_gw}")
                    return True
                
                # Also retrain if models are older than 7 days
                try:
                    model_date = datetime.strptime(model_timestamp, '%Y%m%d_%H%M%S')
                    days_old = (datetime.now() - model_date).days
                    
                    if days_old > 7:
                        print(f"   ⏰ Models are {days_old} days old - retraining recommended")
                        return True
                        
                except ValueError:
                    print("   ⚠️ Could not parse model timestamp - retraining recommended")
                    return True
                
                print(f"   ✅ Models are current (trained on GW {last_trained_gw}, latest completed GW {latest_completed_gw})")
                return False
                
            except Exception as e:
                print(f"   ⚠️ Could not check gameweek status: {e}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error checking retraining needs: {e}")
            return False
    
    def _retrain_models_with_new_data(self):
        """Retrain models with updated data using automated pipeline"""
        try:
            print("🚀 Starting intelligent model retraining...")
            
            # Check if we have the required data
            required_files = [
                self.data_dir / 'players_latest.csv',
                self.data_dir / 'fixtures_latest.csv',
                self.data_dir / 'performance_history' / 'model_features.csv'
            ]
            
            missing_files = [f for f in required_files if not f.exists()]
            if missing_files:
                print(f"❌ Missing required files for retraining: {missing_files}")
                return False
            
            # Import required libraries for training
            try:
                import xgboost as xgb
                import lightgbm as lgb
                from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics import mean_squared_error, mean_absolute_error
                import joblib
            except ImportError as e:
                print(f"❌ Missing ML libraries for retraining: {e}")
                print("💡 Install with: pip install xgboost lightgbm scikit-learn")
                return False
            
            print("📊 Preparing training data...")
            
            # Load and prepare data
            players_df = pd.read_csv(self.data_dir / 'players_latest.csv')
            fixtures_df = pd.read_csv(self.data_dir / 'fixtures_latest.csv')
            performance_df = pd.read_csv(self.data_dir / 'performance_history' / 'model_features.csv')
            
            # Create comprehensive training dataset
            training_data = self._prepare_training_data(players_df, fixtures_df, performance_df)
            
            if training_data is None or len(training_data) < 100:
                print(f"❌ Insufficient training data: {len(training_data) if training_data is not None else 0} samples")
                return False
            
            print(f"✅ Training dataset prepared: {len(training_data)} samples")
            
            # Split features and target
            # Exclude non-numeric and identifier columns
            exclude_cols = [
                'total_points', 'player_id', 'name', 'team_name', 'position_name', 
                'web_name', 'first_name', 'second_name', 'photo', 'news', 
                'news_added', 'status', 'team_join_date', 'birth_date', 'opta_code',
                'id', 'code', 'removed', 'special', 'region', 'has_temporary_code'
            ]
            
            feature_cols = [col for col in training_data.columns if col not in exclude_cols]
            
            # Ensure only numeric features
            numeric_features = []
            for col in feature_cols:
                if training_data[col].dtype in ['int64', 'float64', 'int32', 'float32', 'bool']:
                    numeric_features.append(col)
                elif training_data[col].dtype == 'object':
                    # Try to convert to numeric
                    try:
                        training_data[col] = pd.to_numeric(training_data[col], errors='coerce')
                        numeric_features.append(col)
                    except:
                        pass
            
            X = training_data[numeric_features].fillna(0)
            y = training_data['total_points']
            
            print(f"🔢 Using {len(numeric_features)} numeric features for training")
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            print(f"📈 Training models on {len(X_train)} samples, testing on {len(X_test)} samples...")
            
            # Train ensemble models
            models = self._train_ensemble_models(X_train, y_train, X_test, y_test)
            
            if not models:
                print("❌ Model training failed")
                return False
            
            # Validate new models against current ones
            if self._validate_new_models(models, X_test, y_test):
                # Save new models
                self._save_retrained_models(models, X_train, y_train, numeric_features)
                print("✅ Model retraining completed successfully!")
                return True
            else:
                print("⚠️ New models did not improve performance - keeping current models")
                return False
                
        except Exception as e:
            print(f"❌ Model retraining failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _prepare_training_data(self, players_df, fixtures_df, performance_df):
        """Prepare comprehensive training dataset"""
        try:
            # Merge performance features with player data
            training_data = players_df.merge(
                performance_df,
                left_on='id',
                right_on='player_id',
                how='inner'
            )
            
            # Add fixture difficulty features
            current_gw = fixtures_df['event'].max()
            upcoming_fixtures = fixtures_df[
                (fixtures_df['event'] >= current_gw) & 
                (fixtures_df['event'] <= current_gw + 3)
            ]
            
            # Calculate upcoming fixture difficulty
            fixture_difficulty = {}
            for _, fixture in upcoming_fixtures.iterrows():
                home_team = fixture['team_h']
                away_team = fixture['team_a']
                
                if home_team not in fixture_difficulty:
                    fixture_difficulty[home_team] = []
                if away_team not in fixture_difficulty:
                    fixture_difficulty[away_team] = []
                
                fixture_difficulty[home_team].append(fixture['team_h_difficulty'])
                fixture_difficulty[away_team].append(fixture['team_a_difficulty'])
            
            # Add average fixture difficulty using team_code mapping
            # Map team codes to team IDs if needed
            if 'team_code' in training_data.columns:
                # Use team_code directly
                training_data['avg_fixture_difficulty'] = training_data['team_code'].map(
                    lambda x: np.mean(fixture_difficulty.get(x, [3])) if x in fixture_difficulty else 3
                )
            elif 'team' in training_data.columns:
                # Use team ID
                training_data['avg_fixture_difficulty'] = training_data['team'].map(
                    lambda x: np.mean(fixture_difficulty.get(x, [3])) if x in fixture_difficulty else 3
                )
            else:
                # Default fixture difficulty
                training_data['avg_fixture_difficulty'] = 3
            
            # Engineering additional features
            training_data['price_performance_ratio'] = training_data['total_points'] / (training_data['now_cost'] / 10)
            training_data['minutes_per_point'] = training_data['minutes'] / (training_data['total_points'] + 1)
            training_data['form_momentum'] = training_data.get('avg_PTS', 0) - training_data['points_per_game']
            
            # Clean data
            training_data = training_data.dropna(subset=['total_points'])
            training_data = training_data[training_data['total_points'] >= 0]
            
            # Ensure we have minimum required columns
            required_cols = ['total_points', 'now_cost', 'minutes', 'points_per_game']
            missing_cols = [col for col in required_cols if col not in training_data.columns]
            
            if missing_cols:
                print(f"❌ Missing required columns: {missing_cols}")
                return None
            
            print(f"✅ Training data prepared: {len(training_data)} samples with {len(training_data.columns)} features")
            
            return training_data
            
        except Exception as e:
            print(f"❌ Error preparing training data: {e}")
            print(f"Available columns in players_df: {list(players_df.columns)}")
            print(f"Available columns in performance_df: {list(performance_df.columns)}")
            return None
    
    def _train_ensemble_models(self, X_train, y_train, X_test, y_test):
        """Train ensemble of models"""
        try:
            import xgboost as xgb
            import lightgbm as lgb
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            
            models = {}
            
            print("   🔄 Training XGBoost...")
            xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            xgb_model.fit(X_train, y_train)
            models['xgboost'] = xgb_model
            
            print("   🔄 Training LightGBM...")
            lgb_model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
            lgb_model.fit(X_train, y_train)
            models['lightgbm'] = lgb_model
            
            print("   🔄 Training Random Forest...")
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            models['random_forest'] = rf_model
            
            print("   🔄 Training Gradient Boosting...")
            gb_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            gb_model.fit(X_train, y_train)
            models['gradient_boosting'] = gb_model
            
            return models
            
        except Exception as e:
            print(f"❌ Error training models: {e}")
            return None
    
    def _validate_new_models(self, new_models, X_test, y_test):
        """Validate new models against current ones"""
        try:
            from sklearn.metrics import mean_squared_error
            
            # Get best new model performance
            best_new_rmse = float('inf')
            for name, model in new_models.items():
                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                print(f"   📊 New {name} RMSE: {rmse:.3f}")
                
                if rmse < best_new_rmse:
                    best_new_rmse = rmse
            
            # Load current model metadata for comparison
            metadata_files = list(self.models_dir.glob('model_metadata_*.json'))
            if metadata_files:
                latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
                with open(latest_metadata, 'r') as f:
                    metadata = json.load(f)
                
                current_best_rmse = metadata.get('model_performance', {}).get('xgboost', {}).get('test_rmse', float('inf'))
                
                print(f"   📈 Current best RMSE: {current_best_rmse:.3f}")
                print(f"   📈 New best RMSE: {best_new_rmse:.3f}")
                
                # Require meaningful improvement (at least 1% better)
                improvement_threshold = current_best_rmse * 0.99
                
                if best_new_rmse < improvement_threshold:
                    print(f"   ✅ New models show improvement: {((current_best_rmse - best_new_rmse) / current_best_rmse * 100):.1f}%")
                    return True
                else:
                    print(f"   ⚠️ New models do not show significant improvement")
                    return False
            else:
                print("   ✅ No current models - accepting new models")
                return True
                
        except Exception as e:
            print(f"❌ Error validating models: {e}")
            return False
    
    def _save_retrained_models(self, models, X_train, y_train, feature_cols):
        """Save retrained models with versioning"""
        try:
            import joblib
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            print("💾 Saving retrained models...")
            
            # Save best model (XGBoost)
            best_model_path = self.models_dir / f"xgboost_best_{timestamp}.joblib"
            joblib.dump(models['xgboost'], best_model_path)
            print(f"   ✅ Saved XGBoost: {best_model_path.name}")
            
            # Save ensemble models
            ensemble_path = self.models_dir / f"ensemble_models_{timestamp}.joblib"
            joblib.dump(models, ensemble_path)
            print(f"   ✅ Saved ensemble: {ensemble_path.name}")
            
            # Save preprocessors
            preprocessors = {
                'feature_names': feature_cols,
                'scaler': None,  # Add if using scaling
                'label_encoders': {}  # Add if using encoding
            }
            
            preprocessor_path = self.models_dir / f"preprocessors_{timestamp}.joblib"
            joblib.dump(preprocessors, preprocessor_path)
            print(f"   ✅ Saved preprocessors: {preprocessor_path.name}")
            
            # Create metadata
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            model_performance = {}
            for name, model in models.items():
                y_pred = model.predict(X_train)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
                train_mae = mean_absolute_error(y_train, y_pred)
                
                model_performance[name] = {
                    'train_rmse': train_rmse,
                    'train_mae': train_mae
                }
            
            # Get current gameweek info
            try:
                fixtures_df = pd.read_csv(self.data_dir / 'fixtures_latest.csv')
                completed_gameweeks = fixtures_df[fixtures_df['finished'] == True]['event'].unique()
                latest_completed_gw = max(completed_gameweeks) if len(completed_gameweeks) > 0 else 0
                gameweeks_trained = f"1-{latest_completed_gw}"
            except:
                gameweeks_trained = "Unknown"
            
            metadata = {
                'timestamp': timestamp,
                'model_performance': model_performance,
                'dataset_info': {
                    'training_samples': len(X_train),
                    'features_count': len(feature_cols),
                    'gameweeks_trained': gameweeks_trained
                },
                'retrained': True,
                'previous_model_improved': True
            }
            
            metadata_path = self.models_dir / f"model_metadata_{timestamp}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"   ✅ Saved metadata: {metadata_path.name}")
            
            # Update current model references
            self._load_models()  # Reload to use new models
            
            print(f"🎯 Model retraining complete! New models active.")
            
        except Exception as e:
            print(f"❌ Error saving retrained models: {e}")
    
    def _export_comprehensive_data_pipeline(self, timestamp):
        """Export all FPL data in standardized formats for forms and models"""
        try:
            # Create exports directory
            exports_dir = self.data_dir / "exports"
            exports_dir.mkdir(exist_ok=True)
            
            print("🔄 Creating comprehensive data exports...")
            
            # 1. PLAYERS EXPORT - Complete player database
            players_df = pd.read_csv(self.data_dir / "players_latest.csv")
            teams_df = pd.read_csv(self.data_dir / "teams_latest.csv")
            
            # Create team lookup for player exports
            team_lookup = dict(zip(teams_df['id'], teams_df['short_name']))
            
            # Enhanced players export
            players_export = players_df.copy()
            players_export['team_short_name'] = players_export['team'].map(team_lookup)
            players_export['position_name'] = players_export['element_type'].map({
                1: 'Goalkeeper', 2: 'Defender', 3: 'Midfielder', 4: 'Forward'
            })
            players_export['price_millions'] = players_export['now_cost'] / 10
            players_export['value_score'] = players_export['total_points'] / players_export['price_millions']
            
            # Save players export
            players_export_file = exports_dir / "all_players.json"
            players_export.to_json(players_export_file, orient='records', indent=2)
            print(f"✅ Players export: {len(players_export)} players → {players_export_file.name}")
            
            # 2. TEAMS EXPORT - Complete team data
            teams_export = teams_df.copy()
            teams_export_file = exports_dir / "all_teams.json"
            teams_export.to_json(teams_export_file, orient='records', indent=2)
            print(f"✅ Teams export: {len(teams_export)} teams → {teams_export_file.name}")
            
            # 3. FIXTURES EXPORT - All fixtures with enhanced data
            fixtures_df = pd.read_csv(self.data_dir / "fixtures_latest.csv")
            fixtures_export = fixtures_df.copy()
            
            # Add team names to fixtures
            fixtures_export['home_team_name'] = fixtures_export['team_h'].map(team_lookup)
            fixtures_export['away_team_name'] = fixtures_export['team_a'].map(team_lookup)
            
            # Add difficulty labels
            difficulty_map = {1: 'Very Easy', 2: 'Easy', 3: 'Medium', 4: 'Hard', 5: 'Very Hard'}
            fixtures_export['home_difficulty_label'] = fixtures_export['team_h_difficulty'].map(difficulty_map)
            fixtures_export['away_difficulty_label'] = fixtures_export['team_a_difficulty'].map(difficulty_map)
            
            fixtures_export_file = exports_dir / "all_fixtures.json"
            fixtures_export.to_json(fixtures_export_file, orient='records', indent=2)
            print(f"✅ Fixtures export: {len(fixtures_export)} fixtures → {fixtures_export_file.name}")
            
            # 4. GAMEWEEKS EXPORT - Season structure
            gameweeks_df = pd.read_csv(self.data_dir / "gameweeks_latest.csv")
            gameweeks_export_file = exports_dir / "all_gameweeks.json"
            gameweeks_df.to_json(gameweeks_export_file, orient='records', indent=2)
            print(f"✅ Gameweeks export: {len(gameweeks_df)} gameweeks → {gameweeks_export_file.name}")
            
            # 5. POSITION-BASED EXPORTS - For form selections
            positions = {
                1: ('goalkeepers', 'GKP'),
                2: ('defenders', 'DEF'), 
                3: ('midfielders', 'MID'),
                4: ('forwards', 'FWD')
            }
            
            for pos_id, (pos_name, pos_short) in positions.items():
                pos_players = players_export[players_export['element_type'] == pos_id].copy()
                pos_players = pos_players.sort_values('total_points', ascending=False)
                
                pos_file = exports_dir / f"{pos_name}.json"
                pos_players.to_json(pos_file, orient='records', indent=2)
                print(f"✅ {pos_short} players: {len(pos_players)} players → {pos_file.name}")
            
            # 6. TOP PERFORMERS EXPORT - For quick selections
            top_performers = {
                'top_scorers': players_export.nlargest(50, 'total_points'),
                'best_value': players_export.nlargest(50, 'value_score'),
                'most_selected': players_export.nlargest(50, 'selected_by_percent'),
                'in_form': players_export.nlargest(50, 'form')
            }
            
            for category, data in top_performers.items():
                top_file = exports_dir / f"{category}.json"
                data.to_json(top_file, orient='records', indent=2)
                print(f"✅ {category.replace('_', ' ').title()}: {len(data)} players → {top_file.name}")
            
            # 7. CURRENT GAMEWEEK DATA - For immediate use
            current_gw = gameweeks_df[gameweeks_df['is_current'] == True]
            if not current_gw.empty:
                current_gw_num = current_gw.iloc[0]['id']
                
                # Current gameweek fixtures
                current_fixtures = fixtures_export[fixtures_export['event'] == current_gw_num]
                current_fixtures_file = exports_dir / "current_gameweek_fixtures.json"
                current_fixtures.to_json(current_fixtures_file, orient='records', indent=2)
                print(f"✅ Current GW{current_gw_num} fixtures: {len(current_fixtures)} matches → {current_fixtures_file.name}")
            
            # 8. SUMMARY METADATA - For form validation
            metadata = {
                'last_updated': timestamp,
                'data_source': 'official_fpl_api',
                'total_players': int(len(players_export)),
                'total_teams': int(len(teams_export)),
                'total_fixtures': int(len(fixtures_export)),
                'total_gameweeks': int(len(gameweeks_df)),
                'current_gameweek': int(current_gw_num) if not current_gw.empty else None,
                'price_range': {
                    'min': float(players_export['price_millions'].min()),
                    'max': float(players_export['price_millions'].max())
                },
                'export_files': [
                    'all_players.json', 'all_teams.json', 'all_fixtures.json', 'all_gameweeks.json',
                    'goalkeepers.json', 'defenders.json', 'midfielders.json', 'forwards.json',
                    'top_scorers.json', 'best_value.json', 'most_selected.json', 'in_form.json',
                    'current_gameweek_fixtures.json'
                ]
            }
            
            metadata_file = exports_dir / "data_metadata.json"
            with open(metadata_file, 'w') as f:
                import json
                json.dump(metadata, f, indent=2)
            print(f"✅ Metadata export: Complete data summary → {metadata_file.name}")
            
            # 9. CSV EXPORTS - For spreadsheet compatibility
            csv_dir = exports_dir / "csv"
            csv_dir.mkdir(exist_ok=True)
            
            players_export.to_csv(csv_dir / "all_players.csv", index=False)
            teams_export.to_csv(csv_dir / "all_teams.csv", index=False) 
            fixtures_export.to_csv(csv_dir / "all_fixtures.csv", index=False)
            print(f"✅ CSV exports: 3 files → csv/ directory")
            
            print(f"\n📦 EXPORT SUMMARY:")
            print(f"   📁 Export directory: {exports_dir}")
            print(f"   📄 JSON files: {len(metadata['export_files'])} files")
            print(f"   📊 CSV files: 3 files")
            print(f"   🏷️ Metadata file: data_metadata.json")
            print(f"   🕐 Last updated: {timestamp}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in data export pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_position_name(self, pos_id):
        """Convert position ID to name"""
        pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        return pos_map.get(pos_id, 'UNK')


def main():
    parser = argparse.ArgumentParser(description='FPL Predictor - Maximum Accuracy Terminal Interface')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict points for a specific player')
    predict_parser.add_argument('--player', '-p', required=True, help='Player name or partial name')
    
    # Top picks command
    picks_parser = subparsers.add_parser('top-picks', help='Get top predicted players')
    picks_parser.add_argument('--position', choices=['GKP', 'DEF', 'MID', 'FWD'], help='Filter by position')
    picks_parser.add_argument('--budget', type=float, help='Maximum price per player')
    picks_parser.add_argument('--limit', type=int, default=10, help='Number of players to show')
    
    # Team optimization command
    team_parser = subparsers.add_parser('optimize-team', help='Optimize full team selection')
    team_parser.add_argument('--budget', type=float, default=100.0, help='Total team budget')
    
    # Fixture analysis command
    fixture_parser = subparsers.add_parser('fixtures', help='Analyze fixture difficulty by team')
    fixture_parser.add_argument('--gameweeks', default='5-10', help='Gameweek range (e.g., "5-10")')
    
    # Fixture-based picks command
    fixture_picks_parser = subparsers.add_parser('fixture-picks', help='Get picks based on fixture difficulty')
    fixture_picks_parser.add_argument('--gameweeks', default='5-10', help='Gameweek range (e.g., "5-10")')
    fixture_picks_parser.add_argument('--position', choices=['GKP', 'DEF', 'MID', 'FWD'], help='Filter by position')
    fixture_picks_parser.add_argument('--limit', type=int, default=10, help='Number of players to show')
    
    # Data update command
    subparsers.add_parser('update-data', help='Update FPL data from API')
    
    # Data export command
    subparsers.add_parser('export-data', help='Export all FPL data to standardized files for forms/models')
    
    # Performance history command
    history_parser = subparsers.add_parser('collect-history', help='Collect historical performance data for all players')
    history_parser.add_argument('--gameweeks', type=int, default=4, help='Number of past gameweeks to collect (default: 4)')
    history_parser.add_argument('--save', action='store_true', help='Save collected data to files')
    
    # Current team analysis command
    team_analysis_parser = subparsers.add_parser('analyze-team', help='Analyze your current FPL team')
    team_analysis_parser.add_argument('--team-id', type=int, help='Your FPL team ID (e.g., 5135491)')
    
    # Transfer suggestions command
    difficulty_parser = subparsers.add_parser('fixture-difficulty', help='Show fixture difficulty for each team by gameweek')
    difficulty_parser.add_argument('--gameweeks', default='5-10', help='Gameweek range for analysis (e.g., "5-10" or "5")')
    difficulty_parser.add_argument('--team', help='Filter by specific team (e.g., "CHE", "LIV")')
    
    # Matches display command
    matches_parser = subparsers.add_parser('matches', help='Display all matches for a specific gameweek/round')
    matches_parser.add_argument('--gameweek', type=int, required=True, help='Gameweek number (e.g., 5)')
    matches_parser.add_argument('--format', choices=['simple', 'detailed', 'json', 'ai-analysis'], default='simple', 
                               help='Output format: simple (clean), detailed (with stats), json (for models), ai-analysis (auto AI analysis)')
    matches_parser.add_argument('--save-for-models', action='store_true', 
                               help='Save match data in format suitable for AI model consumption')
    
    # Fetch team command
    fetch_parser = subparsers.add_parser('fetch-team', help='Fetch and save current team from FPL API')
    fetch_parser.add_argument('--team-id', type=int, required=True, help='Your FPL team ID (e.g., 5135491)')
    
    # Chip strategy commands
    chip_parser = subparsers.add_parser('chip-advice', help='Get comprehensive chip strategy recommendations')
    chip_parser.add_argument('--gameweeks', default='5-15', help='Gameweek range for analysis (e.g., "5-15")')
    
    # Specific chip timing analysis
    timing_parser = subparsers.add_parser('chip-timing', help='Analyze optimal timing for specific chip')
    timing_parser.add_argument('--chip', choices=['wildcard', 'bench_boost', 'triple_captain', 'free_hit', 'all'], 
                              default='all', help='Specific chip to analyze')
    timing_parser.add_argument('--gameweeks', default='5-15', help='Gameweek range for analysis (e.g., "5-15")')
    
    # Model retraining command
    retrain_parser = subparsers.add_parser('retrain-models', help='Manually retrain ML models with latest data')
    retrain_parser.add_argument('--force', action='store_true', help='Force retraining even if models are current')
    
    # Model health check command
    health_parser = subparsers.add_parser('health-check', help='Run comprehensive model health check and validation')
    health_parser.add_argument('--detailed', action='store_true', help='Show detailed validation results')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize predictor
    predictor = FPLPredictor(verbose=args.verbose)
    
    # Execute command
    if args.command == 'predict':
        predictor.predict_player(args.player)
    
    elif args.command == 'top-picks':
        predictor.top_picks(
            position=args.position,
            max_price=args.budget,
            limit=args.limit
        )
    
    elif args.command == 'optimize-team':
        predictor.optimize_team(budget=args.budget)
    
    elif args.command == 'fixtures':
        try:
            start_gw, end_gw = map(int, args.gameweeks.split('-'))
            predictor.analyze_fixtures(start_gw, end_gw)
        except:
            print("❌ Invalid gameweek format. Use format like '5-10'")
    
    elif args.command == 'fixture-picks':
        predictor.fixture_based_picks(
            gameweeks=args.gameweeks,
            position=args.position,
            limit=args.limit
        )
    
    elif args.command == 'update-data':
        predictor.update_data()
    
    elif args.command == 'export-data':
        # Export data using current timestamp without updating
        print("📤 Exporting existing FPL data to standardized files...")
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check if data exists
        if not (predictor.data_dir / "players_latest.csv").exists():
            print("❌ No data found. Run 'update-data' first.")
            return
        
        predictor._export_comprehensive_data_pipeline(timestamp)
    
    elif args.command == 'collect-history':
        # Collect historical performance data
        print(f"🔄 Collecting last {args.gameweeks} gameweeks of performance data...")
        
        # Check if base data exists
        if not (predictor.data_dir / "players_latest.csv").exists():
            print("❌ No player data found. Run 'update-data' first.")
            return
        
        # Collect performance history
        performance_data = predictor.collect_player_performance_history(args.gameweeks)
        
        if args.save and performance_data:
            # Save the collected data
            history_dir = predictor.save_performance_history(performance_data)
            print(f"\n📁 Performance history saved to: {history_dir}")
            print("📊 Files created:")
            print("   • all_players_performance.json - Complete dataset")
            print("   • goalkeepers_performance.json - GK performance data")
            print("   • defenders_performance.json - DEF performance data") 
            print("   • midfielders_performance.json - MID performance data")
            print("   • forwards_performance.json - FWD performance data")
            print("   • model_features.csv - Model-ready features")
            print("   • feature_metadata.json - Feature descriptions")
        else:
            print(f"\n✅ Collected performance data for {len(performance_data)} players")
            print("   Use --save flag to save data to files")
    
    elif args.command == 'analyze-team':
        predictor.analyze_current_team(args.team_id)
    
    elif args.command == 'fixture-difficulty':
        predictor.display_fixture_difficulty(args.gameweeks, args.team)
    
    elif args.command == 'matches':
        predictor.display_matches(args.gameweek, args.format, getattr(args, 'save_for_models', False))
    
    elif args.command == 'fetch-team':
        predictor.fetch_current_team(args.team_id)
    
    elif args.command == 'chip-advice':
        predictor.get_chip_recommendations(args.gameweeks)
    
    elif args.command == 'chip-timing':
        predictor.analyze_chip_timing(args.chip, args.gameweeks)
    
    elif args.command == 'retrain-models':
        if args.force or predictor._should_retrain_models():
            print("🚀 Starting model retraining...")
            if predictor._retrain_models_with_new_data():
                print("✅ Model retraining completed successfully!")
            else:
                print("❌ Model retraining failed!")
        else:
            print("ℹ️ Models are already current. Use --force to retrain anyway.")
    
    elif args.command == 'health-check':
        health_status = predictor._run_model_health_check()
        
        if args.detailed and isinstance(health_status, dict) and 'error' not in health_status:
            print(f"\n📋 Detailed Health Report:")
            print("-" * 30)
            for check, status in health_status.items():
                status_icon = "✅" if status else "❌"
                print(f"{status_icon} {check.replace('_', ' ').title()}: {'PASS' if status else 'FAIL'}")


if __name__ == '__main__':
    main()