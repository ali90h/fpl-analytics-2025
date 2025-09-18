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
import pandas as pd
import numpy as np
import sqlite3
import joblib
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
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("💡 Run the training notebook first to create models")
            sys.exit(1)
    
    def _load_current_data(self):
        """Load current FPL data from CSV file"""
        try:
            # Load from enhanced CSV file first
            csv_path = self.data_dir / 'enhanced_fpl_features.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                
                # Get latest gameweek data
                latest_gameweek = df['gameweek'].max()
                df = df[df['gameweek'] == latest_gameweek].copy()
                
                self._log(f"📊 Loaded {len(df)} players from gameweek {latest_gameweek} (CSV)")
                return df
            
            # Fallback to database
            conn = sqlite3.connect(self.db_path)
            
            # Try different table names
            tables = ['enhanced_fpl_data', 'fpl_data', 'player_gameweeks']
            df = pd.DataFrame()
            
            for table in tables:
                try:
                    query = f"SELECT * FROM {table} ORDER BY gameweek DESC LIMIT 1000"
                    df = pd.read_sql_query(query, conn)
                    if not df.empty:
                        self._log(f"📊 Loaded {len(df)} players from table '{table}'")
                        break
                except:
                    continue
            
            conn.close()
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("💡 Make sure the enhanced_fpl_features.csv file exists")
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
        
        prediction = self.model.predict(X)[0]
        
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
        
        predictions = self.model.predict(X)
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
        
        predictions = self.model.predict(X)
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
        """Analyze fixture difficulty for teams across gameweeks"""
        print(f"📅 Analyzing fixture difficulty for gameweeks {start_gw}-{end_gw}...")
        
        # This is a simplified fixture difficulty analysis
        # In a full implementation, you'd fetch actual fixture data from FPL API
        
        # For now, let's analyze team performance patterns from our data
        df = self._load_current_data()
        if df.empty:
            return None
        
        # Group by team and calculate team strength metrics
        team_analysis = []
        
        for team in df['team'].unique():
            team_players = df[df['team'] == team].copy()
            
            if len(team_players) == 0:
                continue
            
            # Prepare features and get predictions for this team
            X = self._prepare_features(team_players)
            if X is None:
                continue
            
            predictions = self.model.predict(X)
            team_players['predicted_points'] = predictions
            
            # Calculate team metrics
            avg_predicted = team_players['predicted_points'].mean()
            total_predicted = team_players['predicted_points'].sum()
            avg_form = team_players.get('form', pd.Series([0] * len(team_players))).mean()
            
            # Calculate attacking and defensive strength
            defenders = team_players[team_players['element_type'] == 2]
            attackers = team_players[team_players['element_type'].isin([3, 4])]
            
            def_strength = defenders['predicted_points'].mean() if len(defenders) > 0 else 0
            att_strength = attackers['predicted_points'].mean() if len(attackers) > 0 else 0
            
            team_analysis.append({
                'team': team,
                'avg_predicted_points': avg_predicted,
                'total_predicted_points': total_predicted,
                'avg_form': avg_form,
                'defensive_strength': def_strength,
                'attacking_strength': att_strength,
                'player_count': len(team_players)
            })
        
        # Sort by various metrics
        team_df = pd.DataFrame(team_analysis)
        
        print(f"\n🏆 Team Strength Analysis (GW{start_gw}-{end_gw}):")
        print(f"{'Rank':<4} {'Team':<12} {'Avg Pts':<8} {'Form':<6} {'Def':<6} {'Att':<6} {'Difficulty':<10}")
        print("=" * 65)
        
        # Calculate fixture difficulty (lower avg predicted points = harder opponents)
        team_df['fixture_difficulty'] = 'Medium'
        team_df = team_df.sort_values('avg_predicted_points', ascending=False)
        
        # Assign difficulty ratings
        total_teams = len(team_df)
        for i, (_, team) in enumerate(team_df.iterrows()):
            if i < total_teams * 0.3:  # Top 30% - easiest
                difficulty = 'Easy'
            elif i > total_teams * 0.7:  # Bottom 30% - hardest  
                difficulty = 'Hard'
            else:
                difficulty = 'Medium'
            
            team_df.loc[team.name, 'fixture_difficulty'] = difficulty
            
            print(f"{i+1:<4} {team['team']:<12} {team['avg_predicted_points']:<8.2f} "
                  f"{team['avg_form']:<6.1f} {team['defensive_strength']:<6.2f} "
                  f"{team['attacking_strength']:<6.2f} {difficulty:<10}")
        
        return team_df
    
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
        
        # Get teams with easy fixtures
        easy_teams = team_analysis[team_analysis['fixture_difficulty'] == 'Easy']['team'].tolist()
        
        print(f"\n✅ Teams with easiest fixtures: {', '.join(easy_teams)}")
        
        # Load player data and filter by easy fixture teams
        df = self._load_current_data()
        if df.empty:
            return None
        
        easy_fixture_players = df[df['team'].isin(easy_teams)].copy()
        
        # Filter by position if specified
        if position:
            pos_map = {'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
            if position.upper() in pos_map:
                easy_fixture_players = easy_fixture_players[
                    easy_fixture_players['element_type'] == pos_map[position.upper()]
                ]
        
        if len(easy_fixture_players) == 0:
            print("❌ No players found with easy fixtures for the specified criteria")
            return None
        
        # Get predictions
        X = self._prepare_features(easy_fixture_players)
        if X is None:
            return None
        
        predictions = self.model.predict(X)
        easy_fixture_players['predicted_points'] = predictions
        easy_fixture_players['value_rating'] = predictions / (easy_fixture_players['now_cost'] / 10)
        
        # Sort by predicted points
        top_easy_picks = easy_fixture_players.nlargest(limit, 'predicted_points')
        
        pos_str = f" {position.upper()}" if position else ""
        print(f"\n🌟 Top{pos_str} Picks (Easy Fixtures GW{start_gw}-{end_gw}):")
        print(f"{'Rank':<4} {'Player':<18} {'Team':<5} {'Pred':<6} {'Price':<7} {'Value':<6}")
        print("=" * 55)
        
        for i, (_, player) in enumerate(top_easy_picks.iterrows(), 1):
            print(f"{i:<4} {player['web_name']:<18} {player['team']:<5} "
                  f"{player['predicted_points']:<6.2f} £{player['now_cost']/10:<6.1f} "
                  f"{player['value_rating']:<6.2f}")
        
        return top_easy_picks[['web_name', 'team', 'predicted_points', 'now_cost', 'value_rating']].to_dict('records')

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
        """Analyze current team and provide recommendations"""
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
        
        # Analyze each player
        total_predicted = 0
        starters_predicted = 0
        bench_predicted = 0
        
        for pick in current_team['picks']:
            # Get prediction for this player
            prediction = self.predict_player_by_name(pick['web_name'], verbose=False)
            
            if prediction:
                pick['predicted_points'] = prediction['predicted_points']
                total_predicted += prediction['predicted_points']
                
                if pick['is_starter']:
                    starters_predicted += prediction['predicted_points']
                else:
                    bench_predicted += prediction['predicted_points']
            else:
                pick['predicted_points'] = 0
            
            # Display player info
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
        
        return current_team
    
    def suggest_transfers(self, team_id=None, gameweeks="5-10"):
        """Suggest transfers based on current team and fixture analysis"""
        current_team = self.analyze_current_team(team_id)
        if not current_team:
            return None
        
        print(f"\n🔄 Transfer Suggestions (GW{gameweeks}):")
        print("=" * 60)
        
        # Get fixture analysis
        start_gw, end_gw = map(int, gameweeks.split('-'))
        fixture_analysis = self.analyze_fixtures(start_gw, end_gw)
        if fixture_analysis is None or fixture_analysis.empty:
            return None
        
        # Get easy fixture teams
        easy_teams = fixture_analysis[fixture_analysis['fixture_difficulty'] == 'Easy']['team'].tolist()
        
        suggestions = []
        
        # Analyze each position
        for position in ['GKP', 'DEF', 'MID', 'FWD']:
            current_players = [p for p in current_team['picks'] if p['position_name'] == position]
            
            # Get best options for this position with easy fixtures
            fixture_picks = self.fixture_based_picks(gameweeks, position, limit=8)
            if not fixture_picks:
                continue
            
            # Find transfer opportunities
            for current_player in current_players:
                # Find better alternatives
                current_pred = current_player.get('predicted_points', 0)
                current_cost = current_player['now_cost'] / 10
                
                for alternative in fixture_picks:
                    alt_pred = alternative['predicted_points']
                    alt_cost = alternative['now_cost'] / 10
                    
                    # Check if alternative is significantly better
                    if (alt_pred > current_pred + 1.0 and  # At least 1 point better
                        alternative['web_name'] != current_player['web_name'] and  # Not same player
                        alt_cost <= current_cost + current_team['bank']):  # Affordable
                        
                        suggestion = {
                            'out': current_player['web_name'],
                            'in': alternative['web_name'],
                            'position': position,
                            'points_diff': alt_pred - current_pred,
                            'cost_diff': alt_cost - current_cost,
                            'current_pred': current_pred,
                            'new_pred': alt_pred
                        }
                        suggestions.append(suggestion)
        
        # Sort by points improvement
        suggestions.sort(key=lambda x: x['points_diff'], reverse=True)
        
        if suggestions:
            print("🎯 Recommended Transfers:")
            for i, suggestion in enumerate(suggestions[:5], 1):  # Top 5 suggestions
                cost_str = f"+£{suggestion['cost_diff']:.1f}m" if suggestion['cost_diff'] > 0 else f"£{suggestion['cost_diff']:.1f}m"
                print(f"{i}. {suggestion['out']} → {suggestion['in']} ({suggestion['position']})")
                print(f"   Points: {suggestion['current_pred']:.1f} → {suggestion['new_pred']:.1f} (+{suggestion['points_diff']:.1f})")
                print(f"   Cost: {cost_str}")
                print()
        else:
            print("✅ Your team looks good! No obvious improvements found.")
        
        return suggestions

    def update_data(self):
        """Fetch latest FPL data"""
        print("🔄 Fetching latest FPL data...")
        print("💡 This is a placeholder - implement FPL API integration here")
        print("📊 For now, use the existing data from your training notebooks")
        return True
    
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
    
    # Current team analysis command
    team_analysis_parser = subparsers.add_parser('analyze-team', help='Analyze your current FPL team')
    team_analysis_parser.add_argument('--team-id', type=int, help='Your FPL team ID (e.g., 5135491)')
    
    # Transfer suggestions command
    transfer_parser = subparsers.add_parser('suggest-transfers', help='Get transfer suggestions based on fixtures')
    transfer_parser.add_argument('--team-id', type=int, help='Your FPL team ID (e.g., 5135491)')
    transfer_parser.add_argument('--gameweeks', default='5-10', help='Gameweek range for analysis (e.g., "5-10")')
    
    # Fetch team command
    fetch_parser = subparsers.add_parser('fetch-team', help='Fetch and save current team from FPL API')
    fetch_parser.add_argument('--team-id', type=int, required=True, help='Your FPL team ID (e.g., 5135491)')
    
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
    
    elif args.command == 'analyze-team':
        predictor.analyze_current_team(args.team_id)
    
    elif args.command == 'suggest-transfers':
        predictor.suggest_transfers(args.team_id, args.gameweeks)
    
    elif args.command == 'fetch-team':
        predictor.fetch_current_team(args.team_id)


if __name__ == '__main__':
    main()