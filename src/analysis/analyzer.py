"""
Football Analytics Analyzer - Terminal-based analysis and prediction system
Provides focused insights for 2025/26 season fantasy football decisions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sqlite3
import logging
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

class FootballAnalyzer:
    """
    Provides analysis and prediction capabilities for fantasy football
    Optimized for terminal-based decision making
    """
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.season = data_manager.season
        
        # Analysis configuration
        self.min_minutes_threshold = 60  # Minimum minutes for analysis
        self.form_gameweeks = 5  # Gameweeks to consider for form
        self.prediction_confidence_threshold = 0.7
        
        logger.info(f"Football Analyzer initialized for season {self.season}")
    
    def analyze_player_performance(self, player_id: int, detailed: bool = False) -> Dict:
        """
        Analyze individual player performance with focus on fantasy relevance
        """
        try:
            # Get player data
            player_data = self.data_manager.get_player_data(player_id=player_id)
            
            if player_data.empty:
                return {"error": f"No data found for player {player_id}"}
            
            # Basic player info
            latest_row = player_data.iloc[0]
            
            analysis = {
                "player_id": player_id,
                "name": latest_row.get('name', 'Unknown'),
                "team": latest_row.get('team_name', 'Unknown'),
                "position": latest_row.get('position_short', 'Unknown'),
                "current_price": latest_row.get('price', 0),
                "total_points": latest_row.get('total_points', 0)
            }
            
            # Calculate key metrics
            if len(player_data) > 0:
                points_per_game = self._calculate_points_per_game(player_data)
                consistency_score = self._calculate_consistency(player_data)
                form_trend = self._calculate_form_trend(player_data)
                value_rating = self._calculate_value_rating(player_data)
                injury_risk = self._assess_injury_risk(player_data)
                
                analysis.update({
                    "points_per_game": round(points_per_game, 2),
                    "consistency_score": round(consistency_score, 2),
                    "form_trend": form_trend,
                    "value_rating": round(value_rating, 2),
                    "injury_risk": injury_risk,
                    "recommendation": self._generate_player_recommendation(
                        points_per_game, consistency_score, form_trend, value_rating, injury_risk
                    )
                })
                
                if detailed:
                    # Add detailed metrics for comprehensive analysis
                    detailed_metrics = self._calculate_detailed_metrics(player_data)
                    analysis.update(detailed_metrics)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing player {player_id}: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def analyze_gameweek_top_performers(self, gameweek: int, count: int = 10) -> List[Dict]:
        """
        Analyze top performers for a specific gameweek
        """
        try:
            summary = self.data_manager.get_gameweek_summary(gameweek)
            
            if not summary.get('top_performers'):
                return []
            
            performers = []
            for performer in summary['top_performers'][:count]:
                # Add analysis context
                analysis = {
                    "name": performer['name'],
                    "team": performer['team_name'],
                    "position": performer['position_short'],
                    "points": performer['points'],
                    "opponent": performer['opponent'],
                    "venue": performer['venue'],
                    "performance_rating": self._rate_performance(performer['points'], performer['position_short'])
                }
                performers.append(analysis)
            
            return performers
            
        except Exception as e:
            logger.error(f"Error analyzing gameweek {gameweek} performers: {str(e)}")
            return []
    
    def predict_player_points(self, player_id: int, upcoming_opponents: List[str] = None) -> Dict:
        """
        Predict player points for upcoming gameweeks
        """
        try:
            player_data = self.data_manager.get_player_data(player_id=player_id, gameweeks=10)
            
            if player_data.empty:
                return {"error": f"Insufficient data for player {player_id}"}
            
            # Calculate baseline prediction
            recent_avg = player_data['points'].mean()
            form_multiplier = self._calculate_form_multiplier(player_data)
            
            # Difficulty adjustment (if opponent data available)
            difficulty_adjustment = 1.0
            if upcoming_opponents:
                difficulty_adjustment = self._calculate_difficulty_adjustment(upcoming_opponents)
            
            predicted_points = recent_avg * form_multiplier * difficulty_adjustment
            
            # Calculate confidence
            consistency = self._calculate_consistency(player_data)
            sample_size_factor = min(len(player_data) / 5, 1.0)  # Max confidence with 5+ games
            confidence = consistency * sample_size_factor
            
            prediction = {
                "player_id": player_id,
                "predicted_points": round(predicted_points, 1),
                "confidence": round(confidence, 2),
                "baseline_avg": round(recent_avg, 1),
                "form_multiplier": round(form_multiplier, 2),
                "difficulty_adjustment": round(difficulty_adjustment, 2),
                "recommendation": "BUY" if predicted_points > 6 and confidence > 0.7 else 
                               "HOLD" if predicted_points > 4 else "SELL"
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting points for player {player_id}: {str(e)}")
            return {"error": f"Prediction failed: {str(e)}"}
    
    def analyze_position_trends(self, gameweeks: int = 5) -> Dict:
        """
        Analyze scoring trends by position
        """
        try:
            with sqlite3.connect(self.data_manager.db_path) as conn:
                # Get recent gameweek performances by position
                query = '''
                    SELECT p.position_short, gp.gameweek, gp.points, gp.minutes
                    FROM gameweek_performances gp
                    JOIN players p ON gp.player_id = p.id
                    WHERE gp.gameweek > (SELECT MAX(gameweek) FROM gameweek_performances) - ?
                    AND gp.minutes >= ?
                '''
                df = pd.read_sql_query(query, conn, params=(gameweeks, self.min_minutes_threshold))
            
            if df.empty:
                return {"error": "No recent performance data available"}
            
            position_analysis = {}
            
            for position in df['position_short'].unique():
                pos_data = df[df['position_short'] == position]
                
                position_analysis[position] = {
                    "avg_points": round(pos_data['points'].mean(), 2),
                    "total_games": len(pos_data),
                    "high_scorers": len(pos_data[pos_data['points'] >= 8]),  # 8+ points
                    "consistency": round(1 - (pos_data['points'].std() / pos_data['points'].mean()), 2) if pos_data['points'].mean() > 0 else 0,
                    "trend": self._calculate_position_trend(pos_data)
                }
            
            # Overall insights
            insights = self._generate_position_insights(position_analysis)
            
            return {
                "analysis_period": f"Last {gameweeks} gameweeks",
                "positions": position_analysis,
                "insights": insights,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            logger.error(f"Error analyzing position trends: {str(e)}")
            return {"error": f"Position analysis failed: {str(e)}"}
    
    def find_value_picks(self, max_price: float = 7.0, min_gameweeks: int = 3) -> List[Dict]:
        """
        Find undervalued players (value picks) based on price vs performance
        """
        try:
            with sqlite3.connect(self.data_manager.db_path) as conn:
                # Get players with recent performances and reasonable price
                query = '''
                    SELECT p.id, p.name, p.team_name, p.position_short, p.price,
                           COUNT(gp.gameweek) as games_played,
                           AVG(gp.points) as avg_points,
                           SUM(gp.points) as total_points,
                           AVG(gp.minutes) as avg_minutes
                    FROM players p
                    JOIN gameweek_performances gp ON p.id = gp.player_id
                    WHERE p.price <= ? AND gp.minutes >= ?
                    GROUP BY p.id
                    HAVING games_played >= ?
                '''
                df = pd.read_sql_query(query, conn, params=(max_price, self.min_minutes_threshold, min_gameweeks))
            
            if df.empty:
                return []
            
            value_picks = []
            
            for _, player in df.iterrows():
                # Calculate value metrics
                points_per_million = player['avg_points'] / player['price']
                total_value = player['total_points'] / player['price']
                
                # Get recent form
                recent_data = self.data_manager.get_player_data(player_id=player['id'], gameweeks=3)
                recent_form = recent_data['points'].mean() if not recent_data.empty else 0
                
                value_score = (points_per_million * 0.4 + total_value * 0.3 + recent_form * 0.3)
                
                if value_score >= 1.0:  # Threshold for value picks
                    pick = {
                        "player_id": int(player['id']),
                        "name": player['name'],
                        "team": player['team_name'],
                        "position": player['position_short'],
                        "price": player['price'],
                        "avg_points": round(player['avg_points'], 1),
                        "recent_form": round(recent_form, 1),
                        "value_score": round(value_score, 2),
                        "games_played": int(player['games_played']),
                        "recommendation_strength": "STRONG" if value_score >= 1.5 else "MODERATE"
                    }
                    value_picks.append(pick)
            
            # Sort by value score
            value_picks.sort(key=lambda x: x['value_score'], reverse=True)
            
            return value_picks[:15]  # Top 15 value picks
            
        except Exception as e:
            logger.error(f"Error finding value picks: {str(e)}")
            return []
    
    def compare_players(self, player_ids: List[int]) -> Dict:
        """
        Compare multiple players across key metrics
        """
        try:
            if len(player_ids) < 2:
                return {"error": "Need at least 2 players to compare"}
            
            comparison = {
                "players": [],
                "winner_by_metric": {},
                "overall_ranking": []
            }
            
            player_scores = {}
            
            for player_id in player_ids:
                analysis = self.analyze_player_performance(player_id, detailed=False)
                
                if "error" not in analysis:
                    comparison["players"].append(analysis)
                    
                    # Calculate overall score for ranking
                    score = (
                        analysis.get('points_per_game', 0) * 0.3 +
                        analysis.get('consistency_score', 0) * 0.2 +
                        analysis.get('value_rating', 0) * 0.3 +
                        (3 - analysis.get('injury_risk', 3)) * 0.2  # Lower injury risk is better
                    )
                    player_scores[player_id] = score
            
            # Determine winners by metric
            if comparison["players"]:
                metrics = ["points_per_game", "consistency_score", "value_rating"]
                for metric in metrics:
                    best_player = max(comparison["players"], key=lambda x: x.get(metric, 0))
                    comparison["winner_by_metric"][metric] = best_player["name"]
                
                # Overall ranking
                ranked_players = sorted(player_scores.items(), key=lambda x: x[1], reverse=True)
                for rank, (player_id, score) in enumerate(ranked_players, 1):
                    player_info = next(p for p in comparison["players"] if p["player_id"] == player_id)
                    comparison["overall_ranking"].append({
                        "rank": rank,
                        "name": player_info["name"],
                        "overall_score": round(score, 2)
                    })
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing players: {str(e)}")
            return {"error": f"Comparison failed: {str(e)}"}
    
    # Helper methods for calculations
    
    def _calculate_points_per_game(self, player_data: pd.DataFrame) -> float:
        """Calculate points per game for players with sufficient minutes"""
        games_with_minutes = player_data[player_data['minutes'] >= self.min_minutes_threshold]
        return games_with_minutes['points'].mean() if not games_with_minutes.empty else 0
    
    def _calculate_consistency(self, player_data: pd.DataFrame) -> float:
        """Calculate consistency score (1 - coefficient of variation)"""
        if len(player_data) < 2:
            return 0
        
        points = player_data['points']
        mean_points = points.mean()
        
        if mean_points == 0:
            return 0
        
        cv = points.std() / mean_points
        return max(0, 1 - cv)  # Higher score = more consistent
    
    def _calculate_form_trend(self, player_data: pd.DataFrame) -> str:
        """Calculate recent form trend"""
        if len(player_data) < 3:
            return "INSUFFICIENT_DATA"
        
        recent_games = player_data.head(self.form_gameweeks)
        
        # Simple trend analysis
        first_half = recent_games.iloc[len(recent_games)//2:]['points'].mean()
        second_half = recent_games.iloc[:len(recent_games)//2]['points'].mean()
        
        if second_half > first_half * 1.2:
            return "IMPROVING"
        elif second_half < first_half * 0.8:
            return "DECLINING"
        else:
            return "STABLE"
    
    def _calculate_value_rating(self, player_data: pd.DataFrame) -> float:
        """Calculate value rating (points per price unit)"""
        latest_row = player_data.iloc[0]
        price = latest_row.get('price', 1)
        avg_points = self._calculate_points_per_game(player_data)
        
        if price == 0:
            return 0
        
        return avg_points / price
    
    def _assess_injury_risk(self, player_data: pd.DataFrame) -> int:
        """Assess injury risk based on minutes played pattern (1=low, 3=high)"""
        if len(player_data) < 3:
            return 2  # Medium risk due to insufficient data
        
        recent_games = player_data.head(5)
        games_with_low_minutes = len(recent_games[recent_games['minutes'] < 60])
        
        if games_with_low_minutes >= 3:
            return 3  # High risk
        elif games_with_low_minutes >= 1:
            return 2  # Medium risk
        else:
            return 1  # Low risk
    
    def _generate_player_recommendation(self, ppg: float, consistency: float, 
                                      form: str, value: float, injury_risk: int) -> str:
        """Generate overall player recommendation"""
        score = 0
        
        # Points per game contribution
        if ppg >= 6:
            score += 3
        elif ppg >= 4:
            score += 2
        elif ppg >= 2:
            score += 1
        
        # Consistency contribution
        if consistency >= 0.7:
            score += 2
        elif consistency >= 0.5:
            score += 1
        
        # Form contribution
        if form == "IMPROVING":
            score += 2
        elif form == "STABLE":
            score += 1
        
        # Value contribution
        if value >= 1.0:
            score += 2
        elif value >= 0.7:
            score += 1
        
        # Injury risk penalty
        if injury_risk == 3:
            score -= 2
        elif injury_risk == 2:
            score -= 1
        
        # Final recommendation
        if score >= 8:
            return "STRONG_BUY"
        elif score >= 6:
            return "BUY"
        elif score >= 4:
            return "HOLD"
        elif score >= 2:
            return "CONSIDER_SELL"
        else:
            return "SELL"
    
    def _calculate_detailed_metrics(self, player_data: pd.DataFrame) -> Dict:
        """Calculate detailed metrics for comprehensive analysis"""
        return {
            "games_played": len(player_data),
            "minutes_per_game": round(player_data['minutes'].mean(), 0),
            "goals_per_game": round(player_data['goals'].mean(), 2),
            "assists_per_game": round(player_data['assists'].mean(), 2),
            "bonus_points": player_data['bonus'].sum(),
            "clean_sheets": player_data['clean_sheets'].sum()
        }
    
    def _rate_performance(self, points: int, position: str) -> str:
        """Rate individual gameweek performance"""
        if position == "GK":
            if points >= 10: return "EXCELLENT"
            elif points >= 6: return "GOOD"
            elif points >= 3: return "AVERAGE"
            else: return "POOR"
        elif position == "DEF":
            if points >= 12: return "EXCELLENT"
            elif points >= 7: return "GOOD"
            elif points >= 4: return "AVERAGE"
            else: return "POOR"
        else:  # MID, FWD
            if points >= 15: return "EXCELLENT"
            elif points >= 9: return "GOOD"
            elif points >= 5: return "AVERAGE"
            else: return "POOR"
    
    def _calculate_form_multiplier(self, player_data: pd.DataFrame) -> float:
        """Calculate form multiplier for predictions"""
        if len(player_data) < 3:
            return 1.0
        
        recent_3 = player_data.head(3)['points'].mean()
        season_avg = player_data['points'].mean()
        
        if season_avg == 0:
            return 1.0
        
        multiplier = recent_3 / season_avg
        return max(0.5, min(2.0, multiplier))  # Cap between 0.5 and 2.0
    
    def _calculate_difficulty_adjustment(self, opponents: List[str]) -> float:
        """Calculate difficulty adjustment based on upcoming opponents"""
        # This would use actual team strength data in a full implementation
        # For now, return neutral adjustment
        return 1.0
    
    def _calculate_position_trend(self, pos_data: pd.DataFrame) -> str:
        """Calculate scoring trend for a position"""
        if len(pos_data) < 3:
            return "STABLE"
        
        # Group by gameweek and calculate averages
        gw_averages = pos_data.groupby('gameweek')['points'].mean().sort_index()
        
        if len(gw_averages) >= 3:
            recent = gw_averages.iloc[-2:].mean()
            earlier = gw_averages.iloc[:-2].mean()
            
            if recent > earlier * 1.1:
                return "IMPROVING"
            elif recent < earlier * 0.9:
                return "DECLINING"
        
        return "STABLE"
    
    def _generate_position_insights(self, position_analysis: Dict) -> List[str]:
        """Generate insights from position analysis"""
        insights = []
        
        # Find highest scoring position
        best_position = max(position_analysis.keys(), 
                          key=lambda x: position_analysis[x]['avg_points'])
        insights.append(f"{best_position} players averaging highest points ({position_analysis[best_position]['avg_points']})")
        
        # Find most consistent position
        most_consistent = max(position_analysis.keys(),
                            key=lambda x: position_analysis[x]['consistency'])
        insights.append(f"{most_consistent} players showing highest consistency ({position_analysis[most_consistent]['consistency']})")
        
        # Find trending positions
        improving_positions = [pos for pos, data in position_analysis.items() 
                             if data['trend'] == 'IMPROVING']
        if improving_positions:
            insights.append(f"Improving positions: {', '.join(improving_positions)}")
        
        return insights


# Example usage and testing
if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.utils.data_manager import DataManager
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        print("Testing Football Analyzer...")
        
        # Initialize components
        manager = DataManager("2025-26")
        analyzer = FootballAnalyzer(manager)
        
        # Test position trends (will work with any data)
        print("\n📊 Testing position trends analysis...")
        trends = analyzer.analyze_position_trends(gameweeks=5)
        print(f"Position trends result: {len(trends)} items")
        
        # Test value picks
        print("\n💰 Testing value picks analysis...")
        picks = analyzer.find_value_picks(max_price=8.0)
        print(f"Found {len(picks)} value picks")
        
        print("\n✅ Football Analyzer test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing Football Analyzer: {str(e)}")
        sys.exit(1)