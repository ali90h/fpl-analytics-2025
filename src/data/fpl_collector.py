"""
Official Fantasy Premier League API Data Collector
Focused on 2025/26 season data collection from the trusted FPL source
"""

import requests
import time
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class FPLDataCollector:
    """
    Collects data from the official Fantasy Premier League API
    This is our primary and most trusted data source
    """
    
    def __init__(self, season: str = "2025-26"):
        self.season = season
        self.base_url = "https://fantasy.premierleague.com/api"
        self.session = requests.Session()
        
        # Request headers to mimic browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-GB,en;q=0.9',
            'Referer': 'https://fantasy.premierleague.com/'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum 1 second between requests
        
        # Data storage
        self.data_dir = Path(f"data/fpl_data/{season}")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"FPL Data Collector initialized for season {season}")
    
    def _rate_limit(self):
        """Implement respectful rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with error handling and rate limiting"""
        self._rate_limit()
        
        url = f"{self.base_url}/{endpoint}/"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            logger.debug(f"Successfully fetched: {endpoint}")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {endpoint}: {str(e)}")
            raise
    
    def get_bootstrap_static(self) -> Dict:
        """
        Get the main FPL bootstrap data containing:
        - All players and their current stats
        - All teams
        - Current gameweek info
        - Player types and positions
        """
        logger.info("Fetching FPL bootstrap static data...")
        data = self._make_request("bootstrap-static")
        
        # Save raw data for backup
        self._save_raw_data("bootstrap_static", data)
        
        return data
    
    def get_current_gameweek(self) -> int:
        """Get the current active gameweek"""
        bootstrap = self.get_bootstrap_static()
        
        for event in bootstrap['events']:
            if event['is_current']:
                return event['id']
        
        # If no current gameweek found, return the next upcoming one
        for event in bootstrap['events']:
            if event['is_next']:
                return event['id']
        
        return 1  # Fallback
    
    def get_player_detailed_data(self, player_id: int) -> Dict:
        """
        Get detailed data for a specific player including:
        - Gameweek by gameweek performance
        - Fixture history
        - Upcoming fixtures
        """
        logger.debug(f"Fetching detailed data for player {player_id}")
        data = self._make_request(f"element-summary/{player_id}")
        
        return data
    
    def get_gameweek_live_data(self, gameweek: int) -> Dict:
        """
        Get live data for a specific gameweek including:
        - Player points
        - Bonus points
        - Match status
        """
        logger.info(f"Fetching live data for gameweek {gameweek}")
        data = self._make_request(f"event/{gameweek}/live")
        
        return data
    
    def collect_all_players_data(self) -> pd.DataFrame:
        """
        Collect comprehensive data for all players
        """
        logger.info("Collecting comprehensive player data...")
        
        # Get bootstrap data
        bootstrap = self.get_bootstrap_static()
        
        # Extract players data
        players_df = pd.DataFrame(bootstrap['elements'])
        teams_df = pd.DataFrame(bootstrap['teams'])
        positions_df = pd.DataFrame(bootstrap['element_types'])
        
        # Merge team names
        players_df = players_df.merge(
            teams_df[['id', 'name', 'short_name']], 
            left_on='team', 
            right_on='id', 
            suffixes=('', '_team')
        )
        
        # Merge position names
        players_df = players_df.merge(
            positions_df[['id', 'singular_name', 'singular_name_short']], 
            left_on='element_type', 
            right_on='id', 
            suffixes=('', '_position')
        )
        
        # Clean and organize data
        players_df = self._clean_players_data(players_df)
        
        logger.info(f"Collected data for {len(players_df)} players")
        return players_df
    
    def collect_gameweek_data(self, gameweek: int) -> Dict:
        """
        Collect all data for a specific gameweek
        """
        logger.info(f"Collecting comprehensive data for gameweek {gameweek}")
        
        data = {
            'gameweek': gameweek,
            'collected_at': datetime.now().isoformat(),
            'players': None,
            'live_data': None,
            'fixtures': None
        }
        
        try:
            # Get all players current data
            players_df = self.collect_all_players_data()
            data['players'] = players_df.to_dict('records')
            
            # Get live gameweek data
            live_data = self.get_gameweek_live_data(gameweek)
            data['live_data'] = live_data
            
            # Get fixtures for this gameweek
            fixtures = self._get_gameweek_fixtures(gameweek)
            data['fixtures'] = fixtures
            
            # Save to file
            self._save_gameweek_data(gameweek, data)
            
            logger.info(f"Successfully collected gameweek {gameweek} data")
            return data
            
        except Exception as e:
            logger.error(f"Failed to collect gameweek {gameweek} data: {str(e)}")
            raise
    
    def update_gameweek_results(self, gameweek: int) -> Dict:
        """
        Update gameweek data after matches are completed
        """
        logger.info(f"Updating results for gameweek {gameweek}")
        
        # Get fresh live data
        live_data = self.get_gameweek_live_data(gameweek)
        
        # Update any bonus points that may have changed
        updated_data = {
            'gameweek': gameweek,
            'updated_at': datetime.now().isoformat(),
            'live_data': live_data,
            'final_results': True
        }
        
        # Save updated data
        self._save_gameweek_update(gameweek, updated_data)
        
        return updated_data
    
    def get_player_history(self, player_id: int, gameweeks: int = 10) -> pd.DataFrame:
        """
        Get detailed history for a specific player
        """
        player_data = self.get_player_detailed_data(player_id)
        
        # Get recent gameweek history
        history_df = pd.DataFrame(player_data.get('history', []))
        
        if not history_df.empty:
            # Sort by round and get last N gameweeks
            history_df = history_df.sort_values('round').tail(gameweeks)
            
            # Add useful calculated fields
            history_df['points_per_minute'] = history_df['total_points'] / history_df['minutes'].replace(0, 1)
            history_df['value_score'] = history_df['total_points'] / history_df['value'] * 10
        
        return history_df
    
    def get_top_performers(self, position: str = None, limit: int = 20) -> pd.DataFrame:
        """
        Get top performing players, optionally filtered by position
        """
        players_df = self.collect_all_players_data()
        
        # Filter by position if specified
        if position:
            players_df = players_df[players_df['position_short'] == position]
        
        # Sort by total points and get top performers
        top_players = players_df.nlargest(limit, 'total_points')
        
        return top_players[['web_name', 'team_name', 'position_short', 'total_points', 
                          'points_per_game', 'now_cost', 'form', 'selected_by_percent']]
    
    def _clean_players_data(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and organize players dataframe
        """
        # Rename columns for clarity
        column_mapping = {
            'web_name': 'name',
            'name_team': 'team_name',
            'short_name': 'team_short',
            'singular_name_short': 'position_short',
            'singular_name': 'position_full',
            'now_cost': 'price',
            'total_points': 'total_points',
            'points_per_game': 'points_per_game',
            'form': 'form',
            'selected_by_percent': 'ownership_percent'
        }
        
        # Apply renaming
        for old_col, new_col in column_mapping.items():
            if old_col in players_df.columns:
                players_df[new_col] = players_df[old_col]
        
        # Convert price from tenths to actual price
        players_df['price'] = players_df['price'] / 10.0
        
        # Calculate additional metrics
        players_df['value_score'] = (players_df['total_points'] / players_df['price']).round(2)
        players_df['form_score'] = pd.to_numeric(players_df['form'], errors='coerce').fillna(0)
        players_df['ownership_percent'] = pd.to_numeric(players_df['ownership_percent'], errors='coerce').fillna(0)
        
        # Select key columns
        key_columns = [
            'id', 'name', 'team_name', 'team_short', 'position_short', 'position_full',
            'price', 'total_points', 'points_per_game', 'form_score', 'ownership_percent',
            'value_score', 'minutes', 'goals_scored', 'assists', 'clean_sheets',
            'goals_conceded', 'own_goals', 'penalties_saved', 'penalties_missed',
            'yellow_cards', 'red_cards', 'saves', 'bonus', 'bps'
        ]
        
        # Return only existing columns from the key columns list
        existing_columns = [col for col in key_columns if col in players_df.columns]
        return players_df[existing_columns].copy()
    
    def _get_gameweek_fixtures(self, gameweek: int) -> List[Dict]:
        """
        Get fixtures for a specific gameweek
        """
        bootstrap = self.get_bootstrap_static()
        fixtures = []
        
        for event in bootstrap['events']:
            if event['id'] == gameweek:
                # This is a simplified version - in reality we'd need to fetch fixtures endpoint
                fixtures.append({
                    'gameweek': gameweek,
                    'deadline_time': event['deadline_time'],
                    'finished': event['finished'],
                    'average_entry_score': event.get('average_entry_score', 0)
                })
                break
        
        return fixtures
    
    def _save_raw_data(self, data_type: str, data: Dict):
        """Save raw API response data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.data_dir / f"raw_{data_type}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Saved raw data to {filename}")
    
    def _save_gameweek_data(self, gameweek: int, data: Dict):
        """Save gameweek data"""
        filename = self.data_dir / f"gameweek_{gameweek:02d}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved gameweek {gameweek} data to {filename}")
    
    def _save_gameweek_update(self, gameweek: int, data: Dict):
        """Save gameweek update data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.data_dir / f"gameweek_{gameweek:02d}_update_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved gameweek {gameweek} update to {filename}")

# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Setup logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize collector
    collector = FPLDataCollector("2025-26")
    
    try:
        # Test basic functionality
        print("Testing FPL Data Collector...")
        
        # Get current gameweek
        current_gw = collector.get_current_gameweek()
        print(f"Current gameweek: {current_gw}")
        
        # Get all players data
        players_df = collector.collect_all_players_data()
        print(f"Total players: {len(players_df)}")
        
        # Show top 5 players by points
        top_players = players_df.nlargest(5, 'total_points')
        print("\nTop 5 players by points:")
        for _, player in top_players.iterrows():
            print(f"  {player['name']} ({player['team_short']}) - {player['total_points']} pts")
        
        print("\n✅ FPL Data Collector test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing FPL Data Collector: {str(e)}")
        sys.exit(1)