"""
Match Context Data Collector
Collects additional match context from Football-Data.org and other sources
"""

import requests
import time
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class MatchDataCollector:
    """
    Collects match context data to supplement FPL data
    Sources: Football-Data.org, team news, weather, etc.
    """
    
    def __init__(self, season: str = "2025-26"):
        self.season = season
        self.football_data_api_key = os.getenv("FOOTBALL_DATA_API_KEY")
        
        # Football-Data.org API setup
        self.fd_base_url = "https://api.football-data.org/v4"
        self.fd_headers = {
            "X-Auth-Token": self.football_data_api_key,
            "Content-Type": "application/json"
        } if self.football_data_api_key else {}
        
        # Session for requests
        self.session = requests.Session()
        
        # Rate limiting
        self.last_fd_request = 0
        self.fd_rate_limit = 6.0  # 10 requests per minute = 6 seconds between requests
        
        # Data storage
        self.data_dir = Path(f"data/match_data/{season}")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Premier League competition ID in Football-Data.org
        self.pl_competition_id = "PL"
        
        logger.info(f"Match Data Collector initialized for season {season}")
    
    def _rate_limit_fd(self):
        """Rate limiting for Football-Data.org API"""
        elapsed = time.time() - self.last_fd_request
        if elapsed < self.fd_rate_limit:
            time.sleep(self.fd_rate_limit - elapsed)
        self.last_fd_request = time.time()
    
    def _make_fd_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make Football-Data.org API request"""
        if not self.football_data_api_key:
            logger.warning("Football-Data.org API key not configured, using mock data")
            return self._get_mock_match_data()
        
        self._rate_limit_fd()
        
        url = f"{self.fd_base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, headers=self.fd_headers, params=params, timeout=30)
            response.raise_for_status()
            
            logger.debug(f"Successfully fetched from Football-Data.org: {endpoint}")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Football-Data.org API request failed for {endpoint}: {str(e)}")
            # Return mock data on failure for development
            return self._get_mock_match_data()
    
    def get_pl_teams(self) -> List[Dict]:
        """Get Premier League teams"""
        logger.info("Fetching Premier League teams...")
        
        data = self._make_fd_request(f"competitions/{self.pl_competition_id}/teams")
        teams = data.get('teams', [])
        
        # Save teams data
        self._save_data("teams", teams)
        
        return teams
    
    def get_pl_matches(self, date_from: str = None, date_to: str = None) -> List[Dict]:
        """
        Get Premier League matches for a date range
        """
        logger.info("Fetching Premier League matches...")
        
        params = {}
        if date_from:
            params['dateFrom'] = date_from
        if date_to:
            params['dateTo'] = date_to
        
        data = self._make_fd_request(f"competitions/{self.pl_competition_id}/matches", params)
        matches = data.get('matches', [])
        
        return matches
    
    def get_gameweek_matches(self, gameweek: int) -> Dict:
        """
        Get matches for a specific gameweek
        Since Football-Data.org doesn't use gameweeks, we estimate based on dates
        """
        logger.info(f"Fetching matches for gameweek {gameweek}")
        
        # Calculate approximate dates for the gameweek
        # Premier League typically starts in mid-August
        season_start = datetime(2025, 8, 16)  # Approximate 2025/26 season start
        gameweek_start = season_start + timedelta(days=(gameweek - 1) * 7)
        gameweek_end = gameweek_start + timedelta(days=6)
        
        date_from = gameweek_start.strftime("%Y-%m-%d")
        date_to = gameweek_end.strftime("%Y-%m-%d")
        
        matches = self.get_pl_matches(date_from, date_to)
        
        # Enhance matches with additional context
        enhanced_matches = []
        for match in matches:
            enhanced_match = self._enhance_match_data(match, gameweek)
            enhanced_matches.append(enhanced_match)
        
        # Calculate team difficulty ratings for this gameweek
        team_difficulty = self._calculate_team_difficulties(enhanced_matches)
        
        # Return structured data
        return {
            'gameweek': gameweek,
            'matches': enhanced_matches,
            'team_difficulty': team_difficulty,
            'date_range': {
                'from': date_from,
                'to': date_to
            },
            'collected_at': datetime.now().isoformat()
        }

    def collect_gameweek_matches(self, gameweek: int) -> Dict:
        """
        Collect comprehensive match data for a gameweek
        """
        logger.info(f"Collecting comprehensive match data for gameweek {gameweek}")
        
        data = {
            'gameweek': gameweek,
            'collected_at': datetime.now().isoformat(),
            'matches': [],
            'team_difficulties': {},
            'injury_news': [],
            'weather_forecast': []
        }
        
        try:
            # Get matches
            matches = self.get_gameweek_matches(gameweek)
            data['matches'] = matches
            
            # Calculate team difficulties
            data['team_difficulties'] = self._calculate_team_difficulties(matches)
            
            # Get injury news (mock implementation)
            data['injury_news'] = self._get_injury_news()
            
            # Get weather forecast (mock implementation)
            data['weather_forecast'] = self._get_weather_forecast(matches)
            
            # Save data
            self._save_gameweek_matches(gameweek, data)
            
            logger.info(f"Successfully collected match data for gameweek {gameweek}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to collect match data for gameweek {gameweek}: {str(e)}")
            raise
    
    def _enhance_match_data(self, match: Dict, gameweek: int) -> Dict:
        """
        Enhance match data with additional context
        """
        enhanced = match.copy()
        
        # Add gameweek
        enhanced['gameweek'] = gameweek
        
        # Add difficulty ratings (simplified)
        home_team = match.get('homeTeam', {}).get('name', '')
        away_team = match.get('awayTeam', {}).get('name', '')
        
        enhanced['home_difficulty'] = self._get_team_difficulty(away_team, 'away')
        enhanced['away_difficulty'] = self._get_team_difficulty(home_team, 'home')
        
        # Add venue advantage
        enhanced['home_advantage'] = 1.2  # Standard home advantage multiplier
        
        # Add kickoff time importance
        kickoff = match.get('utcDate', '')
        if kickoff:
            kickoff_hour = datetime.fromisoformat(kickoff.replace('Z', '+00:00')).hour
            enhanced['prime_time'] = 12 <= kickoff_hour <= 18  # Weekend afternoon games
        
        return enhanced
    
    def _get_team_difficulty(self, opponent: str, venue: str) -> int:
        """
        Calculate difficulty rating (1-5) based on opponent strength
        This is a simplified version - in reality would use historical data
        """
        # Mock difficulty ratings - in reality would be data-driven
        difficulty_ratings = {
            'Manchester City': 5, 'Arsenal': 5, 'Liverpool': 5, 'Chelsea': 4,
            'Manchester United': 4, 'Tottenham': 4, 'Newcastle': 4, 'Brighton': 3,
            'West Ham': 3, 'Aston Villa': 3, 'Crystal Palace': 3, 'Fulham': 3,
            'Brentford': 2, 'Wolves': 2, 'Everton': 2, 'Nottingham Forest': 2,
            'Sheffield United': 1, 'Burnley': 1, 'Luton': 1, 'Bournemouth': 2
        }
        
        base_difficulty = difficulty_ratings.get(opponent, 3)
        
        # Adjust for venue
        if venue == 'away':
            base_difficulty = min(5, base_difficulty + 1)
        
        return base_difficulty
    
    def _calculate_team_difficulties(self, matches: List[Dict]) -> Dict:
        """
        Calculate difficulty ratings for all teams based on their fixtures
        """
        team_difficulties = {}
        
        for match in matches:
            home_team = match.get('homeTeam', {}).get('name', '')
            away_team = match.get('awayTeam', {}).get('name', '')
            
            if home_team:
                team_difficulties[home_team] = match.get('home_difficulty', 3)
            if away_team:
                team_difficulties[away_team] = match.get('away_difficulty', 3)
        
        return team_difficulties
    
    def _get_injury_news(self) -> List[Dict]:
        """
        Get injury news (mock implementation)
        In reality, this would scrape reliable sources or use APIs
        """
        # Mock injury data
        return [
            {
                'player': 'Mock Player 1',
                'team': 'Arsenal',
                'injury_type': 'Hamstring',
                'expected_return': '2-3 weeks',
                'severity': 'Minor'
            },
            {
                'player': 'Mock Player 2',
                'team': 'Liverpool',
                'injury_type': 'Ankle',
                'expected_return': '1 week',
                'severity': 'Minor'
            }
        ]
    
    def _get_weather_forecast(self, matches: List[Dict]) -> List[Dict]:
        """
        Get weather forecast for match venues (mock implementation)
        """
        # Mock weather data
        weather_data = []
        for match in matches:
            weather_data.append({
                'match_id': match.get('id', 'unknown'),
                'venue': match.get('venue', 'Unknown'),
                'temperature': 15,  # Celsius
                'weather_condition': 'Partly Cloudy',
                'wind_speed': 10,  # km/h
                'precipitation_chance': 20  # %
            })
        
        return weather_data
    
    def _get_mock_match_data(self) -> Dict:
        """
        Provide mock match data for development when API is not available
        """
        return {
            'matches': [
                {
                    'id': 'mock_1',
                    'homeTeam': {'name': 'Arsenal', 'id': 1},
                    'awayTeam': {'name': 'Liverpool', 'id': 2},
                    'utcDate': datetime.now().isoformat(),
                    'status': 'SCHEDULED',
                    'venue': 'Emirates Stadium'
                },
                {
                    'id': 'mock_2',
                    'homeTeam': {'name': 'Manchester City', 'id': 3},
                    'awayTeam': {'name': 'Chelsea', 'id': 4},
                    'utcDate': (datetime.now() + timedelta(days=1)).isoformat(),
                    'status': 'SCHEDULED',
                    'venue': 'Etihad Stadium'
                }
            ]
        }
    
    def _save_data(self, data_type: str, data: Any):
        """Save data to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.data_dir / f"{data_type}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Saved {data_type} data to {filename}")
    
    def _save_gameweek_matches(self, gameweek: int, data: Dict):
        """Save gameweek match data"""
        filename = self.data_dir / f"gameweek_{gameweek:02d}_matches.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved gameweek {gameweek} match data to {filename}")

# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Setup logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize collector
    collector = MatchDataCollector("2025-26")
    
    try:
        # Test basic functionality
        print("Testing Match Data Collector...")
        
        # Get teams
        teams = collector.get_pl_teams()
        print(f"Fetched {len(teams)} teams")
        
        # Get matches for current gameweek (mock gameweek 1)
        match_data = collector.collect_gameweek_matches(1)
        print(f"Collected data for {len(match_data['matches'])} matches")
        
        # Show sample match
        if match_data['matches']:
            sample_match = match_data['matches'][0]
            home_team = sample_match.get('homeTeam', {}).get('name', 'Unknown')
            away_team = sample_match.get('awayTeam', {}).get('name', 'Unknown')
            print(f"Sample match: {home_team} vs {away_team}")
        
        print("\n✅ Match Data Collector test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing Match Data Collector: {str(e)}")
        sys.exit(1)