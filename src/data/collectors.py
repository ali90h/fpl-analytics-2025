# Football Analytics - Data Collection Module

import requests
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from config.config import API_CONFIGS

logger = logging.getLogger(__name__)

class FootballDataAPI:
    """
    Client for Football-Data.org API
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = API_CONFIGS["football_data"]["base_url"]
        self.headers = {
            "X-Auth-Token": api_key,
            "Content-Type": "application/json"
        }
        self.rate_limit = API_CONFIGS["football_data"]["rate_limit"]
        self.last_request = 0
    
    def _rate_limit_wait(self):
        """Implement rate limiting"""
        elapsed = time.time() - self.last_request
        wait_time = 60 / self.rate_limit
        
        if elapsed < wait_time:
            time.sleep(wait_time - elapsed)
        
        self.last_request = time.time()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with error handling"""
        self._rate_limit_wait()
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            logger.info(f"Successfully fetched data from {endpoint}")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {endpoint}: {str(e)}")
            raise
    
    def get_competitions(self) -> List[Dict]:
        """Get available competitions"""
        data = self._make_request("competitions")
        return data.get("competitions", [])
    
    def get_teams(self, competition_id: str, season: str = None) -> List[Dict]:
        """Get teams in a competition"""
        endpoint = f"competitions/{competition_id}/teams"
        params = {"season": season} if season else {}
        
        data = self._make_request(endpoint, params)
        return data.get("teams", [])
    
    def get_matches(self, competition_id: str = None, team_id: str = None, 
                   date_from: str = None, date_to: str = None) -> List[Dict]:
        """Get matches with optional filters"""
        if competition_id:
            endpoint = f"competitions/{competition_id}/matches"
        elif team_id:
            endpoint = f"teams/{team_id}/matches"
        else:
            endpoint = "matches"
        
        params = {}
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to
        
        data = self._make_request(endpoint, params)
        return data.get("matches", [])
    
    def get_match_details(self, match_id: str) -> Dict:
        """Get detailed match information"""
        endpoint = f"matches/{match_id}"
        return self._make_request(endpoint)

class SportmonksAPI:
    """
    Client for Sportmonks API
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = API_CONFIGS["sportmonks"]["base_url"]
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.rate_limit = API_CONFIGS["sportmonks"]["rate_limit"]
        self.last_request = 0
    
    def _rate_limit_wait(self):
        """Implement rate limiting"""
        elapsed = time.time() - self.last_request
        wait_time = 60 / self.rate_limit
        
        if elapsed < wait_time:
            time.sleep(wait_time - elapsed)
        
        self.last_request = time.time()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with error handling"""
        self._rate_limit_wait()
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            logger.info(f"Successfully fetched data from Sportmonks: {endpoint}")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Sportmonks API request failed for {endpoint}: {str(e)}")
            raise
    
    def get_player_statistics(self, season_id: str, player_id: str = None) -> List[Dict]:
        """Get player statistics for a season"""
        if player_id:
            endpoint = f"football/players/{player_id}/statistics/season/{season_id}"
        else:
            endpoint = f"football/statistics/players/season/{season_id}"
        
        data = self._make_request(endpoint)
        return data.get("data", [])
    
    def get_team_statistics(self, season_id: str, team_id: str = None) -> List[Dict]:
        """Get team statistics"""
        if team_id:
            endpoint = f"football/teams/{team_id}/statistics/season/{season_id}"
        else:
            endpoint = f"football/statistics/teams/season/{season_id}"
        
        data = self._make_request(endpoint)
        return data.get("data", [])
    
    def get_match_statistics(self, match_id: str) -> Dict:
        """Get detailed match statistics"""
        endpoint = f"football/fixtures/{match_id}/statistics"
        return self._make_request(endpoint)

class DataCollector:
    """
    Main data collection orchestrator
    """
    
    def __init__(self, football_data_key: str, sportmonks_key: str):
        self.fd_api = FootballDataAPI(football_data_key)
        self.sm_api = SportmonksAPI(sportmonks_key)
        self.logger = logging.getLogger(__name__)
    
    def collect_league_data(self, league_code: str, season: str) -> Dict:
        """
        Collect comprehensive data for a league and season
        """
        self.logger.info(f"Starting data collection for {league_code} {season}")
        
        collected_data = {
            "teams": [],
            "matches": [],
            "player_stats": [],
            "team_stats": []
        }
        
        try:
            # Get teams
            teams = self.fd_api.get_teams(league_code, season)
            collected_data["teams"] = teams
            
            # Get matches
            matches = self.fd_api.get_matches(league_code)
            collected_data["matches"] = matches
            
            # Get player statistics from Sportmonks
            player_stats = self.sm_api.get_player_statistics(season)
            collected_data["player_stats"] = player_stats
            
            # Get team statistics
            team_stats = self.sm_api.get_team_statistics(season)
            collected_data["team_stats"] = team_stats
            
            self.logger.info(f"Data collection completed for {league_code} {season}")
            
        except Exception as e:
            self.logger.error(f"Data collection failed: {str(e)}")
            raise
        
        return collected_data
    
    def collect_recent_matches(self, days_back: int = 7) -> List[Dict]:
        """
        Collect recent match data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        date_from = start_date.strftime("%Y-%m-%d")
        date_to = end_date.strftime("%Y-%m-%d")
        
        return self.fd_api.get_matches(date_from=date_from, date_to=date_to)
    
    def save_to_csv(self, data: Dict, output_dir: str = "data/raw"):
        """
        Save collected data to CSV files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for data_type, records in data.items():
            if records:
                df = pd.DataFrame(records)
                filename = f"{output_dir}/{data_type}_{timestamp}.csv"
                df.to_csv(filename, index=False)
                self.logger.info(f"Saved {len(records)} {data_type} records to {filename}")

# Example usage and data collection script
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize collector
    fd_key = os.getenv("FOOTBALL_DATA_API_KEY")
    sm_key = os.getenv("SPORTMONKS_API_KEY")
    
    if not fd_key or not sm_key:
        print("Error: API keys not found in environment variables")
        exit(1)
    
    collector = DataCollector(fd_key, sm_key)
    
    # Collect Premier League data
    try:
        data = collector.collect_league_data("PL", "2024")
        collector.save_to_csv(data)
        print("Data collection completed successfully!")
        
    except Exception as e:
        print(f"Data collection failed: {str(e)}")