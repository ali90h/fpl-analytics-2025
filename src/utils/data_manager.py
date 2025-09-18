"""
Data Manager - High-quality data organization and validation
Ensures data accuracy and provides update mechanisms
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import shutil
import sqlite3

logger = logging.getLogger(__name__)

class DataManager:
    """
    Manages data storage, validation, and organization for the football analytics tool
    Ensures high data quality and provides efficient access to data
    """
    
    def __init__(self, season: str = "2025-26"):
        self.season = season
        
        # Data directories
        self.base_dir = Path(f"data/organized/{season}")
        self.fpl_dir = self.base_dir / "fpl"
        self.match_dir = self.base_dir / "matches" 
        self.processed_dir = self.base_dir / "processed"
        self.backup_dir = self.base_dir / "backups"
        
        # Create directories
        for directory in [self.fpl_dir, self.match_dir, self.processed_dir, self.backup_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # SQLite database for structured data
        self.db_path = self.base_dir / f"football_data_{season.replace('-', '_')}.db"
        self._init_database()
        
        # Data validation rules
        self.validation_rules = self._setup_validation_rules()
        
        logger.info(f"Data Manager initialized for season {season}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Players table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS players (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    team_name TEXT,
                    position_short TEXT,
                    price REAL,
                    total_points INTEGER DEFAULT 0,
                    form_score REAL DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Gameweek performances table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS gameweek_performances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id INTEGER,
                    gameweek INTEGER,
                    points INTEGER DEFAULT 0,
                    minutes INTEGER DEFAULT 0,
                    goals INTEGER DEFAULT 0,
                    assists INTEGER DEFAULT 0,
                    clean_sheets INTEGER DEFAULT 0,
                    goals_conceded INTEGER DEFAULT 0,
                    saves INTEGER DEFAULT 0,
                    bonus INTEGER DEFAULT 0,
                    price REAL,
                    opponent TEXT,
                    venue TEXT,
                    difficulty INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (player_id) REFERENCES players (id)
                )
            ''')
            
            # Data quality table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gameweek INTEGER,
                    data_type TEXT,
                    quality_score REAL,
                    issues_found INTEGER DEFAULT 0,
                    issues_details TEXT,
                    validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def _setup_validation_rules(self) -> Dict:
        """Setup data validation rules"""
        return {
            'player_data': {
                'required_fields': ['id', 'name', 'team_name', 'position_short', 'price'],
                'numeric_fields': ['id', 'price', 'total_points', 'form_score'],
                'price_range': (4.0, 15.0),  # Typical FPL price range
                'points_range': (0, 500),    # Reasonable total points range
                'positions': ['GK', 'DEF', 'MID', 'FWD']
            },
            'performance_data': {
                'required_fields': ['player_id', 'gameweek', 'points'],
                'numeric_fields': ['player_id', 'gameweek', 'points', 'minutes'],
                'points_range': (-5, 25),    # Typical gameweek points range
                'minutes_range': (0, 120)    # Match minutes range
            }
        }
    
    def save_gameweek_data(self, gameweek: int, fpl_data: Dict, match_data: Dict) -> bool:
        """
        Save and validate gameweek data from both FPL and match sources
        """
        logger.info(f"Saving and validating gameweek {gameweek} data")
        
        try:
            # Validate data quality
            fpl_quality = self._validate_fpl_data(fpl_data, gameweek)
            match_quality = self._validate_match_data(match_data, gameweek)
            
            if fpl_quality['score'] < 0.8 or match_quality['score'] < 0.8:
                logger.warning(f"Data quality below threshold for gameweek {gameweek}")
                logger.warning(f"FPL quality: {fpl_quality['score']:.2f}, Match quality: {match_quality['score']:.2f}")
            
            # Save raw data
            self._save_raw_gameweek_data(gameweek, fpl_data, match_data)
            
            # Process and save to database
            self._save_to_database(gameweek, fpl_data, match_data)
            
            # Update processed data files
            self._update_processed_data(gameweek)
            
            # Log data quality
            self._log_data_quality(gameweek, fpl_quality, match_quality)
            
            logger.info(f"Successfully saved gameweek {gameweek} data")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save gameweek {gameweek} data: {str(e)}")
            return False
    
    def gameweek_data_exists(self, gameweek: int) -> bool:
        """Check if data for a gameweek already exists"""
        fpl_file = self.fpl_dir / f"gameweek_{gameweek:02d}.json"
        match_file = self.match_dir / f"gameweek_{gameweek:02d}_matches.json"
        
        return fpl_file.exists() and match_file.exists()
    
    def get_player_data(self, player_id: int = None, gameweeks: int = None) -> pd.DataFrame:
        """
        Get player data from database
        """
        with sqlite3.connect(self.db_path) as conn:
            if player_id:
                if gameweeks:
                    # Get specific player's recent gameweeks
                    query = '''
                        SELECT * FROM gameweek_performances 
                        WHERE player_id = ? 
                        ORDER BY gameweek DESC 
                        LIMIT ?
                    '''
                    df = pd.read_sql_query(query, conn, params=(player_id, gameweeks))
                else:
                    # Get all data for specific player
                    query = '''
                        SELECT p.name, p.team_name, p.position_short, p.price,
                               gp.* FROM gameweek_performances gp
                        JOIN players p ON gp.player_id = p.id
                        WHERE gp.player_id = ?
                        ORDER BY gp.gameweek DESC
                    '''
                    df = pd.read_sql_query(query, conn, params=(player_id,))
            else:
                # Get all current player data
                query = 'SELECT * FROM players ORDER BY total_points DESC'
                df = pd.read_sql_query(query, conn)
        
        return df
    
    def get_gameweek_summary(self, gameweek: int) -> Dict:
        """Get summary of a specific gameweek"""
        with sqlite3.connect(self.db_path) as conn:
            # Player performances for the gameweek
            query = '''
                SELECT p.name, p.team_name, p.position_short,
                       gp.points, gp.minutes, gp.opponent, gp.venue
                FROM gameweek_performances gp
                JOIN players p ON gp.player_id = p.id
                WHERE gp.gameweek = ?
                ORDER BY gp.points DESC
            '''
            performances_df = pd.read_sql_query(query, conn, params=(gameweek,))
            
            # Calculate summary statistics
            summary = {
                'gameweek': gameweek,
                'total_players': len(performances_df),
                'average_points': performances_df['points'].mean() if not performances_df.empty else 0,
                'highest_scorer': performances_df.iloc[0].to_dict() if not performances_df.empty else None,
                'top_performers': performances_df.head(10).to_dict('records'),
                'position_averages': performances_df.groupby('position_short')['points'].mean().to_dict()
            }
        
        return summary
    
    def recalculate_form_metrics(self, gameweek: int):
        """
        Recalculate form metrics after gameweek completion
        """
        logger.info(f"Recalculating form metrics for gameweek {gameweek}")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get all players
            players_df = pd.read_sql_query('SELECT id FROM players', conn)
            
            for player_id in players_df['id']:
                # Get last 5 gameweeks performance
                query = '''
                    SELECT points FROM gameweek_performances 
                    WHERE player_id = ? AND gameweek <= ?
                    ORDER BY gameweek DESC
                    LIMIT 5
                '''
                recent_performances = pd.read_sql_query(query, conn, params=(player_id, gameweek))
                
                if not recent_performances.empty:
                    # Calculate form score (weighted recent performance)
                    weights = np.array([0.4, 0.3, 0.2, 0.1, 0.05])[:len(recent_performances)]
                    form_score = np.average(recent_performances['points'], weights=weights)
                    
                    # Update player's form score
                    cursor.execute(
                        'UPDATE players SET form_score = ?, last_updated = ? WHERE id = ?',
                        (form_score, datetime.now().isoformat(), player_id)
                    )
            
            conn.commit()
            logger.info(f"Form metrics recalculated for {len(players_df)} players")
    
    def get_data_status(self) -> Dict:
        """Get current data status and quality metrics"""
        with sqlite3.connect(self.db_path) as conn:
            # Get basic counts
            player_count = pd.read_sql_query('SELECT COUNT(*) as count FROM players', conn).iloc[0]['count']
            
            # Get gameweeks with data
            gameweeks_query = '''
                SELECT DISTINCT gameweek FROM gameweek_performances 
                ORDER BY gameweek DESC
            '''
            gameweeks_df = pd.read_sql_query(gameweeks_query, conn)
            
            # Get latest data quality
            quality_query = '''
                SELECT gameweek, AVG(quality_score) as avg_quality 
                FROM data_quality 
                GROUP BY gameweek 
                ORDER BY gameweek DESC 
                LIMIT 5
            '''
            quality_df = pd.read_sql_query(quality_query, conn)
            
            # Recent gameweek details
            recent_gameweeks = []
            for _, row in gameweeks_df.head(3).iterrows():
                gw = int(row['gameweek'])
                gw_query = '''
                    SELECT COUNT(*) as updated_players, AVG(points) as avg_points
                    FROM gameweek_performances 
                    WHERE gameweek = ?
                '''
                gw_stats = pd.read_sql_query(gw_query, conn, params=(gw,)).iloc[0]
                
                recent_gameweeks.append({
                    'gameweek': gw,
                    'updated_players': int(gw_stats['updated_players']),
                    'avg_points': float(gw_stats['avg_points']) if gw_stats['avg_points'] else 0,
                    'matches': 10  # Assume 10 matches per gameweek
                })
        
        return {
            'current_gameweek': gameweeks_df['gameweek'].max() if not gameweeks_df.empty else 1,
            'gameweeks_collected': len(gameweeks_df),
            'total_players': player_count,
            'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data_quality': int(quality_df['avg_quality'].mean() * 100) if not quality_df.empty else 0,
            'recent_gameweeks': recent_gameweeks
        }
    
    def create_backup(self, backup_dir: str = "backups") -> str:
        """Create backup of all data"""
        backup_path = Path(backup_dir)
        backup_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{self.season}_{timestamp}"
        full_backup_path = backup_path / backup_name
        
        # Copy all data
        shutil.copytree(self.base_dir, full_backup_path)
        
        logger.info(f"Backup created at {full_backup_path}")
        return str(full_backup_path)
    
    def _validate_fpl_data(self, data: Dict, gameweek: int) -> Dict:
        """Validate FPL data quality"""
        issues = []
        players = data.get('players', [])
        
        if not players:
            issues.append("No players data found")
            return {'score': 0.0, 'issues': issues}
        
        rules = self.validation_rules['player_data']
        valid_players = 0
        
        for player in players:
            player_valid = True
            
            # Check required fields
            for field in rules['required_fields']:
                if field not in player or player[field] is None:
                    issues.append(f"Missing {field} for player {player.get('name', 'unknown')}")
                    player_valid = False
            
            # Check numeric fields
            for field in rules['numeric_fields']:
                if field in player:
                    try:
                        float(player[field])
                    except (ValueError, TypeError):
                        issues.append(f"Invalid numeric value for {field}: {player[field]}")
                        player_valid = False
            
            # Check price range
            if 'price' in player:
                price = float(player['price'])
                if not (rules['price_range'][0] <= price <= rules['price_range'][1]):
                    issues.append(f"Price out of range for {player.get('name', 'unknown')}: {price}")
                    player_valid = False
            
            # Check position
            if 'position_short' in player:
                if player['position_short'] not in rules['positions']:
                    issues.append(f"Invalid position for {player.get('name', 'unknown')}: {player['position_short']}")
                    player_valid = False
            
            if player_valid:
                valid_players += 1
        
        quality_score = valid_players / len(players) if players else 0
        
        return {
            'score': quality_score,
            'issues': issues,
            'total_players': len(players),
            'valid_players': valid_players
        }
    
    def _validate_match_data(self, data: Dict, gameweek: int) -> Dict:
        """Validate match data quality"""
        issues = []
        matches = data.get('matches', [])
        
        if not matches:
            issues.append("No matches data found")
            return {'score': 0.0, 'issues': issues}
        
        valid_matches = 0
        
        for match in matches:
            match_valid = True
            
            # Check required fields
            required_fields = ['homeTeam', 'awayTeam']
            for field in required_fields:
                if field not in match:
                    issues.append(f"Missing {field} in match data")
                    match_valid = False
            
            # Check team data structure
            for team_key in ['homeTeam', 'awayTeam']:
                if team_key in match and isinstance(match[team_key], dict):
                    if 'name' not in match[team_key]:
                        issues.append(f"Missing team name in {team_key}")
                        match_valid = False
            
            if match_valid:
                valid_matches += 1
        
        quality_score = valid_matches / len(matches) if matches else 0
        
        return {
            'score': quality_score,
            'issues': issues,
            'total_matches': len(matches),
            'valid_matches': valid_matches
        }
    
    def _save_raw_gameweek_data(self, gameweek: int, fpl_data: Dict, match_data: Dict):
        """Save raw data files"""
        # Save FPL data
        fpl_file = self.fpl_dir / f"gameweek_{gameweek:02d}.json"
        with open(fpl_file, 'w') as f:
            json.dump(fpl_data, f, indent=2)
        
        # Save match data
        match_file = self.match_dir / f"gameweek_{gameweek:02d}_matches.json"
        with open(match_file, 'w') as f:
            json.dump(match_data, f, indent=2)
        
        logger.debug(f"Saved raw data files for gameweek {gameweek}")
    
    def _save_to_database(self, gameweek: int, fpl_data: Dict, match_data: Dict):
        """Save data to SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Process FPL player data
            players = fpl_data.get('players', [])
            for player in players:
                # Insert or update player
                cursor.execute('''
                    INSERT OR REPLACE INTO players 
                    (id, name, team_name, position_short, price, total_points, form_score, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    player.get('id'),
                    player.get('name'),
                    player.get('team_name'),
                    player.get('position_short'),
                    player.get('price'),
                    player.get('total_points', 0),
                    player.get('form_score', 0),
                    datetime.now().isoformat()
                ))
                
                # If we have live data, add gameweek performance
                live_data = fpl_data.get('live_data', {})
                if live_data and 'elements' in live_data:
                    player_live = next((p for p in live_data['elements'] if p['id'] == player.get('id')), None)
                    if player_live:
                        stats = player_live.get('stats', {})
                        cursor.execute('''
                            INSERT OR REPLACE INTO gameweek_performances
                            (player_id, gameweek, points, minutes, goals, assists, 
                             clean_sheets, goals_conceded, saves, bonus, price)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            player.get('id'),
                            gameweek,
                            stats.get('total_points', 0),
                            stats.get('minutes', 0),
                            stats.get('goals_scored', 0),
                            stats.get('assists', 0),
                            stats.get('clean_sheets', 0),
                            stats.get('goals_conceded', 0),
                            stats.get('saves', 0),
                            stats.get('bonus', 0),
                            player.get('price')
                        ))
            
            conn.commit()
            logger.debug(f"Saved {len(players)} players to database for gameweek {gameweek}")
    
    def _update_processed_data(self, gameweek: int):
        """Update processed data files for easy access"""
        # Create summary files
        summary = self.get_gameweek_summary(gameweek)
        summary_file = self.processed_dir / f"gameweek_{gameweek:02d}_summary.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.debug(f"Updated processed data for gameweek {gameweek}")
    
    def _log_data_quality(self, gameweek: int, fpl_quality: Dict, match_quality: Dict):
        """Log data quality metrics to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Log FPL data quality
            cursor.execute('''
                INSERT INTO data_quality 
                (gameweek, data_type, quality_score, issues_found, issues_details)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                gameweek,
                'fpl',
                fpl_quality['score'],
                len(fpl_quality['issues']),
                '; '.join(fpl_quality['issues'][:5])  # First 5 issues
            ))
            
            # Log match data quality
            cursor.execute('''
                INSERT INTO data_quality 
                (gameweek, data_type, quality_score, issues_found, issues_details)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                gameweek,
                'match',
                match_quality['score'],
                len(match_quality['issues']),
                '; '.join(match_quality['issues'][:5])  # First 5 issues
            ))
            
            conn.commit()

# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Setup logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize data manager
    manager = DataManager("2025-26")
    
    try:
        print("Testing Data Manager...")
        
        # Test database initialization
        status = manager.get_data_status()
        print(f"Data status: {status}")
        
        # Test with mock data
        mock_fpl_data = {
            'players': [
                {
                    'id': 1,
                    'name': 'Test Player',
                    'team_name': 'Arsenal',
                    'position_short': 'MID',
                    'price': 8.5,
                    'total_points': 45,
                    'form_score': 7.2
                }
            ]
        }
        
        mock_match_data = {
            'matches': [
                {
                    'homeTeam': {'name': 'Arsenal'},
                    'awayTeam': {'name': 'Liverpool'}
                }
            ]
        }
        
        # Test validation
        fpl_quality = manager._validate_fpl_data(mock_fpl_data, 1)
        print(f"FPL data quality: {fpl_quality['score']:.2f}")
        
        print("\n✅ Data Manager test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing Data Manager: {str(e)}")
        sys.exit(1)