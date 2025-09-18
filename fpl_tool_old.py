#!/usr/bin/env python3
"""
Fantasy Premier League Data Collection and Analysis Tool
Terminal-based tool for 2025/26 season with focus on official FPL data
"""

import click
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import logging

# Add src directory to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    from src.data.fpl_collector import FPLDataCollector
    from src.data.match_collector import MatchDataCollector
    from src.utils.data_manager import DataManager
    from src.analysis.analyzer import FootballAnalyzer
    
    # Components available
    COMPONENTS_AVAILABLE = True
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Some modules may not be available yet. Creating placeholder functionality.")
    
    COMPONENTS_AVAILABLE = False
    
    class FPLDataCollector:
        def __init__(self):
            pass
        def collect_bootstrap_data(self):
            return {"status": "Mock data - collector not available"}
    
    class MatchDataCollector:
        def __init__(self):
            pass
        def collect_gameweek_matches(self, gameweek):
            return {"status": "Mock data - collector not available"}
    
    class DataManager:
        def __init__(self, season):
            self.season = season
        def get_data_status(self):
            return {"status": "Mock data - manager not available"}
    
    class FootballAnalyzer:
        def __init__(self, data_manager):
            pass
        def analyze_position_trends(self):
            return {"status": "Mock data - analyzer not available"}

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)

# Global context for CLI
@click.group()
@click.option('--season', default='2025-26', help='Season to analyze (default: 2025-26)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, season, verbose):
    """
    Football Analytics CLI Tool - 2025/26 Season
    
    A focused terminal tool for fantasy football data collection and analysis.
    """
    ctx.ensure_object(dict)
    ctx.obj['season'] = season
    ctx.obj['verbose'] = verbose
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    click.echo(f"🏈 Football Analytics CLI - Season {season}")
    click.echo("=" * 50)

@cli.command()
@click.option('--gameweek', '-gw', type=int, help='Specific gameweek to collect (default: current)')
@click.option('--force', '-f', is_flag=True, help='Force update even if data exists')
@click.pass_context
def collect(ctx, gameweek, force):
    """
    Collect latest fantasy and match data from trusted sources.
    
    Sources:
    - Official FPL API (primary source)
    - Football-Data.org API (match context)
    - Injury/news data from reliable sources
    """
    season = ctx.obj['season']
    
    click.echo("🔄 Starting data collection...")
    
    try:
        # Initialize collectors
        fpl_collector = FPLDataCollector(season=season)
        match_collector = MatchDataCollector(season=season)
        
        # Get current gameweek if not specified
        if not gameweek:
            gameweek = fpl_collector.get_current_gameweek()
            click.echo(f"📅 Current gameweek: {gameweek}")
        
        # Check if data exists and force flag
        data_manager = DataManager(season)
        if data_manager.gameweek_data_exists(gameweek) and not force:
            click.echo(f"⚠️  Data for gameweek {gameweek} already exists. Use --force to update.")
            return
        
        # Collect FPL data
        click.echo("🎯 Collecting FPL player data...")
        fpl_data = fpl_collector.collect_gameweek_data(gameweek)
        
        # Collect match data
        click.echo("⚽ Collecting match context data...")
        match_data = match_collector.collect_gameweek_matches(gameweek)
        
        # Save data
        click.echo("💾 Saving collected data...")
        data_manager.save_gameweek_data(gameweek, fpl_data, match_data)
        
        click.echo(f"✅ Data collection completed for gameweek {gameweek}")
        click.echo(f"   - Players: {len(fpl_data.get('players', []))}")
        click.echo(f"   - Matches: {len(match_data.get('matches', []))}")
        
    except Exception as e:
        logger.error(f"Data collection failed: {str(e)}")
        click.echo(f"❌ Error: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--gameweek', '-gw', type=int, help='Gameweek to update (default: latest)')
@click.pass_context
def update(ctx, gameweek):
    """
    Update data after gameweek completion.
    
    This command updates:
    - Player points and statistics
    - Match results and bonus points
    - Injury status updates
    - Form calculations
    """
    season = ctx.obj['season']
    
    click.echo("🔄 Updating post-gameweek data...")
    
    try:
        fpl_collector = FPLDataCollector(season=season)
        
        # Get latest gameweek if not specified
        if not gameweek:
            gameweek = fpl_collector.get_current_gameweek() - 1  # Previous completed GW
            click.echo(f"📅 Updating gameweek: {gameweek}")
        
        # Update player performances
        click.echo("📊 Updating player performances...")
        updated_data = fpl_collector.update_gameweek_results(gameweek)
        
        # Recalculate form and trends
        click.echo("📈 Recalculating form metrics...")
        data_manager = DataManager(season)
        data_manager.recalculate_form_metrics(gameweek)
        
        click.echo(f"✅ Update completed for gameweek {gameweek}")
        
    except Exception as e:
        logger.error(f"Update failed: {str(e)}")
        click.echo(f"❌ Error: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--player', '-p', help='Specific player name to analyze')
@click.option('--position', type=click.Choice(['GK', 'DEF', 'MID', 'FWD']), help='Filter by position')
@click.option('--team', help='Filter by team')
@click.option('--gameweeks', '-gw', default=5, help='Number of recent gameweeks to analyze')
@click.pass_context
def analyze(ctx, player, position, team, gameweeks):
    """
    Analyze player performance and predictions.
    
    Provides detailed analysis including:
    - Recent form trends
    - Fixture difficulty
    - Expected points predictions
    - Value for money analysis
    """
    season = ctx.obj['season']
    
    click.echo("📊 Starting performance analysis...")
    
    try:
        analyzer = FantasyAnalyzer(season)
        
        if player:
            # Single player analysis
            click.echo(f"🔍 Analyzing player: {player}")
            analysis = analyzer.analyze_player(player, gameweeks)
            _display_player_analysis(analysis)
            
        else:
            # Position/team analysis
            click.echo(f"📈 Analyzing {position or 'all'} players...")
            analysis = analyzer.analyze_position(
                position=position, 
                team=team, 
                gameweeks=gameweeks
            )
            _display_position_analysis(analysis)
            
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        click.echo(f"❌ Error: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--gameweek', '-gw', type=int, help='Gameweek to predict (default: next)')
@click.option('--top', '-n', default=10, help='Show top N players')
@click.option('--budget', type=float, help='Budget constraint (in millions)')
@click.pass_context
def predict(ctx, gameweek, top, budget):
    """
    Generate predictions for upcoming gameweek.
    
    Predictions include:
    - Expected points for each player
    - Captain recommendations
    - Transfer suggestions
    - Differential picks
    """
    season = ctx.obj['season']
    
    click.echo("🔮 Generating predictions...")
    
    try:
        analyzer = FantasyAnalyzer(season)
        
        # Get current gameweek if not specified
        if not gameweek:
            fpl_collector = FPLDataCollector(season=season)
            gameweek = fpl_collector.get_current_gameweek()
            click.echo(f"📅 Predicting for gameweek: {gameweek}")
        
        # Generate predictions
        predictions = analyzer.predict_gameweek(gameweek, budget_limit=budget)
        
        # Display top predictions
        _display_predictions(predictions, top)
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        click.echo(f"❌ Error: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.pass_context
def status(ctx):
    """
    Show current data status and statistics.
    """
    season = ctx.obj['season']
    
    try:
        data_manager = DataManager(season)
        status_info = data_manager.get_data_status()
        
        click.echo("📊 Data Status")
        click.echo("-" * 30)
        click.echo(f"Season: {season}")
        click.echo(f"Current Gameweek: {status_info['current_gameweek']}")
        click.echo(f"Gameweeks Collected: {status_info['gameweeks_collected']}")
        click.echo(f"Total Players: {status_info['total_players']}")
        click.echo(f"Last Update: {status_info['last_update']}")
        click.echo(f"Data Quality: {status_info['data_quality']}%")
        
        # Show recent gameweek summary
        if status_info.get('recent_gameweeks'):
            click.echo("\n📈 Recent Gameweeks:")
            for gw_info in status_info['recent_gameweeks'][-3:]:
                click.echo(f"  GW{gw_info['gameweek']}: {gw_info['matches']} matches, "
                          f"{gw_info['updated_players']} player updates")
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        click.echo(f"❌ Error: {str(e)}", err=True)

@cli.command()
@click.option('--backup-dir', default='backups', help='Backup directory')
@click.pass_context
def backup(ctx, backup_dir):
    """
    Create backup of current season data.
    """
    season = ctx.obj['season']
    
    click.echo("💾 Creating data backup...")
    
    try:
        data_manager = DataManager(season)
        backup_path = data_manager.create_backup(backup_dir)
        
        click.echo(f"✅ Backup created: {backup_path}")
        
    except Exception as e:
        logger.error(f"Backup failed: {str(e)}")
        click.echo(f"❌ Error: {str(e)}", err=True)

# Helper functions for display
def _display_player_analysis(analysis):
    """Display detailed player analysis"""
    player = analysis['player_info']
    
    click.echo(f"\n🏃 {player['name']} ({player['team']} - {player['position']})")
    click.echo("=" * 50)
    click.echo(f"Price: £{player['price']}m | Form: {analysis['form_score']:.1f}")
    click.echo(f"Total Points: {player['total_points']} | PPG: {analysis['points_per_game']:.1f}")
    
    click.echo("\n📊 Recent Performance:")
    for gw in analysis['recent_gameweeks'][-5:]:
        click.echo(f"  GW{gw['gameweek']}: {gw['points']} pts vs {gw['opponent']}")
    
    click.echo(f"\n🔮 Next Fixture: vs {analysis['next_fixture']['opponent']}")
    click.echo(f"   Difficulty: {analysis['next_fixture']['difficulty']}/5")
    click.echo(f"   Expected Points: {analysis['predicted_points']:.1f}")

def _display_position_analysis(analysis):
    """Display position analysis"""
    click.echo(f"\n📈 Top {analysis['position']} Players:")
    click.echo("-" * 60)
    
    for i, player in enumerate(analysis['top_players'][:10], 1):
        click.echo(f"{i:2d}. {player['name']:<20} {player['team']:<3} "
                  f"£{player['price']:<4} {player['total_points']:>3}pts "
                  f"({player['form']:.1f})")

def _display_predictions(predictions, top):
    """Display gameweek predictions"""
    click.echo(f"\n🔮 Top {top} Predictions:")
    click.echo("-" * 70)
    
    for i, pred in enumerate(predictions['player_predictions'][:top], 1):
        click.echo(f"{i:2d}. {pred['name']:<20} {pred['team']:<3} "
                  f"£{pred['price']:<4} EP: {pred['expected_points']:.1f} "
                  f"({pred['confidence']:.0%})")
    
    if predictions.get('captain_pick'):
        captain = predictions['captain_pick']
        click.echo(f"\n⭐ Captain Pick: {captain['name']} (EP: {captain['expected_points']:.1f})")

if __name__ == '__main__':
    cli()