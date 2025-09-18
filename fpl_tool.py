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
        def collect_gameweek_data(self, gameweek):
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
            return {
                "current_gameweek": 1,
                "gameweeks_collected": 0,
                "total_players": 0,
                "last_update": "Not available",
                "data_quality": 0,
                "recent_gameweeks": []
            }
        def gameweek_data_exists(self, gameweek):
            return False
        def save_gameweek_data(self, gameweek, fpl_data, match_data):
            return False
        def recalculate_form_metrics(self, gameweek):
            pass
    
    class FootballAnalyzer:
        def __init__(self, data_manager):
            pass
        def analyze_position_trends(self, gameweeks=5):
            return {"error": "Analyzer not available"}
        def analyze_player_performance(self, player_id, detailed=False):
            return {"error": "Analyzer not available"}
        def find_value_picks(self, max_price=7.0, min_gameweeks=3):
            return []
        def compare_players(self, player_ids):
            return {"error": "Analyzer not available"}
        def predict_player_points(self, player_id):
            return {"error": "Analyzer not available"}
        def analyze_gameweek_top_performers(self, gameweek, count=10):
            return []


@click.group()
@click.option('--season', default='2025-26', help='Season to analyze (default: 2025-26)')
@click.pass_context
def cli(ctx, season):
    """
    🏈 Football Analytics Tool for 2025/26 Season
    
    Terminal-based tool for collecting and analyzing fantasy football data.
    Focus on official FPL data and trusted sources only.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    ctx.obj['season'] = season
    ctx.obj['data_manager'] = DataManager(season) if COMPONENTS_AVAILABLE else None
    
    # Welcome message
    if ctx.invoked_subcommand is None:
        click.echo(f"🏈 Football Analytics Tool - Season {season}")
        click.echo("Use --help to see available commands")


@cli.command()
@click.option('--gameweek', type=int, help='Specific gameweek to collect (optional)')
@click.option('--force', is_flag=True, help='Force re-collection even if data exists')
@click.pass_context
def collect(ctx, gameweek, force):
    """
    📊 Collect fresh data from official FPL API and match sources
    
    Gathers player data, team information, and match context.
    Creates organized data structure for analysis.
    """
    season = ctx.obj.get('season', '2025-26')
    click.echo(f"🔄 Collecting data for season {season}...")
    
    try:
        # Initialize collectors
        fpl_collector = FPLDataCollector()
        match_collector = MatchDataCollector()
        data_manager = ctx.obj.get('data_manager')
        
        # Collect bootstrap data (teams, players, general info)
        click.echo("📡 Fetching FPL bootstrap data...")
        bootstrap_data = fpl_collector.get_bootstrap_static()
        
        if gameweek:
            # Collect specific gameweek
            click.echo(f"📅 Collecting gameweek {gameweek} data...")
            
            # Check if data already exists
            if data_manager and data_manager.gameweek_data_exists(gameweek) and not force:
                click.echo(f"⚠️  Data for gameweek {gameweek} already exists. Use --force to re-collect.")
                return
            
            # Collect FPL gameweek data
            gameweek_data = fpl_collector.get_gameweek_live_data(gameweek)
            
            # Collect match context
            match_data = match_collector.get_gameweek_matches(gameweek)
            
            # Save data using data manager
            if data_manager:
                success = data_manager.save_gameweek_data(gameweek, gameweek_data, match_data)
                if success:
                    click.echo(f"✅ Gameweek {gameweek} data collected and saved successfully")
                    
                    # Recalculate form metrics
                    data_manager.recalculate_form_metrics(gameweek)
                    click.echo("📈 Form metrics updated")
                else:
                    click.echo(f"❌ Failed to save gameweek {gameweek} data")
            else:
                # Fallback: save raw data
                os.makedirs('data/raw', exist_ok=True)
                with open(f'data/raw/gameweek_{gameweek}.json', 'w') as f:
                    json.dump({
                        'fpl_data': gameweek_data,
                        'match_data': match_data,
                        'collected_at': datetime.now().isoformat()
                    }, f, indent=2)
                click.echo(f"✅ Gameweek {gameweek} data saved to data/raw/")
        else:
            # Collect current/latest gameweek
            click.echo("📅 Collecting latest gameweek data...")
            current_gw = bootstrap_data.get('events', [{}])[0].get('id', 1) if bootstrap_data else 1
            
            gameweek_data = fpl_collector.get_gameweek_live_data(current_gw)
            match_data = match_collector.get_gameweek_matches(current_gw)
            
            if data_manager:
                success = data_manager.save_gameweek_data(current_gw, gameweek_data, match_data)
                if success:
                    click.echo(f"✅ Current gameweek {current_gw} data collected successfully")
                else:
                    click.echo(f"❌ Failed to save current gameweek data")
        
        # Show collection summary
        if bootstrap_data:
            teams_count = len(bootstrap_data.get('teams', []))
            players_count = len(bootstrap_data.get('elements', []))
            click.echo(f"📋 Collected data for {teams_count} teams and {players_count} players")
        
    except Exception as e:
        click.echo(f"❌ Error collecting data: {str(e)}")
        if click.confirm("Show detailed error information?"):
            import traceback
            click.echo(traceback.format_exc())


@cli.command()
@click.option('--from-gameweek', type=int, help='Update from specific gameweek onwards')
@click.option('--to-gameweek', type=int, help='Update up to specific gameweek')
@click.pass_context
def update(ctx, from_gameweek, to_gameweek):
    """
    🔄 Update existing data with latest gameweek information
    
    Performs incremental updates to keep data current.
    Designed for round-by-round updates after each gameweek.
    """
    season = ctx.obj.get('season', '2025-26')
    data_manager = ctx.obj.get('data_manager')
    
    click.echo(f"🔄 Updating data for season {season}...")
    
    try:
        if not data_manager:
            click.echo("❌ Data manager not available")
            return
        
        # Get current data status
        status = data_manager.get_data_status()
        current_gw = status.get('current_gameweek', 1)
        
        if from_gameweek:
            start_gw = from_gameweek
        else:
            start_gw = current_gw
        
        if to_gameweek:
            end_gw = to_gameweek
        else:
            # Try to get next gameweek
            end_gw = start_gw + 1
        
        click.echo(f"📅 Updating gameweeks {start_gw} to {end_gw}")
        
        # Initialize collectors
        fpl_collector = FPLDataCollector()
        match_collector = MatchDataCollector()
        
        updated_gameweeks = []
        
        for gw in range(start_gw, end_gw + 1):
            click.echo(f"🔄 Processing gameweek {gw}...")
            
            try:
                # Collect updated data
                gameweek_data = fpl_collector.get_gameweek_live_data(gw)
                match_data = match_collector.get_gameweek_matches(gw)
                
                # Save data
                success = data_manager.save_gameweek_data(gw, gameweek_data, match_data)
                
                if success:
                    updated_gameweeks.append(gw)
                    click.echo(f"✅ Updated gameweek {gw}")
                else:
                    click.echo(f"⚠️  Failed to update gameweek {gw}")
                    
            except Exception as e:
                click.echo(f"❌ Error updating gameweek {gw}: {str(e)}")
        
        if updated_gameweeks:
            click.echo(f"📈 Recalculating form metrics for updated gameweeks...")
            for gw in updated_gameweeks:
                data_manager.recalculate_form_metrics(gw)
            
            click.echo(f"✅ Successfully updated {len(updated_gameweeks)} gameweeks")
        else:
            click.echo("⚠️  No gameweeks were updated")
    
    except Exception as e:
        click.echo(f"❌ Error during update: {str(e)}")


@cli.command()
@click.option('--player', type=int, help='Analyze specific player by ID')
@click.option('--gameweek', type=int, help='Analyze specific gameweek')
@click.option('--position', help='Analyze players by position (GK, DEF, MID, FWD)')
@click.option('--value-picks', is_flag=True, help='Find value picks under specified price')
@click.option('--max-price', type=float, default=7.0, help='Maximum price for value picks')
@click.option('--detailed', is_flag=True, help='Show detailed analysis')
@click.pass_context
def analyze(ctx, player, gameweek, position, value_picks, max_price, detailed):
    """
    📊 Analyze player performances, trends, and identify opportunities
    
    Provides focused insights for fantasy football decision making.
    Multiple analysis options available.
    """
    season = ctx.obj.get('season', '2025-26')
    data_manager = ctx.obj.get('data_manager')
    
    if not data_manager:
        click.echo("❌ Data manager not available. Run 'collect' command first.")
        return
    
    try:
        # Initialize analyzer
        analyzer = FootballAnalyzer(data_manager)
        
        if player:
            # Analyze specific player
            click.echo(f"🔍 Analyzing player {player}...")
            analysis = analyzer.analyze_player_performance(player, detailed=detailed)
            
            if "error" in analysis:
                click.echo(f"❌ {analysis['error']}")
                return
            
            # Display player analysis
            click.echo(f"\n👤 {analysis['name']} ({analysis['team']}) - {analysis['position']}")
            click.echo(f"💰 Price: £{analysis['current_price']}m | Total Points: {analysis['total_points']}")
            click.echo(f"📊 PPG: {analysis['points_per_game']} | Consistency: {analysis['consistency_score']}")
            click.echo(f"📈 Form: {analysis['form_trend']} | Value: {analysis['value_rating']}")
            click.echo(f"🏥 Injury Risk: {analysis['injury_risk']}/3")
            click.echo(f"💡 Recommendation: {analysis['recommendation']}")
            
            if detailed and 'games_played' in analysis:
                click.echo(f"\n📋 Detailed Stats:")
                click.echo(f"   Games: {analysis['games_played']} | Minutes/Game: {analysis['minutes_per_game']}")
                click.echo(f"   Goals/Game: {analysis['goals_per_game']} | Assists/Game: {analysis['assists_per_game']}")
                click.echo(f"   Bonus Points: {analysis['bonus_points']} | Clean Sheets: {analysis['clean_sheets']}")
        
        elif gameweek:
            # Analyze specific gameweek
            click.echo(f"📅 Analyzing gameweek {gameweek}...")
            performers = analyzer.analyze_gameweek_top_performers(gameweek, count=10)
            
            if not performers:
                click.echo(f"❌ No performance data found for gameweek {gameweek}")
                return
            
            click.echo(f"\n🏆 Top Performers - Gameweek {gameweek}")
            click.echo("─" * 70)
            
            for i, performer in enumerate(performers, 1):
                click.echo(f"{i:2d}. {performer['name']:20} ({performer['team']}) - {performer['points']:2d} pts")
                click.echo(f"    {performer['position']} vs {performer['opponent']} ({performer['venue']}) - {performer['performance_rating']}")
        
        elif value_picks:
            # Find value picks
            click.echo(f"💎 Finding value picks under £{max_price}m...")
            picks = analyzer.find_value_picks(max_price=max_price, min_gameweeks=3)
            
            if not picks:
                click.echo(f"❌ No value picks found under £{max_price}m")
                return
            
            click.echo(f"\n💰 Value Picks (Under £{max_price}m)")
            click.echo("─" * 80)
            
            for pick in picks[:10]:  # Top 10
                strength_icon = "🔥" if pick['recommendation_strength'] == "STRONG" else "⭐"
                click.echo(f"{strength_icon} {pick['name']:20} ({pick['team']}) - £{pick['price']}m")
                click.echo(f"   {pick['position']} | PPG: {pick['avg_points']} | Form: {pick['recent_form']} | Value Score: {pick['value_score']}")
        
        else:
            # General position trends analysis
            click.echo("📊 Analyzing position trends...")
            trends = analyzer.analyze_position_trends(gameweeks=5)
            
            if "error" in trends:
                click.echo(f"❌ {trends['error']}")
                return
            
            click.echo(f"\n📈 Position Trends ({trends['analysis_period']})")
            click.echo("─" * 60)
            
            for position, data in trends['positions'].items():
                trend_icon = "📈" if data['trend'] == "IMPROVING" else "📉" if data['trend'] == "DECLINING" else "➡️"
                click.echo(f"{trend_icon} {position}:")
                click.echo(f"   Avg Points: {data['avg_points']} | High Scorers: {data['high_scorers']} | Consistency: {data['consistency']}")
            
            if trends.get('insights'):
                click.echo(f"\n💡 Key Insights:")
                for insight in trends['insights']:
                    click.echo(f"   • {insight}")
    
    except Exception as e:
        click.echo(f"❌ Error during analysis: {str(e)}")
        if click.confirm("Show detailed error information?"):
            import traceback
            click.echo(traceback.format_exc())


@cli.command()
@click.option('--player', type=int, help='Predict points for specific player')
@click.option('--compare', help='Compare players (comma-separated IDs)')
@click.option('--gameweeks', type=int, default=1, help='Number of gameweeks to predict')
@click.pass_context
def predict(ctx, player, compare, gameweeks):
    """
    🔮 Predict player performance and compare options
    
    Uses historical data and form trends for predictions.
    Helps with transfer and captaincy decisions.
    """
    season = ctx.obj.get('season', '2025-26')
    data_manager = ctx.obj.get('data_manager')
    
    if not data_manager:
        click.echo("❌ Data manager not available. Run 'collect' command first.")
        return
    
    try:
        analyzer = FootballAnalyzer(data_manager)
        
        if compare:
            # Compare multiple players
            try:
                player_ids = [int(pid.strip()) for pid in compare.split(',')]
                if len(player_ids) < 2:
                    click.echo("❌ Need at least 2 player IDs for comparison")
                    return
                
                click.echo(f"⚖️  Comparing {len(player_ids)} players...")
                comparison = analyzer.compare_players(player_ids)
                
                if "error" in comparison:
                    click.echo(f"❌ {comparison['error']}")
                    return
                
                # Display comparison results
                click.echo(f"\n📊 Player Comparison")
                click.echo("─" * 80)
                
                # Show individual player stats
                for player_data in comparison['players']:
                    click.echo(f"\n👤 {player_data['name']} ({player_data['team']}) - {player_data['position']}")
                    click.echo(f"   PPG: {player_data['points_per_game']} | Consistency: {player_data['consistency_score']}")
                    click.echo(f"   Value: {player_data['value_rating']} | Risk: {player_data['injury_risk']}/3")
                    click.echo(f"   Recommendation: {player_data['recommendation']}")
                
                # Show winners by metric
                click.echo(f"\n🏆 Best in Category:")
                for metric, winner in comparison['winner_by_metric'].items():
                    metric_name = metric.replace('_', ' ').title()
                    click.echo(f"   {metric_name}: {winner}")
                
                # Show overall ranking
                click.echo(f"\n🥇 Overall Ranking:")
                for rank_data in comparison['overall_ranking']:
                    rank_icon = "🥇" if rank_data['rank'] == 1 else "🥈" if rank_data['rank'] == 2 else "🥉" if rank_data['rank'] == 3 else f"{rank_data['rank']}."
                    click.echo(f"   {rank_icon} {rank_data['name']} (Score: {rank_data['overall_score']})")
                
            except ValueError:
                click.echo("❌ Invalid player IDs. Use comma-separated numbers (e.g., 123,456,789)")
                return
        
        elif player:
            # Predict for specific player
            click.echo(f"🔮 Predicting points for player {player}...")
            prediction = analyzer.predict_player_points(player)
            
            if "error" in prediction:
                click.echo(f"❌ {prediction['error']}")
                return
            
            # Display prediction
            confidence_icon = "🎯" if prediction['confidence'] >= 0.8 else "⚡" if prediction['confidence'] >= 0.6 else "❓"
            recommendation_icon = "🔥" if prediction['recommendation'] == "BUY" else "📊" if prediction['recommendation'] == "HOLD" else "📉"
            
            click.echo(f"\n{confidence_icon} Points Prediction")
            click.echo("─" * 40)
            click.echo(f"Predicted Points: {prediction['predicted_points']}")
            click.echo(f"Confidence: {prediction['confidence']:.0%}")
            click.echo(f"Base Average: {prediction['baseline_avg']}")
            click.echo(f"Form Multiplier: {prediction['form_multiplier']}")
            click.echo(f"{recommendation_icon} Recommendation: {prediction['recommendation']}")
        
        else:
            click.echo("❌ Specify --player ID or --compare 'id1,id2,id3' for predictions")
    
    except Exception as e:
        click.echo(f"❌ Error during prediction: {str(e)}")


@cli.command()
@click.pass_context
def status(ctx):
    """
    📋 Show current data status and collection summary
    
    Displays information about collected gameweeks, data quality,
    and system status for the current season.
    """
    season = ctx.obj.get('season', '2025-26')
    data_manager = ctx.obj.get('data_manager')
    
    click.echo(f"📊 Data Status - Season {season}")
    click.echo("=" * 50)
    
    try:
        if not data_manager:
            click.echo("❌ Data manager not available")
            click.echo("💡 Run 'python fpl_tool.py collect' to initialize data collection")
            return
        
        status = data_manager.get_data_status()
        
        # Basic status
        click.echo(f"Current Gameweek: {status['current_gameweek']}")
        click.echo(f"Gameweeks Collected: {status['gameweeks_collected']}")
        click.echo(f"Total Players: {status['total_players']}")
        click.echo(f"Last Update: {status['last_update']}")
        click.echo(f"Data Quality: {status['data_quality']}%")
        
        # Recent gameweeks
        if status.get('recent_gameweeks'):
            click.echo(f"\n📅 Recent Gameweeks:")
            click.echo("─" * 40)
            
            for gw_data in status['recent_gameweeks']:
                quality_icon = "✅" if gw_data['updated_players'] > 500 else "⚠️" if gw_data['updated_players'] > 300 else "❌"
                click.echo(f"{quality_icon} GW {gw_data['gameweek']}: {gw_data['updated_players']} players | Avg: {gw_data['avg_points']:.1f} pts")
        
        # Recommendations
        click.echo(f"\n💡 Recommendations:")
        if status['gameweeks_collected'] == 0:
            click.echo("   • Run 'collect' to gather initial data")
        elif status['data_quality'] < 80:
            click.echo("   • Data quality is low - consider re-collecting recent gameweeks")
        else:
            click.echo("   • System ready for analysis and predictions")
            click.echo("   • Use 'update' to refresh with latest gameweek data")
    
    except Exception as e:
        click.echo(f"❌ Error retrieving status: {str(e)}")


if __name__ == '__main__':
    cli()