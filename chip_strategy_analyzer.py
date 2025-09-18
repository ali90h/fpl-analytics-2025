#!/usr/bin/env python3
"""
FPL Chip Strategy Analyzer
Automated recommendation engine for optimal chip usage timing
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import requests

class ChipStrategyAnalyzer:
    """Analyzes optimal timing for FPL chips based on fixtures, team state, and calendar"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.current_gw = self._get_current_gameweek()
        
        # Chip definitions with availability windows
        self.chips = {
            'wildcard': {
                'name': 'Wildcard',
                'description': 'Unlimited free transfers for one gameweek',
                'max_uses': 2,  # 1 before GW20, 1 after GW20
                'cost': 0,
                'effect': 'Complete team overhaul'
            },
            'bench_boost': {
                'name': 'Bench Boost', 
                'description': 'Points from bench players count',
                'max_uses': 1,
                'cost': 0,
                'effect': 'All 15 players score points'
            },
            'triple_captain': {
                'name': 'Triple Captain',
                'description': 'Captain gets 3x points instead of 2x',
                'max_uses': 1,
                'cost': 0,
                'effect': 'Extra captain multiplier'
            },
            'free_hit': {
                'name': 'Free Hit',
                'description': 'Make unlimited transfers for one GW, then revert',
                'max_uses': 1,
                'cost': 0,
                'effect': 'One-gameweek team change'
            }
        }
    
    def _get_current_gameweek(self) -> int:
        """Get current gameweek from FPL API or local data"""
        try:
            # Try to get from current team data
            team_file = self.data_dir / "current_team.json"
            if team_file.exists():
                with open(team_file) as f:
                    team_data = json.load(f)
                    return team_data.get('gameweek', 4)
        except:
            pass
        
        # Default to gameweek 4 based on current context
        return 4
    
    def analyze_chip_usage_history(self, team_id: Optional[int] = None) -> Dict:
        """Analyze which chips have been used and which are available"""
        # Try to get real chip usage data from current team
        try:
            team_file = self.data_dir / "current_team.json"
            if team_file.exists():
                with open(team_file) as f:
                    team_data = json.load(f)
                    
                # For now, assume all chips available in early season
                # In real implementation, this would check chip usage history from FPL API
                available_chips = {
                    'wildcard_1': True,  # First wildcard (GW2-19) 
                    'wildcard_2': True,  # Second wildcard (GW20-38)
                    'bench_boost': True,
                    'triple_captain': True,
                    'free_hit': True
                }
                
                # Check if we're past GW19 for first wildcard
                if self.current_gw > 19:
                    available_chips['wildcard_1'] = False
                
                used_chips = []
                
                return {
                    'available_chips': available_chips,
                    'used_chips': used_chips,
                    'chips_remaining': sum(available_chips.values())
                }
        except:
            pass
        
        # Default assumption - all chips available
        available_chips = {
            'wildcard_1': True if self.current_gw <= 19 else False,
            'wildcard_2': True,
            'bench_boost': True,
            'triple_captain': True,
            'free_hit': True
        }
        
        used_chips = []
        
        return {
            'available_chips': available_chips,
            'used_chips': used_chips,
            'chips_remaining': sum(available_chips.values())
        }
    
    def analyze_upcoming_fixtures(self, gameweeks: range) -> Dict:
        """Analyze fixture difficulty for upcoming gameweeks"""
        fixture_analysis = {}
        
        try:
            # Load fixture data
            fixtures_file = self.data_dir / "fixtures_latest.csv"
            if fixtures_file.exists():
                fixtures_df = pd.read_csv(fixtures_file)
                
                for gw in gameweeks:
                    gw_fixtures = fixtures_df[fixtures_df['event'] == gw]
                    
                    if not gw_fixtures.empty:
                        # Calculate average difficulty
                        home_diff = gw_fixtures['team_h_difficulty'].mean()
                        away_diff = gw_fixtures['team_a_difficulty'].mean()
                        avg_difficulty = (home_diff + away_diff) / 2
                        
                        # Count fixtures
                        total_fixtures = len(gw_fixtures)
                        
                        # Identify double gameweeks (teams with 2+ fixtures)
                        home_teams = gw_fixtures['team_h'].tolist()
                        away_teams = gw_fixtures['team_a'].tolist()
                        all_teams = home_teams + away_teams
                        team_fixture_counts = pd.Series(all_teams).value_counts()
                        double_gw_teams = team_fixture_counts[team_fixture_counts >= 2].index.tolist()
                        
                        fixture_analysis[gw] = {
                            'avg_difficulty': avg_difficulty,
                            'total_fixtures': total_fixtures,
                            'is_double_gameweek': len(double_gw_teams) > 0,
                            'double_gw_teams': double_gw_teams,
                            'is_blank_gameweek': total_fixtures < 10,
                            'difficulty_rating': self._classify_difficulty(avg_difficulty)
                        }
                    else:
                        # Blank gameweek
                        fixture_analysis[gw] = {
                            'avg_difficulty': 5.0,
                            'total_fixtures': 0,
                            'is_double_gameweek': False,
                            'double_gw_teams': [],
                            'is_blank_gameweek': True,
                            'difficulty_rating': 'Blank'
                        }
        
        except Exception as e:
            print(f"Warning: Could not analyze fixtures - {e}")
            
        return fixture_analysis
    
    def _classify_difficulty(self, avg_difficulty: float) -> str:
        """Classify average difficulty rating"""
        if avg_difficulty <= 2.5:
            return 'Very Easy'
        elif avg_difficulty <= 3.0:
            return 'Easy'
        elif avg_difficulty <= 3.5:
            return 'Medium'
        elif avg_difficulty <= 4.0:
            return 'Hard'
        else:
            return 'Very Hard'
    
    def recommend_wildcard_timing(self, fixture_analysis: Dict, current_team_strength: float = 0.7) -> Dict:
        """Recommend optimal wildcard timing"""
        recommendations = []
        
        # Analyze fixture runs for wildcard opportunities
        gameweeks = sorted(fixture_analysis.keys())
        
        # Look for easy fixture runs (3+ consecutive easy gameweeks)
        easy_runs = []
        current_run = []
        
        for gw in gameweeks:
            if gw <= 19:  # First wildcard window
                difficulty = fixture_analysis[gw]['avg_difficulty']
                if difficulty <= 3.0:  # Easy or medium
                    current_run.append(gw)
                else:
                    if len(current_run) >= 3:
                        easy_runs.append({
                            'start_gw': min(current_run),
                            'end_gw': max(current_run),
                            'length': len(current_run),
                            'avg_difficulty': sum(fixture_analysis[gw]['avg_difficulty'] for gw in current_run) / len(current_run)
                        })
                    current_run = []
        
        # Add final run if exists
        if len(current_run) >= 3:
            easy_runs.append({
                'start_gw': min(current_run),
                'end_gw': max(current_run),
                'length': len(current_run),
                'avg_difficulty': sum(fixture_analysis[gw]['avg_difficulty'] for gw in current_run) / len(current_run)
            })
        
        # Recommend best wildcard timing
        if easy_runs:
            best_run = min(easy_runs, key=lambda x: x['avg_difficulty'])
            recommendations.append({
                'chip': 'wildcard',
                'recommended_gw': best_run['start_gw'],
                'confidence': 'High' if best_run['avg_difficulty'] <= 2.5 else 'Medium',
                'reason': f"Easy fixture run GW{best_run['start_gw']}-{best_run['end_gw']} (avg difficulty: {best_run['avg_difficulty']:.1f})",
                'benefit': f"Build team for {best_run['length']}-gameweek easy run"
            })
        
        # Team strength consideration
        if current_team_strength < 0.5:
            recommendations.append({
                'chip': 'wildcard',
                'recommended_gw': self.current_gw + 1,
                'confidence': 'High',
                'reason': 'Current team needs major overhaul',
                'benefit': 'Immediate team improvement needed'
            })
        
        return recommendations
    
    def recommend_bench_boost_timing(self, fixture_analysis: Dict) -> List[Dict]:
        """Recommend optimal bench boost timing"""
        recommendations = []
        
        # Look for double gameweeks
        double_gameweeks = [gw for gw, data in fixture_analysis.items() 
                           if data.get('is_double_gameweek', False)]
        
        if double_gameweeks:
            # Recommend earliest double gameweek
            best_dgw = min(double_gameweeks)
            recommendations.append({
                'chip': 'bench_boost',
                'recommended_gw': best_dgw,
                'confidence': 'High',
                'reason': f'Double gameweek with {len(fixture_analysis[best_dgw]["double_gw_teams"])} teams playing twice',
                'benefit': 'Maximum points from all 15 players'
            })
        else:
            # Look for gameweeks with easy fixtures for bench players
            easy_gameweeks = [gw for gw, data in fixture_analysis.items() 
                            if data.get('avg_difficulty', 5) <= 2.5]
            
            if easy_gameweeks:
                best_easy_gw = min(easy_gameweeks)
                recommendations.append({
                    'chip': 'bench_boost',
                    'recommended_gw': best_easy_gw,
                    'confidence': 'Medium',
                    'reason': f'Easy fixtures (avg difficulty: {fixture_analysis[best_easy_gw]["avg_difficulty"]:.1f})',
                    'benefit': 'Good scoring potential for bench players'
                })
        
        return recommendations
    
    def recommend_triple_captain_timing(self, fixture_analysis: Dict, top_captains: List[str] = None) -> List[Dict]:
        """Recommend optimal triple captain timing"""
        recommendations = []
        
        # Get current top attacking players if not provided
        if not top_captains:
            top_captains = self._get_premium_attackers()
        
        # Look for double gameweeks first (priority)
        double_gameweeks = [gw for gw, data in fixture_analysis.items() 
                           if data.get('is_double_gameweek', False)]
        
        if double_gameweeks:
            best_dgw = min(double_gameweeks)
            recommendations.append({
                'chip': 'triple_captain',
                'recommended_gw': best_dgw,
                'confidence': 'High',
                'reason': f'Double gameweek - captain plays twice (GW{best_dgw})',
                'benefit': 'Potential for massive captain hauls with 2 fixtures',
                'suggested_captains': top_captains[:3]
            })
        else:
            # Look for easy fixtures for premium attackers
            easy_gameweeks = [(gw, data) for gw, data in fixture_analysis.items() 
                            if data.get('avg_difficulty', 5) <= 2.5 and not data.get('is_blank_gameweek', False)]
            
            if easy_gameweeks:
                # Sort by easiest fixtures
                easy_gameweeks.sort(key=lambda x: x[1]['avg_difficulty'])
                best_gw, best_data = easy_gameweeks[0]
                
                recommendations.append({
                    'chip': 'triple_captain',
                    'recommended_gw': best_gw,
                    'confidence': 'Medium',
                    'reason': f'Very easy fixtures (avg difficulty: {best_data["avg_difficulty"]:.1f})',
                    'benefit': 'Higher ceiling for captain points against weak defenses',
                    'suggested_captains': top_captains[:3]
                })
            else:
                # Look for any reasonable fixtures
                reasonable_gws = [(gw, data) for gw, data in fixture_analysis.items() 
                                if data.get('avg_difficulty', 5) <= 3.5 and not data.get('is_blank_gameweek', False)]
                
                if reasonable_gws:
                    reasonable_gws.sort(key=lambda x: x[1]['avg_difficulty'])
                    best_gw, best_data = reasonable_gws[0]
                    
                    recommendations.append({
                        'chip': 'triple_captain',
                        'recommended_gw': best_gw,
                        'confidence': 'Low',
                        'reason': f'Best available fixtures (avg difficulty: {best_data["avg_difficulty"]:.1f})',
                        'benefit': 'Consider saving for better opportunity or double gameweek',
                        'suggested_captains': top_captains[:3]
                    })
        
        return recommendations
    
    def _get_premium_attackers(self) -> List[str]:
        """Get list of premium attacking players for captaincy"""
        # Try to load current top performers from data
        try:
            # Check for current team data to identify owned premium players
            team_file = self.data_dir / "current_team.json"
            if team_file.exists():
                with open(team_file) as f:
                    team_data = json.load(f)
                    
                # Get attacking players from current team
                premium_attackers = []
                for pick in team_data.get('picks', []):
                    if pick.get('position_name') in ['MID', 'FWD'] and pick.get('now_cost', 0) >= 80:
                        premium_attackers.append(pick['web_name'])
                
                if premium_attackers:
                    return premium_attackers[:5]
        except:
            pass
        
        # Default premium attackers for 2025/26 season
        return ["Haaland", "M.Salah", "Son", "Palmer", "Watkins"]
    
    def recommend_free_hit_timing(self, fixture_analysis: Dict) -> List[Dict]:
        """Recommend optimal free hit timing"""
        recommendations = []
        
        # Look for blank gameweeks
        blank_gameweeks = [gw for gw, data in fixture_analysis.items() 
                          if data.get('is_blank_gameweek', False)]
        
        if blank_gameweeks:
            # Recommend for biggest blank gameweek
            worst_blank = min(blank_gameweeks, key=lambda gw: fixture_analysis[gw]['total_fixtures'])
            recommendations.append({
                'chip': 'free_hit',
                'recommended_gw': worst_blank,
                'confidence': 'High',
                'reason': f'Blank gameweek - only {fixture_analysis[worst_blank]["total_fixtures"]} fixtures',
                'benefit': 'Build temporary team for limited fixtures'
            })
        else:
            # Look for double gameweeks if no blanks
            double_gameweeks = [gw for gw, data in fixture_analysis.items() 
                               if data.get('is_double_gameweek', False)]
            
            if double_gameweeks:
                best_dgw = min(double_gameweeks)
                recommendations.append({
                    'chip': 'free_hit',
                    'recommended_gw': best_dgw,
                    'confidence': 'Medium',
                    'reason': f'Double gameweek opportunity',
                    'benefit': 'Temporary team loaded with double gameweek players'
                })
        
        return recommendations
    
    def get_comprehensive_chip_advice(self, gameweek_range: range = None) -> Dict:
        """Get comprehensive advice for all chips"""
        if gameweek_range is None:
            gameweek_range = range(self.current_gw, min(self.current_gw + 10, 39))
        
        # Analyze current chip status
        chip_status = self.analyze_chip_usage_history()
        
        # Analyze fixtures
        fixture_analysis = self.analyze_upcoming_fixtures(gameweek_range)
        
        # Get recommendations for each chip
        all_recommendations = {}
        
        if chip_status['available_chips'].get('wildcard_1', False):
            all_recommendations['wildcard'] = self.recommend_wildcard_timing(fixture_analysis)
        
        if chip_status['available_chips'].get('bench_boost', False):
            all_recommendations['bench_boost'] = self.recommend_bench_boost_timing(fixture_analysis)
        
        if chip_status['available_chips'].get('triple_captain', False):
            all_recommendations['triple_captain'] = self.recommend_triple_captain_timing(fixture_analysis)
        
        if chip_status['available_chips'].get('free_hit', False):
            all_recommendations['free_hit'] = self.recommend_free_hit_timing(fixture_analysis)
        
        return {
            'current_gameweek': self.current_gw,
            'chip_status': chip_status,
            'fixture_analysis': fixture_analysis,
            'recommendations': all_recommendations,
            'priority_order': self._prioritize_chip_usage(all_recommendations)
        }
    
    def _prioritize_chip_usage(self, recommendations: Dict) -> List[Dict]:
        """Prioritize chip usage based on value and timing"""
        priorities = []
        
        for chip_type, chip_recs in recommendations.items():
            if chip_recs:
                for rec in chip_recs:
                    priority_score = self._calculate_priority_score(rec)
                    priorities.append({
                        **rec,
                        'priority_score': priority_score
                    })
        
        # Sort by priority score (higher = more important)
        return sorted(priorities, key=lambda x: x['priority_score'], reverse=True)
    
    def _calculate_priority_score(self, recommendation: Dict) -> float:
        """Calculate priority score for a chip recommendation"""
        score = 0.0
        
        # Confidence multiplier
        confidence_scores = {'High': 1.0, 'Medium': 0.7, 'Low': 0.4}
        score += confidence_scores.get(recommendation['confidence'], 0.5)
        
        # Urgency based on gameweek timing
        gw_urgency = max(0, 1.0 - (recommendation['recommended_gw'] - self.current_gw) / 10)
        score += gw_urgency
        
        # Chip value multiplier
        chip_values = {
            'wildcard': 1.0,
            'triple_captain': 0.8,
            'bench_boost': 0.7,
            'free_hit': 0.6
        }
        score *= chip_values.get(recommendation['chip'], 0.5)
        
        return score
    
    def display_chip_advice(self, advice: Dict):
        """Display formatted chip advice"""
        print(f"\n🎴 Chip Strategy Analysis (GW{advice['current_gameweek']})")
        print("=" * 70)
        
        # Chip status
        available = advice['chip_status']['available_chips']
        remaining = advice['chip_status']['chips_remaining']
        
        print(f"💳 Available Chips: {remaining}/5")
        print("   Wildcard 1:", "✅" if available.get('wildcard_1') else "❌")
        print("   Bench Boost:", "✅" if available.get('bench_boost') else "❌") 
        print("   Triple Captain:", "✅" if available.get('triple_captain') else "❌")
        print("   Free Hit:", "✅" if available.get('free_hit') else "❌")
        
        # Priority recommendations
        if advice['priority_order']:
            print(f"\n🎯 Priority Chip Recommendations:")
            print("-" * 50)
            
            for i, rec in enumerate(advice['priority_order'][:3], 1):
                confidence_icon = "🎯" if rec['confidence'] == 'High' else "⚡" if rec['confidence'] == 'Medium' else "❓"
                
                print(f"{i}. {confidence_icon} {rec['chip'].upper().replace('_', ' ')} - GW{rec['recommended_gw']}")
                print(f"   Confidence: {rec['confidence']}")
                print(f"   Reason: {rec['reason']}")
                print(f"   Benefit: {rec['benefit']}")
                if 'suggested_captains' in rec:
                    print(f"   Suggested Captains: {', '.join(rec['suggested_captains'])}")
                print()
        else:
            print("\n📝 No immediate chip recommendations")
            print("   Consider saving chips for double/blank gameweeks")
        
        # Fixture outlook
        print(f"\n📅 Fixture Outlook (Next 6 GWs):")
        print("-" * 40)
        
        for gw in sorted(advice['fixture_analysis'].keys())[:6]:
            data = advice['fixture_analysis'][gw]
            
            if data['is_double_gameweek']:
                icon = "⚡"
                status = "DOUBLE GW"
            elif data['is_blank_gameweek']:
                icon = "🚫"
                status = "BLANK GW"
            else:
                icon = "📅"
                status = data['difficulty_rating']
            
            print(f"   GW{gw}: {icon} {status} ({data['total_fixtures']} fixtures)")

def main():
    """Test the chip strategy analyzer"""
    analyzer = ChipStrategyAnalyzer()
    advice = analyzer.get_comprehensive_chip_advice()
    analyzer.display_chip_advice(advice)

if __name__ == "__main__":
    main()