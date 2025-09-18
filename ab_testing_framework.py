"""
A/B Testing Framework for FPL Model Performance
Compare different model versions and configurations
"""

import numpy as np
import pandas as pd
import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, asdict
from scipy import stats
import joblib
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ABTestConfig:
    """Configuration for A/B test"""
    test_name: str
    description: str
    start_date: datetime
    end_date: datetime
    traffic_split: float  # Percentage of traffic for variant (0.0 to 1.0)
    model_a_path: str     # Path to control model
    model_b_path: str     # Path to variant model
    success_metric: str   # Primary metric to optimize
    minimum_sample_size: int = 100
    significance_level: float = 0.05
    power: float = 0.8
    expected_effect_size: float = 0.05

@dataclass
class ABTestResult:
    """Results of an A/B test"""
    test_name: str
    variant_a_metric: float
    variant_b_metric: float
    sample_size_a: int
    sample_size_b: int
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    winner: str
    test_power: float
    recommendation: str

class ABTestManager:
    """Manages A/B tests for model performance"""
    
    def __init__(self, db_path: str = "ab_testing.db"):
        self.db_path = db_path
        self.performance_db = "model_performance.db"
        self.init_database()
        self.setup_logging()
    
    def init_database(self):
        """Initialize A/B testing database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # A/B test configurations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ab_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT UNIQUE NOT NULL,
                description TEXT,
                config TEXT NOT NULL,
                status TEXT DEFAULT 'DRAFT',
                created_at TEXT NOT NULL,
                started_at TEXT,
                ended_at TEXT
            )
        ''')
        
        # A/B test results for each prediction
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ab_test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                player_id INTEGER NOT NULL,
                gameweek INTEGER NOT NULL,
                variant TEXT NOT NULL,
                predicted_points REAL NOT NULL,
                actual_points REAL,
                absolute_error REAL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (test_name) REFERENCES ab_tests (test_name)
            )
        ''')
        
        # A/B test analysis results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ab_test_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                analysis_date TEXT NOT NULL,
                variant_a_metric REAL,
                variant_b_metric REAL,
                sample_size_a INTEGER,
                sample_size_b INTEGER,
                p_value REAL,
                effect_size REAL,
                is_significant BOOLEAN,
                confidence_interval_lower REAL,
                confidence_interval_upper REAL,
                test_power REAL,
                winner TEXT,
                recommendation TEXT,
                FOREIGN KEY (test_name) REFERENCES ab_tests (test_name)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def setup_logging(self):
        """Setup logging for A/B testing"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/ab_testing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ABTestManager')
    
    def create_test(self, config: ABTestConfig) -> bool:
        """Create a new A/B test"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Validate models exist
            if not os.path.exists(config.model_a_path):
                raise FileNotFoundError(f"Model A not found: {config.model_a_path}")
            if not os.path.exists(config.model_b_path):
                raise FileNotFoundError(f"Model B not found: {config.model_b_path}")
            
            # Insert test configuration
            cursor.execute('''
                INSERT INTO ab_tests (test_name, description, config, created_at)
                VALUES (?, ?, ?, ?)
            ''', (
                config.test_name,
                config.description,
                json.dumps(asdict(config), default=str),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Created A/B test: {config.test_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create A/B test: {e}")
            return False
    
    def start_test(self, test_name: str) -> bool:
        """Start an A/B test"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update test status
            cursor.execute('''
                UPDATE ab_tests 
                SET status = 'RUNNING', started_at = ?
                WHERE test_name = ?
            ''', (datetime.now().isoformat(), test_name))
            
            if cursor.rowcount == 0:
                raise ValueError(f"Test not found: {test_name}")
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Started A/B test: {test_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start A/B test: {e}")
            return False
    
    def stop_test(self, test_name: str) -> bool:
        """Stop an A/B test"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update test status
            cursor.execute('''
                UPDATE ab_tests 
                SET status = 'COMPLETED', ended_at = ?
                WHERE test_name = ?
            ''', (datetime.now().isoformat(), test_name))
            
            if cursor.rowcount == 0:
                raise ValueError(f"Test not found: {test_name}")
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Stopped A/B test: {test_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop A/B test: {e}")
            return False
    
    def get_active_tests(self) -> List[Dict[str, Any]]:
        """Get all active A/B tests"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT test_name, description, config, started_at
            FROM ab_tests 
            WHERE status = 'RUNNING'
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        tests = []
        for _, row in df.iterrows():
            config = json.loads(row['config'])
            tests.append({
                'test_name': row['test_name'],
                'description': row['description'],
                'config': config,
                'started_at': row['started_at']
            })
        
        return tests
    
    def should_use_variant(self, test_name: str, user_id: str = None) -> str:
        """Determine which variant to use for this request"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get test configuration
            cursor.execute('''
                SELECT config FROM ab_tests 
                WHERE test_name = ? AND status = 'RUNNING'
            ''', (test_name,))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return 'A'  # Default to control if test not found
            
            config_data = json.loads(result[0])
            traffic_split = config_data.get('traffic_split', 0.5)
            
            # Use deterministic assignment based on user_id or random for anonymous
            if user_id:
                # Hash user_id to get consistent assignment
                import hashlib
                hash_value = int(hashlib.md5(f"{test_name}_{user_id}".encode()).hexdigest(), 16)
                random_value = (hash_value % 100) / 100.0
            else:
                # Random assignment for anonymous requests
                random_value = np.random.random()
            
            return 'B' if random_value < traffic_split else 'A'
            
        except Exception as e:
            self.logger.error(f"Error determining variant: {e}")
            return 'A'  # Default to control on error
    
    def log_prediction(self, test_name: str, variant: str, player_id: int, 
                      gameweek: int, predicted_points: float, 
                      actual_points: float = None):
        """Log a prediction result for A/B testing"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            absolute_error = None
            if actual_points is not None:
                absolute_error = abs(predicted_points - actual_points)
            
            cursor.execute('''
                INSERT INTO ab_test_results 
                (test_name, player_id, gameweek, variant, predicted_points, 
                 actual_points, absolute_error, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                test_name, player_id, gameweek, variant, predicted_points,
                actual_points, absolute_error, datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to log prediction: {e}")
    
    def analyze_test(self, test_name: str) -> Optional[ABTestResult]:
        """Analyze results of an A/B test"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get test configuration
            config_query = '''
                SELECT config FROM ab_tests WHERE test_name = ?
            '''
            config_result = pd.read_sql_query(config_query, conn, params=(test_name,))
            
            if config_result.empty:
                self.logger.error(f"Test not found: {test_name}")
                return None
            
            config_data = json.loads(config_result.iloc[0]['config'])
            success_metric = config_data.get('success_metric', 'absolute_error')
            
            # Get test results
            results_query = '''
                SELECT variant, predicted_points, actual_points, absolute_error
                FROM ab_test_results 
                WHERE test_name = ? AND actual_points IS NOT NULL
            '''
            
            results_df = pd.read_sql_query(results_query, conn, params=(test_name,))
            conn.close()
            
            if results_df.empty:
                self.logger.warning(f"No results found for test: {test_name}")
                return None
            
            # Split by variant
            variant_a = results_df[results_df['variant'] == 'A']
            variant_b = results_df[results_df['variant'] == 'B']
            
            if len(variant_a) == 0 or len(variant_b) == 0:
                self.logger.warning(f"Insufficient data for both variants in test: {test_name}")
                return None
            
            # Calculate metrics based on success metric
            if success_metric == 'absolute_error':
                metric_a = variant_a['absolute_error'].mean()
                metric_b = variant_b['absolute_error'].mean()
                # For error metrics, lower is better
                is_b_better = metric_b < metric_a
            elif success_metric == 'accuracy':
                # Calculate accuracy (predictions within threshold)
                threshold = 1.0  # Points threshold for accuracy
                metric_a = (variant_a['absolute_error'] <= threshold).mean() * 100
                metric_b = (variant_b['absolute_error'] <= threshold).mean() * 100
                # For accuracy, higher is better
                is_b_better = metric_b > metric_a
            else:
                # Default to absolute error
                metric_a = variant_a['absolute_error'].mean()
                metric_b = variant_b['absolute_error'].mean()
                is_b_better = metric_b < metric_a
            
            # Statistical test
            if success_metric == 'absolute_error':
                # Use Mann-Whitney U test for error distributions
                statistic, p_value = stats.mannwhitneyu(
                    variant_a['absolute_error'], 
                    variant_b['absolute_error'],
                    alternative='two-sided'
                )
            else:
                # Use t-test for other metrics
                statistic, p_value = stats.ttest_ind(
                    variant_a['absolute_error'], 
                    variant_b['absolute_error']
                )
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(
                ((len(variant_a) - 1) * variant_a['absolute_error'].var() +
                 (len(variant_b) - 1) * variant_b['absolute_error'].var()) /
                (len(variant_a) + len(variant_b) - 2)
            )
            
            if pooled_std > 0:
                effect_size = abs(metric_b - metric_a) / pooled_std
            else:
                effect_size = 0.0
            
            # Confidence interval for difference
            se_diff = np.sqrt(
                variant_a['absolute_error'].var() / len(variant_a) +
                variant_b['absolute_error'].var() / len(variant_b)
            )
            
            diff = metric_b - metric_a
            margin_error = stats.t.ppf(0.975, len(variant_a) + len(variant_b) - 2) * se_diff
            ci_lower = diff - margin_error
            ci_upper = diff + margin_error
            
            # Statistical power calculation
            test_power = self.calculate_statistical_power(
                len(variant_a), len(variant_b), effect_size, 0.05
            )
            
            # Determine significance and winner
            significance_level = config_data.get('significance_level', 0.05)
            is_significant = p_value < significance_level
            
            if is_significant:
                winner = 'B' if is_b_better else 'A'
                recommendation = f"Variant {winner} is significantly better. Deploy variant {winner}."
            else:
                winner = 'INCONCLUSIVE'
                recommendation = "No significant difference found. Consider running test longer or checking for implementation issues."
            
            result = ABTestResult(
                test_name=test_name,
                variant_a_metric=metric_a,
                variant_b_metric=metric_b,
                sample_size_a=len(variant_a),
                sample_size_b=len(variant_b),
                p_value=p_value,
                effect_size=effect_size,
                confidence_interval=(ci_lower, ci_upper),
                is_significant=is_significant,
                winner=winner,
                test_power=test_power,
                recommendation=recommendation
            )
            
            # Save analysis to database
            self.save_analysis(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to analyze test: {e}")
            return None
    
    def calculate_statistical_power(self, n1: int, n2: int, effect_size: float, 
                                   alpha: float = 0.05) -> float:
        """Calculate statistical power of the test"""
        try:
            from scipy.stats import norm
            
            # Calculate pooled standard error
            pooled_se = np.sqrt(1/n1 + 1/n2)
            
            # Critical value for two-tailed test
            z_alpha = norm.ppf(1 - alpha/2)
            
            # Calculate power
            z_beta = effect_size / pooled_se - z_alpha
            power = norm.cdf(z_beta)
            
            return max(0.0, min(1.0, power))
            
        except Exception:
            return 0.0
    
    def save_analysis(self, result: ABTestResult):
        """Save analysis results to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ab_test_analysis 
                (test_name, analysis_date, variant_a_metric, variant_b_metric,
                 sample_size_a, sample_size_b, p_value, effect_size,
                 is_significant, confidence_interval_lower, confidence_interval_upper,
                 test_power, winner, recommendation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.test_name,
                datetime.now().isoformat(),
                result.variant_a_metric,
                result.variant_b_metric,
                result.sample_size_a,
                result.sample_size_b,
                result.p_value,
                result.effect_size,
                result.is_significant,
                result.confidence_interval[0],
                result.confidence_interval[1],
                result.test_power,
                result.winner,
                result.recommendation
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to save analysis: {e}")
    
    def get_test_summary(self, test_name: str) -> Dict[str, Any]:
        """Get a summary of test results"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get test info
            test_query = '''
                SELECT * FROM ab_tests WHERE test_name = ?
            '''
            test_info = pd.read_sql_query(test_query, conn, params=(test_name,))
            
            # Get latest analysis
            analysis_query = '''
                SELECT * FROM ab_test_analysis 
                WHERE test_name = ? 
                ORDER BY analysis_date DESC LIMIT 1
            '''
            analysis_info = pd.read_sql_query(analysis_query, conn, params=(test_name,))
            
            # Get result counts
            results_query = '''
                SELECT variant, COUNT(*) as count,
                       AVG(absolute_error) as avg_error,
                       COUNT(CASE WHEN actual_points IS NOT NULL THEN 1 END) as complete_count
                FROM ab_test_results 
                WHERE test_name = ?
                GROUP BY variant
            '''
            results_summary = pd.read_sql_query(results_query, conn, params=(test_name,))
            
            conn.close()
            
            summary = {
                'test_name': test_name,
                'status': test_info.iloc[0]['status'] if not test_info.empty else 'NOT_FOUND',
                'description': test_info.iloc[0]['description'] if not test_info.empty else '',
                'created_at': test_info.iloc[0]['created_at'] if not test_info.empty else '',
                'started_at': test_info.iloc[0]['started_at'] if not test_info.empty else None,
                'ended_at': test_info.iloc[0]['ended_at'] if not test_info.empty else None,
                'results_summary': results_summary.to_dict('records') if not results_summary.empty else [],
                'latest_analysis': analysis_info.to_dict('records')[0] if not analysis_info.empty else None
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get test summary: {e}")
            return {'test_name': test_name, 'status': 'ERROR', 'error': str(e)}
    
    def list_all_tests(self) -> pd.DataFrame:
        """List all A/B tests"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT t.test_name, t.description, t.status, t.created_at, t.started_at, t.ended_at,
                       a.winner, a.is_significant, a.test_power
                FROM ab_tests t
                LEFT JOIN (
                    SELECT DISTINCT test_name, winner, is_significant, test_power
                    FROM ab_test_analysis a1
                    WHERE analysis_date = (
                        SELECT MAX(analysis_date) 
                        FROM ab_test_analysis a2 
                        WHERE a2.test_name = a1.test_name
                    )
                ) a ON t.test_name = a.test_name
                ORDER BY t.created_at DESC
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to list tests: {e}")
            return pd.DataFrame()

def create_sample_ab_test():
    """Create a sample A/B test for demonstration"""
    manager = ABTestManager()
    
    # Create test configuration
    config = ABTestConfig(
        test_name="model_comparison_v1",
        description="Compare current model vs new feature engineering approach",
        start_date=datetime.now(),
        end_date=datetime.now() + timedelta(days=14),
        traffic_split=0.5,
        model_a_path="models/fpl_points_model.joblib",
        model_b_path="models/fpl_points_model_v2.joblib",
        success_metric="absolute_error",
        minimum_sample_size=100,
        significance_level=0.05
    )
    
    # Create the test
    success = manager.create_test(config)
    
    if success:
        print(f"✅ Created A/B test: {config.test_name}")
        print(f"📝 Description: {config.description}")
        print(f"🎯 Success metric: {config.success_metric}")
        print(f"🔄 Traffic split: {config.traffic_split * 100}% to variant B")
    else:
        print("❌ Failed to create A/B test")

def main():
    """Main entry point for A/B testing"""
    create_sample_ab_test()

if __name__ == "__main__":
    main()