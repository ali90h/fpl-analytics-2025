"""
Feature Monitoring System for FPL Model
Comprehensive monitoring of feature drift, data quality, and statistical changes
"""

import numpy as np
import pandas as pd
import sqlite3
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FeatureDriftConfig:
    """Configuration for feature drift detection"""
    # Statistical thresholds
    ks_test_threshold: float = 0.05
    psi_threshold: float = 0.1
    chi2_threshold: float = 0.05
    correlation_threshold: float = 0.1
    
    # Data quality thresholds
    missing_data_threshold: float = 0.05  # 5% missing data threshold
    outlier_threshold: float = 0.02       # 2% outlier threshold
    zero_variance_threshold: float = 0.01 # 1% zero variance threshold
    
    # Monitoring settings
    reference_window_days: int = 30
    comparison_window_days: int = 7
    min_samples: int = 100
    
    # Alert settings
    enable_drift_alerts: bool = True
    enable_quality_alerts: bool = True
    alert_cooldown_hours: int = 24

@dataclass
class FeatureStats:
    """Statistics for a single feature"""
    feature_name: str
    data_type: str
    
    # Basic statistics
    count: int
    missing_count: int
    missing_percentage: float
    
    # Numerical statistics (if applicable)
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    q25: Optional[float] = None
    q50: Optional[float] = None
    q75: Optional[float] = None
    
    # Categorical statistics (if applicable)
    unique_count: Optional[int] = None
    mode: Optional[str] = None
    mode_frequency: Optional[float] = None
    
    # Quality indicators
    outlier_count: int = 0
    outlier_percentage: float = 0.0
    zero_variance: bool = False

@dataclass
class DriftResult:
    """Result of drift detection for a feature"""
    feature_name: str
    drift_detected: bool
    drift_score: float
    drift_type: str  # 'statistical', 'distribution', 'quality'
    
    # Test results
    ks_statistic: Optional[float] = None
    ks_p_value: Optional[float] = None
    psi_score: Optional[float] = None
    chi2_statistic: Optional[float] = None
    chi2_p_value: Optional[float] = None
    
    # Change metrics
    mean_change: Optional[float] = None
    std_change: Optional[float] = None
    correlation_change: Optional[float] = None
    
    # Quality changes
    missing_change: float = 0.0
    outlier_change: float = 0.0
    
    severity: str = 'LOW'  # LOW, MEDIUM, HIGH, CRITICAL
    recommendation: str = ''

class FeatureMonitor:
    """Comprehensive feature monitoring and drift detection system"""
    
    def __init__(self, db_path: str = "feature_monitoring.db", config: FeatureDriftConfig = None):
        self.db_path = db_path
        self.config = config or FeatureDriftConfig()
        self.init_database()
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for feature monitoring"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/feature_monitoring.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('FeatureMonitor')
    
    def init_database(self):
        """Initialize feature monitoring database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Feature statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                gameweek INTEGER,
                feature_name TEXT NOT NULL,
                data_type TEXT NOT NULL,
                count INTEGER,
                missing_count INTEGER,
                missing_percentage REAL,
                mean REAL,
                std REAL,
                min_val REAL,
                max_val REAL,
                q25 REAL,
                q50 REAL,
                q75 REAL,
                unique_count INTEGER,
                mode TEXT,
                mode_frequency REAL,
                outlier_count INTEGER,
                outlier_percentage REAL,
                zero_variance BOOLEAN,
                additional_stats TEXT
            )
        ''')
        
        # Feature drift results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_drift (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                drift_detected BOOLEAN NOT NULL,
                has_drift BOOLEAN NOT NULL,
                drift_score REAL NOT NULL,
                drift_type TEXT NOT NULL,
                ks_statistic REAL,
                ks_p_value REAL,
                psi_score REAL,
                chi2_statistic REAL,
                chi2_p_value REAL,
                mean_change REAL,
                std_change REAL,
                correlation_change REAL,
                missing_change REAL,
                outlier_change REAL,
                severity TEXT NOT NULL,
                recommendation TEXT,
                reference_period TEXT,
                comparison_period TEXT
            )
        ''')
        
        # Data quality alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                feature_name TEXT,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                metric_value REAL,
                threshold REAL,
                resolved BOOLEAN DEFAULT FALSE,
                resolved_at TEXT
            )
        ''')
        
        # Feature correlation matrix table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_correlations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                feature1 TEXT NOT NULL,
                feature2 TEXT NOT NULL,
                correlation_value REAL NOT NULL,
                gameweek INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def calculate_feature_statistics(self, df: pd.DataFrame, gameweek: int = None) -> Dict[str, FeatureStats]:
        """Calculate comprehensive statistics for all features"""
        stats_dict = {}
        
        for column in df.columns:
            if column in ['player_id', 'name', 'team', 'position']:
                continue  # Skip metadata columns
            
            series = df[column]
            
            # Basic statistics
            count = len(series)
            missing_count = series.isnull().sum()
            missing_percentage = (missing_count / count) * 100 if count > 0 else 0
            
            # Determine data type
            if pd.api.types.is_numeric_dtype(series):
                data_type = 'numerical'
                
                # Remove missing values for calculations
                clean_series = series.dropna()
                
                if len(clean_series) > 0:
                    mean_val = float(clean_series.mean())
                    std_val = float(clean_series.std())
                    min_val = float(clean_series.min())
                    max_val = float(clean_series.max())
                    q25 = float(clean_series.quantile(0.25))
                    q50 = float(clean_series.quantile(0.50))
                    q75 = float(clean_series.quantile(0.75))
                    
                    # Detect outliers using IQR method
                    iqr = q75 - q25
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
                    outlier_count = len(outliers)
                    outlier_percentage = (outlier_count / count) * 100 if count > 0 else 0
                    
                    # Check for zero variance
                    zero_variance = std_val < 1e-10
                    
                    stats_dict[column] = FeatureStats(
                        feature_name=column,
                        data_type=data_type,
                        count=count,
                        missing_count=missing_count,
                        missing_percentage=missing_percentage,
                        mean=mean_val,
                        std=std_val,
                        min_val=min_val,
                        max_val=max_val,
                        q25=q25,
                        q50=q50,
                        q75=q75,
                        outlier_count=outlier_count,
                        outlier_percentage=outlier_percentage,
                        zero_variance=zero_variance
                    )
                else:
                    # Handle all-missing numerical column
                    stats_dict[column] = FeatureStats(
                        feature_name=column,
                        data_type=data_type,
                        count=count,
                        missing_count=missing_count,
                        missing_percentage=missing_percentage,
                        zero_variance=True
                    )
            
            else:
                data_type = 'categorical'
                
                # Categorical statistics
                unique_count = series.nunique()
                mode_val = series.mode().iloc[0] if len(series.mode()) > 0 else None
                mode_frequency = (series == mode_val).sum() / count if mode_val is not None and count > 0 else 0
                
                stats_dict[column] = FeatureStats(
                    feature_name=column,
                    data_type=data_type,
                    count=count,
                    missing_count=missing_count,
                    missing_percentage=missing_percentage,
                    unique_count=unique_count,
                    mode=str(mode_val) if mode_val is not None else None,
                    mode_frequency=mode_frequency
                )
        
        return stats_dict
    
    def save_feature_statistics(self, stats_dict: Dict[str, FeatureStats], gameweek: int = None):
        """Save feature statistics to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        for feature_name, stats in stats_dict.items():
            cursor.execute('''
                INSERT INTO feature_statistics (
                    timestamp, gameweek, feature_name, data_type, count, missing_count, 
                    missing_percentage, mean, std, min_val, max_val, q25, q50, q75,
                    unique_count, mode, mode_frequency, outlier_count, outlier_percentage,
                    zero_variance
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp, gameweek, stats.feature_name, stats.data_type,
                stats.count, stats.missing_count, stats.missing_percentage,
                stats.mean, stats.std, stats.min_val, stats.max_val,
                stats.q25, stats.q50, stats.q75, stats.unique_count,
                stats.mode, stats.mode_frequency, stats.outlier_count,
                stats.outlier_percentage, stats.zero_variance
            ))
        
        conn.commit()
        conn.close()
        self.logger.info(f"Saved statistics for {len(stats_dict)} features")
    
    def detect_feature_drift(self, current_df: pd.DataFrame, 
                           reference_df: pd.DataFrame = None) -> Dict[str, DriftResult]:
        """Detect drift between current and reference datasets"""
        if reference_df is None:
            reference_df = self.get_reference_data()
        
        if reference_df is None or reference_df.empty:
            self.logger.warning("No reference data available for drift detection")
            return {}
        
        drift_results = {}
        
        # Get common numerical features
        numerical_features = []
        for col in current_df.columns:
            if col in reference_df.columns and pd.api.types.is_numeric_dtype(current_df[col]):
                if col not in ['player_id', 'team', 'position']:
                    numerical_features.append(col)
        
        for feature in numerical_features:
            current_values = current_df[feature].dropna()
            reference_values = reference_df[feature].dropna()
            
            if len(current_values) < self.config.min_samples or len(reference_values) < self.config.min_samples:
                continue
            
            drift_result = self._analyze_feature_drift(feature, current_values, reference_values)
            drift_results[feature] = drift_result
        
        return drift_results
    
    def _analyze_feature_drift(self, feature_name: str, 
                              current_values: pd.Series, 
                              reference_values: pd.Series) -> DriftResult:
        """Analyze drift for a single feature"""
        
        # Kolmogorov-Smirnov test for distribution change
        ks_statistic, ks_p_value = ks_2samp(reference_values, current_values)
        
        # Population Stability Index (PSI)
        psi_score = self._calculate_psi(reference_values, current_values)
        
        # Mean and standard deviation changes
        mean_change = abs(current_values.mean() - reference_values.mean()) / (reference_values.std() + 1e-10)
        std_change = abs(current_values.std() - reference_values.std()) / (reference_values.std() + 1e-10)
        
        # Determine drift detection
        drift_detected = (
            ks_p_value < self.config.ks_test_threshold or
            psi_score > self.config.psi_threshold or
            mean_change > 2.0 or  # 2 standard deviations
            std_change > 0.5      # 50% change in variance
        )
        
        # Calculate overall drift score
        drift_score = max(
            1 - ks_p_value,
            psi_score,
            min(mean_change / 2.0, 1.0),
            min(std_change / 0.5, 1.0)
        )
        
        # Determine severity
        if drift_score > 0.8:
            severity = 'CRITICAL'
        elif drift_score > 0.6:
            severity = 'HIGH'
        elif drift_score > 0.4:
            severity = 'MEDIUM'
        else:
            severity = 'LOW'
        
        # Generate recommendation
        recommendation = self._generate_drift_recommendation(
            feature_name, drift_score, ks_p_value, psi_score, mean_change, std_change
        )
        
        return DriftResult(
            feature_name=feature_name,
            drift_detected=drift_detected,
            drift_score=drift_score,
            drift_type='statistical',
            ks_statistic=ks_statistic,
            ks_p_value=ks_p_value,
            psi_score=psi_score,
            mean_change=mean_change,
            std_change=std_change,
            severity=severity,
            recommendation=recommendation
        )
    
    def _calculate_psi(self, reference: pd.Series, current: pd.Series, buckets: int = 10) -> float:
        """Calculate Population Stability Index (PSI)"""
        try:
            # Create bins based on reference distribution
            _, bin_edges = np.histogram(reference, bins=buckets)
            
            # Ensure finite bins
            bin_edges = bin_edges[np.isfinite(bin_edges)]
            if len(bin_edges) < 2:
                return 0.0
            
            # Calculate distributions
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            curr_counts, _ = np.histogram(current, bins=bin_edges)
            
            # Convert to percentages
            ref_pct = ref_counts / len(reference)
            curr_pct = curr_counts / len(current)
            
            # Avoid division by zero
            ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
            curr_pct = np.where(curr_pct == 0, 0.0001, curr_pct)
            
            # Calculate PSI
            psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
            
            return float(psi)
        
        except Exception as e:
            self.logger.warning(f"PSI calculation failed for feature: {e}")
            return 0.0
    
    def _generate_drift_recommendation(self, feature_name: str, drift_score: float,
                                     ks_p_value: float, psi_score: float,
                                     mean_change: float, std_change: float) -> str:
        """Generate recommendation based on drift analysis"""
        
        recommendations = []
        
        if drift_score > 0.8:
            recommendations.append("CRITICAL: Immediate investigation required")
        
        if ks_p_value < 0.01:
            recommendations.append("Distribution has significantly changed")
        
        if psi_score > 0.25:
            recommendations.append("High population instability detected")
        
        if mean_change > 3.0:
            recommendations.append("Mean has shifted significantly")
        
        if std_change > 1.0:
            recommendations.append("Variance has changed dramatically")
        
        if drift_score > 0.6:
            recommendations.append("Consider model retraining")
        
        if not recommendations:
            recommendations.append("Feature appears stable")
        
        return "; ".join(recommendations)
    
    def get_reference_data(self) -> Optional[pd.DataFrame]:
        """Get reference dataset for drift comparison"""
        try:
            # Load historical feature data as reference
            reference_path = "data/performance_history/model_features.csv"
            if os.path.exists(reference_path):
                return pd.read_csv(reference_path)
            else:
                self.logger.warning(f"Reference data not found: {reference_path}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading reference data: {e}")
            return None
    
    def save_drift_results(self, drift_results: Dict[str, DriftResult]):
        """Save drift detection results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        reference_period = f"{self.config.reference_window_days} days"
        comparison_period = f"{self.config.comparison_window_days} days"
        
        for feature_name, result in drift_results.items():
            cursor.execute('''
                INSERT INTO feature_drift (
                    timestamp, feature_name, drift_detected, has_drift, drift_score, drift_type,
                    ks_statistic, ks_p_value, psi_score, chi2_statistic, chi2_p_value,
                    mean_change, std_change, correlation_change, missing_change, outlier_change,
                    severity, recommendation, reference_period, comparison_period
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp, result.feature_name, result.drift_detected, result.drift_detected, result.drift_score,
                result.drift_type, result.ks_statistic, result.ks_p_value, result.psi_score,
                result.chi2_statistic, result.chi2_p_value, result.mean_change,
                result.std_change, result.correlation_change, result.missing_change,
                result.outlier_change, result.severity, result.recommendation,
                reference_period, comparison_period
            ))
        
        conn.commit()
        conn.close()
        self.logger.info(f"Saved drift results for {len(drift_results)} features")
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality assessment"""
        quality_report = {
            'timestamp': datetime.now().isoformat(),
            'total_features': len(df.columns),
            'total_samples': len(df),
            'quality_issues': [],
            'quality_score': 1.0
        }
        
        issues = []
        
        # Check for missing data
        missing_data = df.isnull().sum()
        high_missing_features = missing_data[missing_data > len(df) * self.config.missing_data_threshold]
        
        for feature, missing_count in high_missing_features.items():
            missing_pct = (missing_count / len(df)) * 100
            issues.append({
                'type': 'HIGH_MISSING_DATA',
                'feature': feature,
                'value': missing_pct,
                'threshold': self.config.missing_data_threshold * 100,
                'severity': 'HIGH' if missing_pct > 20 else 'MEDIUM'
            })
        
        # Check for zero variance features
        numerical_features = df.select_dtypes(include=[np.number]).columns
        for feature in numerical_features:
            if df[feature].std() < 1e-10:
                issues.append({
                    'type': 'ZERO_VARIANCE',
                    'feature': feature,
                    'value': 0.0,
                    'threshold': self.config.zero_variance_threshold,
                    'severity': 'HIGH'
                })
        
        # Check for high outlier percentage
        for feature in numerical_features:
            if feature in ['player_id', 'team', 'position']:
                continue
            
            clean_series = df[feature].dropna()
            if len(clean_series) > 0:
                q1 = clean_series.quantile(0.25)
                q3 = clean_series.quantile(0.75)
                iqr = q3 - q1
                outliers = clean_series[(clean_series < q1 - 1.5 * iqr) | 
                                      (clean_series > q3 + 1.5 * iqr)]
                outlier_pct = len(outliers) / len(clean_series)
                
                if outlier_pct > self.config.outlier_threshold:
                    issues.append({
                        'type': 'HIGH_OUTLIERS',
                        'feature': feature,
                        'value': outlier_pct * 100,
                        'threshold': self.config.outlier_threshold * 100,
                        'severity': 'MEDIUM' if outlier_pct < 0.1 else 'HIGH'
                    })
        
        # Calculate overall quality score
        if issues:
            critical_issues = sum(1 for issue in issues if issue['severity'] == 'HIGH')
            medium_issues = sum(1 for issue in issues if issue['severity'] == 'MEDIUM')
            
            quality_score = max(0.0, 1.0 - (critical_issues * 0.2 + medium_issues * 0.1))
        else:
            quality_score = 1.0
        
        quality_report['quality_issues'] = issues
        quality_report['quality_score'] = quality_score
        
        return quality_report
    
    def save_quality_alerts(self, quality_report: Dict[str, Any]):
        """Save data quality alerts to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        for issue in quality_report['quality_issues']:
            cursor.execute('''
                INSERT INTO quality_alerts (
                    timestamp, alert_type, feature_name, severity, message,
                    metric_value, threshold
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp,
                issue['type'],
                issue['feature'],
                issue['severity'],
                f"{issue['type']} for {issue['feature']}: {issue['value']:.2f}%",
                issue['value'],
                issue['threshold']
            ))
        
        conn.commit()
        conn.close()
    
    def run_comprehensive_monitoring(self, df: pd.DataFrame, gameweek: int = None) -> Dict[str, Any]:
        """Run complete feature monitoring pipeline"""
        self.logger.info("🔍 Starting comprehensive feature monitoring...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'gameweek': gameweek,
            'monitoring_results': {}
        }
        
        try:
            # 1. Calculate feature statistics
            self.logger.info("📊 Calculating feature statistics...")
            feature_stats = self.calculate_feature_statistics(df, gameweek)
            self.save_feature_statistics(feature_stats, gameweek)
            results['monitoring_results']['feature_statistics'] = len(feature_stats)
            
            # 2. Detect feature drift
            self.logger.info("🔄 Detecting feature drift...")
            drift_results = self.detect_feature_drift(df)
            if drift_results:
                self.save_drift_results(drift_results)
                
                # Count drift detections by severity
                drift_summary = {}
                for result in drift_results.values():
                    severity = result.severity
                    drift_summary[severity] = drift_summary.get(severity, 0) + 1
                
                results['monitoring_results']['drift_detections'] = drift_summary
            else:
                results['monitoring_results']['drift_detections'] = {}
            
            # 3. Check data quality
            self.logger.info("🔍 Assessing data quality...")
            quality_report = self.check_data_quality(df)
            self.save_quality_alerts(quality_report)
            results['monitoring_results']['quality_score'] = quality_report['quality_score']
            results['monitoring_results']['quality_issues'] = len(quality_report['quality_issues'])
            
            # 4. Generate summary
            total_drift_detected = sum(1 for r in drift_results.values() if r.drift_detected)
            critical_drifts = sum(1 for r in drift_results.values() if r.severity == 'CRITICAL')
            
            results['summary'] = {
                'features_monitored': len(feature_stats),
                'drift_detected': total_drift_detected,
                'critical_drifts': critical_drifts,
                'overall_quality_score': quality_report['quality_score'],
                'recommendations': self._generate_monitoring_recommendations(
                    drift_results, quality_report
                )
            }
            
            self.logger.info(f"✅ Feature monitoring completed: {results['summary']}")
            
        except Exception as e:
            self.logger.error(f"❌ Error in comprehensive monitoring: {e}")
            results['error'] = str(e)
        
        return results
    
    def _generate_monitoring_recommendations(self, drift_results: Dict[str, DriftResult], 
                                           quality_report: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on monitoring results"""
        recommendations = []
        
        # Drift-based recommendations
        critical_drifts = [r for r in drift_results.values() if r.severity == 'CRITICAL']
        high_drifts = [r for r in drift_results.values() if r.severity == 'HIGH']
        
        if critical_drifts:
            recommendations.append(f"🚨 URGENT: {len(critical_drifts)} features show critical drift - immediate model retraining recommended")
        
        if high_drifts:
            recommendations.append(f"⚠️ {len(high_drifts)} features show high drift - consider model retraining within 1-2 weeks")
        
        # Quality-based recommendations
        if quality_report['quality_score'] < 0.7:
            recommendations.append(f"📉 Low data quality score ({quality_report['quality_score']:.2f}) - data validation needed")
        
        high_severity_issues = [i for i in quality_report['quality_issues'] if i['severity'] == 'HIGH']
        if high_severity_issues:
            recommendations.append(f"🔍 {len(high_severity_issues)} high-severity data quality issues require attention")
        
        # General recommendations
        if not recommendations:
            recommendations.append("✅ Feature monitoring shows stable performance - continue regular monitoring")
        else:
            recommendations.append("📋 Review detailed monitoring logs for specific feature-level insights")
        
        return recommendations
    
    def get_monitoring_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get summary of recent monitoring activity"""
        conn = sqlite3.connect(self.db_path)
        
        # Get recent drift detections
        drift_query = '''
            SELECT feature_name, drift_detected, severity, timestamp
            FROM feature_drift 
            WHERE timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days)
        
        drift_df = pd.read_sql_query(drift_query, conn)
        
        # Get recent quality alerts
        quality_query = '''
            SELECT alert_type, feature_name, severity, timestamp
            FROM quality_alerts 
            WHERE timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days)
        
        quality_df = pd.read_sql_query(quality_query, conn)
        conn.close()
        
        summary = {
            'period_days': days,
            'drift_summary': {
                'total_features_checked': len(drift_df['feature_name'].unique()) if not drift_df.empty else 0,
                'features_with_drift': len(drift_df[drift_df['drift_detected'] == True]['feature_name'].unique()) if not drift_df.empty else 0,
                'critical_drifts': len(drift_df[drift_df['severity'] == 'CRITICAL']) if not drift_df.empty else 0,
                'high_drifts': len(drift_df[drift_df['severity'] == 'HIGH']) if not drift_df.empty else 0
            },
            'quality_summary': {
                'total_alerts': len(quality_df) if not quality_df.empty else 0,
                'high_severity_alerts': len(quality_df[quality_df['severity'] == 'HIGH']) if not quality_df.empty else 0,
                'unique_features_with_issues': len(quality_df['feature_name'].unique()) if not quality_df.empty else 0
            }
        }
        
        return summary

def main():
    """Main entry point for feature monitoring"""
    monitor = FeatureMonitor()
    
    # Load current feature data
    try:
        current_data_path = "data/performance_history/model_features.csv"
        if os.path.exists(current_data_path):
            df = pd.read_csv(current_data_path)
            results = monitor.run_comprehensive_monitoring(df, gameweek=4)
            
            print("🔍 Feature Monitoring Results:")
            print(f"📊 Features monitored: {results['summary']['features_monitored']}")
            print(f"🔄 Drift detected: {results['summary']['drift_detected']}")
            print(f"🚨 Critical drifts: {results['summary']['critical_drifts']}")
            print(f"📈 Quality score: {results['summary']['overall_quality_score']:.3f}")
            
            print("\n💡 Recommendations:")
            for rec in results['summary']['recommendations']:
                print(f"   • {rec}")
        else:
            print("❌ Feature data not found")
    
    except Exception as e:
        print(f"❌ Error in feature monitoring: {e}")

if __name__ == "__main__":
    main()