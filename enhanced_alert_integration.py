"""
Enhanced Alert Integration for Feature Monitoring and Data Quality
Integrates feature drift and data quality alerts with the main alert system
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
import pandas as pd
from feature_monitoring import FeatureMonitor
from data_quality_monitor import DataQualityMonitor
from alert_system import AlertManager

@dataclass
class FeatureAlertConfig:
    """Configuration for feature monitoring alerts"""
    drift_threshold: float = 0.05          # KS test p-value threshold
    psi_threshold: float = 0.1             # PSI threshold
    missing_data_threshold: float = 5.0    # Missing data percentage threshold
    outlier_threshold: float = 10.0        # Outlier percentage threshold
    quality_score_threshold: float = 0.7   # Minimum quality score
    consecutive_drift_threshold: int = 3   # Consecutive drift detections before alert
    drift_check_interval_hours: int = 12   # How often to check for drift
    quality_check_interval_hours: int = 6  # How often to check data quality

class FeatureAlertManager:
    """Enhanced alert manager for feature monitoring and data quality"""
    
    def __init__(self, config_file: str = "feature_alert_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        self.feature_monitor = FeatureMonitor()
        self.quality_monitor = DataQualityMonitor()
        self.alert_manager = AlertManager()
        self.db_path = "feature_alerts.db"
        self.setup_logging()
        self.init_database()
    
    def load_config(self) -> FeatureAlertConfig:
        """Load feature alert configuration"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                return FeatureAlertConfig(**config_data)
            except Exception as e:
                logging.warning(f"Failed to load feature alert config: {e}. Using defaults.")
        
        # Create default config
        default_config = FeatureAlertConfig()
        self.save_config(default_config)
        return default_config
    
    def save_config(self, config: FeatureAlertConfig):
        """Save feature alert configuration"""
        config_dict = {
            'drift_threshold': config.drift_threshold,
            'psi_threshold': config.psi_threshold,
            'missing_data_threshold': config.missing_data_threshold,
            'outlier_threshold': config.outlier_threshold,
            'quality_score_threshold': config.quality_score_threshold,
            'consecutive_drift_threshold': config.consecutive_drift_threshold,
            'drift_check_interval_hours': config.drift_check_interval_hours,
            'quality_check_interval_hours': config.quality_check_interval_hours
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def setup_logging(self):
        """Setup logging for feature alerts"""
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/feature_alerts.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('FeatureAlertManager')
    
    def init_database(self):
        """Initialize feature alerts database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Feature alert history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                feature_name TEXT,
                dataset_name TEXT,
                message TEXT NOT NULL,
                alert_data TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                resolution_timestamp TEXT,
                resolution_notes TEXT
            )
        ''')
        
        # Alert suppression rules
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_suppression (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT NOT NULL,
                feature_name TEXT,
                dataset_name TEXT,
                suppressed_until TEXT NOT NULL,
                reason TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        
        # Feature alert stats
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                count INTEGER NOT NULL,
                avg_severity REAL,
                resolution_rate REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def check_feature_drift_alerts(self) -> List[Dict[str, Any]]:
        """Check for feature drift alerts"""
        alerts = []
        
        try:
            # Get recent drift results
            conn = sqlite3.connect(self.feature_monitor.db_path)
            
            # Check for features with consecutive drift detections
            query = '''
                SELECT feature_name, COUNT(*) as consecutive_drifts,
                       AVG(drift_score) as avg_drift_score,
                       AVG(psi_score) as avg_psi_score,
                       MAX(timestamp) as latest_timestamp
                FROM feature_drift 
                WHERE has_drift = 1 
                AND timestamp >= datetime('now', '-{} hours')
                GROUP BY feature_name
                HAVING consecutive_drifts >= ?
            '''.format(self.config.drift_check_interval_hours * 2)
            
            df = pd.read_sql_query(query, conn, params=(self.config.consecutive_drift_threshold,))
            conn.close()
            
            for _, row in df.iterrows():
                feature_name = row['feature_name']
                consecutive_drifts = row['consecutive_drifts']
                avg_drift_score = row['avg_drift_score']
                avg_psi_score = row['avg_psi_score']
                
                # Check if alert is suppressed
                if self.is_alert_suppressed('FEATURE_DRIFT', feature_name):
                    continue
                
                # Determine severity based on drift scores and consecutive occurrences
                if consecutive_drifts >= self.config.consecutive_drift_threshold * 2:
                    severity = 'CRITICAL'
                elif avg_drift_score > self.config.drift_threshold * 2:
                    severity = 'HIGH'
                else:
                    severity = 'MEDIUM'
                
                alert = {
                    'type': 'FEATURE_DRIFT',
                    'severity': severity,
                    'feature_name': feature_name,
                    'message': f'Feature "{feature_name}" shows persistent drift ({consecutive_drifts} consecutive detections, avg drift: {avg_drift_score:.4f})',
                    'alert_data': {
                        'consecutive_drifts': consecutive_drifts,
                        'avg_drift_score': avg_drift_score,
                        'avg_psi_score': avg_psi_score,
                        'drift_threshold': self.config.drift_threshold,
                        'psi_threshold': self.config.psi_threshold
                    },
                    'timestamp': datetime.now()
                }
                
                alerts.append(alert)
                self.logger.warning(f"Drift alert for feature {feature_name}: {consecutive_drifts} consecutive drifts")
        
        except Exception as e:
            self.logger.error(f"Error checking feature drift alerts: {e}")
        
        return alerts
    
    def check_data_quality_alerts(self) -> List[Dict[str, Any]]:
        """Check for data quality alerts"""
        alerts = []
        
        try:
            # Get latest quality metrics
            conn = sqlite3.connect(self.quality_monitor.db_path)
            
            query = '''
                SELECT * FROM quality_metrics 
                WHERE timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp DESC
                LIMIT 5
            '''.format(self.config.quality_check_interval_hours)
            
            df = pd.read_sql_query(query, conn)
            
            if not df.empty:
                latest_metrics = df.iloc[0]
                
                # Check quality score threshold
                if latest_metrics['quality_score'] < self.config.quality_score_threshold:
                    if not self.is_alert_suppressed('DATA_QUALITY_SCORE', latest_metrics['dataset_name']):
                        severity = 'CRITICAL' if latest_metrics['quality_score'] < 0.5 else 'HIGH'
                        
                        alert = {
                            'type': 'DATA_QUALITY_SCORE',
                            'severity': severity,
                            'dataset_name': latest_metrics['dataset_name'],
                            'message': f'Data quality score ({latest_metrics["quality_score"]:.3f}) below threshold ({self.config.quality_score_threshold:.3f}) for dataset "{latest_metrics["dataset_name"]}"',
                            'alert_data': {
                                'quality_score': latest_metrics['quality_score'],
                                'threshold': self.config.quality_score_threshold,
                                'quality_grade': latest_metrics['quality_grade'],
                                'missing_percentage': latest_metrics['missing_values_percentage'],
                                'schema_violations': latest_metrics['schema_violations']
                            },
                            'timestamp': datetime.now()
                        }
                        alerts.append(alert)
                
                # Check missing data threshold
                if latest_metrics['missing_values_percentage'] > self.config.missing_data_threshold:
                    if not self.is_alert_suppressed('HIGH_MISSING_DATA', latest_metrics['dataset_name']):
                        severity = 'HIGH' if latest_metrics['missing_values_percentage'] > 20 else 'MEDIUM'
                        
                        alert = {
                            'type': 'HIGH_MISSING_DATA',
                            'severity': severity,
                            'dataset_name': latest_metrics['dataset_name'],
                            'message': f'High missing data percentage ({latest_metrics["missing_values_percentage"]:.1f}%) in dataset "{latest_metrics["dataset_name"]}"',
                            'alert_data': {
                                'missing_percentage': latest_metrics['missing_values_percentage'],
                                'threshold': self.config.missing_data_threshold,
                                'total_rows': latest_metrics['total_rows'],
                                'complete_rows': latest_metrics['complete_rows_count']
                            },
                            'timestamp': datetime.now()
                        }
                        alerts.append(alert)
                
                # Check schema violations
                if latest_metrics['schema_violations'] > 0:
                    if not self.is_alert_suppressed('SCHEMA_VIOLATIONS', latest_metrics['dataset_name']):
                        severity = 'HIGH' if latest_metrics['schema_violations'] > 10 else 'MEDIUM'
                        
                        alert = {
                            'type': 'SCHEMA_VIOLATIONS',
                            'severity': severity,
                            'dataset_name': latest_metrics['dataset_name'],
                            'message': f'Schema violations detected ({latest_metrics["schema_violations"]} violations) in dataset "{latest_metrics["dataset_name"]}"',
                            'alert_data': {
                                'schema_violations': latest_metrics['schema_violations'],
                                'data_type_errors': latest_metrics['data_type_errors'],
                                'range_violations': latest_metrics['range_violations'],
                                'pattern_violations': latest_metrics['pattern_violations']
                            },
                            'timestamp': datetime.now()
                        }
                        alerts.append(alert)
                
                # Check outlier percentage
                if latest_metrics['outliers_percentage'] > self.config.outlier_threshold:
                    if not self.is_alert_suppressed('HIGH_OUTLIERS', latest_metrics['dataset_name']):
                        severity = 'MEDIUM'
                        
                        alert = {
                            'type': 'HIGH_OUTLIERS',
                            'severity': severity,
                            'dataset_name': latest_metrics['dataset_name'],
                            'message': f'High outlier percentage ({latest_metrics["outliers_percentage"]:.1f}%) in dataset "{latest_metrics["dataset_name"]}"',
                            'alert_data': {
                                'outliers_percentage': latest_metrics['outliers_percentage'],
                                'threshold': self.config.outlier_threshold,
                                'outliers_count': latest_metrics['outliers_count']
                            },
                            'timestamp': datetime.now()
                        }
                        alerts.append(alert)
            
            conn.close()
        
        except Exception as e:
            self.logger.error(f"Error checking data quality alerts: {e}")
        
        return alerts
    
    def check_feature_correlation_alerts(self) -> List[Dict[str, Any]]:
        """Check for unexpected feature correlation changes"""
        alerts = []
        
        try:
            # Get correlation matrix changes
            conn = sqlite3.connect(self.feature_monitor.db_path)
            
            query = '''
                SELECT feature1, feature2, correlation_value, timestamp
                FROM feature_correlations 
                WHERE timestamp >= datetime('now', '-{} hours')
                AND ABS(correlation_value) > 0.8
                ORDER BY ABS(correlation_value) DESC
                LIMIT 10
            '''.format(self.config.drift_check_interval_hours)
            
            df = pd.read_sql_query(query, conn)
            
            if not df.empty:
                high_correlations = df[df['correlation_value'].abs() > 0.9]
                
                if len(high_correlations) > 0:
                    if not self.is_alert_suppressed('HIGH_FEATURE_CORRELATION'):
                        features_list = [f"{row['feature1']}-{row['feature2']}" for _, row in high_correlations.iterrows()]
                        
                        alert = {
                            'type': 'HIGH_FEATURE_CORRELATION',
                            'severity': 'MEDIUM',
                            'message': f'High feature correlations detected: {", ".join(features_list[:3])}{"..." if len(features_list) > 3 else ""}',
                            'alert_data': {
                                'high_correlation_pairs': len(high_correlations),
                                'max_correlation': high_correlations['correlation_value'].abs().max(),
                                'feature_pairs': features_list
                            },
                            'timestamp': datetime.now()
                        }
                        alerts.append(alert)
            
            conn.close()
        
        except Exception as e:
            self.logger.error(f"Error checking correlation alerts: {e}")
        
        return alerts
    
    def is_alert_suppressed(self, alert_type: str, feature_name: str = None, dataset_name: str = None) -> bool:
        """Check if an alert is currently suppressed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT COUNT(*) FROM alert_suppression 
            WHERE alert_type = ? 
            AND (feature_name = ? OR feature_name IS NULL)
            AND (dataset_name = ? OR dataset_name IS NULL)
            AND suppressed_until > datetime('now')
        '''
        
        cursor.execute(query, (alert_type, feature_name, dataset_name))
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
    
    def suppress_alert(self, alert_type: str, hours: int, reason: str, 
                      feature_name: str = None, dataset_name: str = None):
        """Suppress an alert type for a specified duration"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        suppressed_until = datetime.now() + timedelta(hours=hours)
        
        cursor.execute('''
            INSERT INTO alert_suppression 
            (alert_type, feature_name, dataset_name, suppressed_until, reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            alert_type, feature_name, dataset_name,
            suppressed_until.isoformat(), reason, datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Suppressed alert {alert_type} for {hours} hours: {reason}")
    
    def save_feature_alert(self, alert: Dict[str, Any]):
        """Save feature alert to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert numpy types to Python types for JSON serialization
        alert_data = alert.get('alert_data', {})
        serializable_data = {}
        for key, value in alert_data.items():
            if hasattr(value, 'item'):  # numpy types
                serializable_data[key] = value.item()
            else:
                serializable_data[key] = value
        
        cursor.execute('''
            INSERT INTO feature_alerts 
            (timestamp, alert_type, severity, feature_name, dataset_name, message, alert_data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert['timestamp'].isoformat(),
            alert['type'],
            alert['severity'],
            alert.get('feature_name'),
            alert.get('dataset_name'),
            alert['message'],
            json.dumps(serializable_data)
        ))
        
        conn.commit()
        conn.close()
    
    def format_alert_for_main_system(self, feature_alert: Dict[str, Any]) -> Dict[str, Any]:
        """Format feature alert for the main alert system"""
        
        # Map feature alert types to main system types
        type_mapping = {
            'FEATURE_DRIFT': 'FEATURE_DRIFT',
            'DATA_QUALITY_SCORE': 'DATA_QUALITY',
            'HIGH_MISSING_DATA': 'DATA_COMPLETENESS',
            'SCHEMA_VIOLATIONS': 'SCHEMA_COMPLIANCE',
            'HIGH_OUTLIERS': 'DATA_ANOMALY',
            'HIGH_FEATURE_CORRELATION': 'FEATURE_CORRELATION'
        }
        
        main_alert = {
            'type': type_mapping.get(feature_alert['type'], 'FEATURE_MONITORING'),
            'severity': feature_alert['severity'],
            'message': feature_alert['message'],
            'timestamp': feature_alert['timestamp'],
            'source': 'FeatureMonitoring',
            'additional_info': {
                'original_type': feature_alert['type'],
                'feature_name': feature_alert.get('feature_name'),
                'dataset_name': feature_alert.get('dataset_name'),
                'alert_data': feature_alert.get('alert_data', {})
            }
        }
        
        return main_alert
    
    def run_comprehensive_feature_monitoring(self):
        """Run comprehensive feature monitoring and alerting"""
        self.logger.info("🔍 Starting comprehensive feature monitoring cycle...")
        
        all_alerts = []
        
        try:
            # Check for feature drift alerts
            drift_alerts = self.check_feature_drift_alerts()
            all_alerts.extend(drift_alerts)
            
            # Check for data quality alerts
            quality_alerts = self.check_data_quality_alerts()
            all_alerts.extend(quality_alerts)
            
            # Check for correlation alerts
            correlation_alerts = self.check_feature_correlation_alerts()
            all_alerts.extend(correlation_alerts)
            
            # Process and send alerts
            if all_alerts:
                self.logger.info(f"📊 Found {len(all_alerts)} feature monitoring alerts")
                
                # Save feature alerts
                for alert in all_alerts:
                    self.save_feature_alert(alert)
                
                # Convert to main alert system format and send
                main_system_alerts = [self.format_alert_for_main_system(alert) for alert in all_alerts]
                self.alert_manager.send_alerts(main_system_alerts)
                
                # Update alert statistics
                self.update_alert_statistics(all_alerts)
            else:
                self.logger.info("✅ No feature monitoring alerts detected")
            
            # Log monitoring cycle completion
            self.logger.info("✅ Feature monitoring cycle completed successfully")
        
        except Exception as e:
            self.logger.error(f"❌ Error in feature monitoring cycle: {e}")
            
            # Send system error alert
            system_alert = {
                'type': 'FEATURE_MONITORING_ERROR',
                'severity': 'HIGH',
                'message': f'Error in feature monitoring system: {str(e)}',
                'timestamp': datetime.now(),
                'source': 'FeatureMonitoring'
            }
            self.alert_manager.send_alerts([system_alert])
    
    def update_alert_statistics(self, alerts: List[Dict[str, Any]]):
        """Update daily alert statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.now().date().isoformat()
        
        # Group alerts by type
        alert_counts = {}
        severity_mapping = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
        
        for alert in alerts:
            alert_type = alert['type']
            severity_score = severity_mapping.get(alert['severity'], 2)
            
            if alert_type not in alert_counts:
                alert_counts[alert_type] = {'count': 0, 'severity_sum': 0}
            
            alert_counts[alert_type]['count'] += 1
            alert_counts[alert_type]['severity_sum'] += severity_score
        
        # Update statistics
        for alert_type, stats in alert_counts.items():
            avg_severity = stats['severity_sum'] / stats['count']
            
            cursor.execute('''
                INSERT OR REPLACE INTO alert_statistics 
                (date, alert_type, count, avg_severity, resolution_rate)
                VALUES (?, ?, ?, ?, ?)
            ''', (today, alert_type, stats['count'], avg_severity, 0.0))  # Resolution rate calculated separately
        
        conn.commit()
        conn.close()
    
    def get_alert_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get summary of recent alerts"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT alert_type, severity, COUNT(*) as count
            FROM feature_alerts 
            WHERE timestamp >= datetime('now', '-{} days')
            GROUP BY alert_type, severity
            ORDER BY count DESC
        '''.format(days)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        summary = {
            'period_days': days,
            'total_alerts': df['count'].sum() if not df.empty else 0,
            'alert_breakdown': df.to_dict('records') if not df.empty else [],
            'most_common_alert': df.iloc[0]['alert_type'] if not df.empty else None
        }
        
        return summary

def main():
    """Main entry point for enhanced feature alert monitoring"""
    feature_alert_manager = FeatureAlertManager()
    
    # Run comprehensive monitoring
    feature_alert_manager.run_comprehensive_feature_monitoring()
    
    # Print alert summary
    summary = feature_alert_manager.get_alert_summary()
    print(f"📊 Alert Summary (Last 7 days): {summary['total_alerts']} total alerts")
    
    if summary['alert_breakdown']:
        print("🔍 Alert Breakdown:")
        for alert in summary['alert_breakdown'][:5]:
            print(f"   • {alert['alert_type']} ({alert['severity']}): {alert['count']} alerts")

if __name__ == "__main__":
    main()