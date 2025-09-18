"""
Automated Alert System for FPL Model Performance
Monitors model performance and sends notifications when issues are detected
"""

import smtplib
import json
import os
import sqlite3
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Tuple, Optional, Any
import logging
import requests
import pandas as pd
from dataclasses import dataclass
import numpy as np
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AlertConfig:
    """Configuration for alert thresholds and settings"""
    rmse_threshold: float = 0.3
    accuracy_threshold: float = 80.0
    drift_threshold: float = 5.0
    prediction_count_threshold: int = 100
    email_enabled: bool = True
    slack_enabled: bool = False
    discord_enabled: bool = False
    check_interval_hours: int = 6
    
class AlertManager:
    """Manages alerts and notifications for model performance"""
    
    def __init__(self, config_file: str = "alert_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        self.db_path = "model_performance.db"
        self.setup_logging()
        
    def load_config(self) -> AlertConfig:
        """Load alert configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                return AlertConfig(**config_data)
            except Exception as e:
                logging.warning(f"Failed to load config: {e}. Using defaults.")
        
        # Create default config file
        default_config = AlertConfig()
        self.save_config(default_config)
        return default_config
    
    def save_config(self, config: AlertConfig):
        """Save alert configuration to file"""
        config_dict = {
            'rmse_threshold': config.rmse_threshold,
            'accuracy_threshold': config.accuracy_threshold,
            'drift_threshold': config.drift_threshold,
            'prediction_count_threshold': config.prediction_count_threshold,
            'email_enabled': config.email_enabled,
            'slack_enabled': config.slack_enabled,
            'discord_enabled': config.discord_enabled,
            'check_interval_hours': config.check_interval_hours
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def setup_logging(self):
        """Setup logging for alerts"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/alerts.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('AlertManager')
    
    def check_performance_alerts(self) -> List[Dict[str, Any]]:
        """Check for performance-related alerts"""
        alerts = []
        
        try:
            # Check RMSE threshold
            rmse_alert = self.check_rmse_threshold()
            if rmse_alert:
                alerts.append(rmse_alert)
            
            # Check accuracy threshold
            accuracy_alert = self.check_accuracy_threshold()
            if accuracy_alert:
                alerts.append(accuracy_alert)
            
            # Check performance drift
            drift_alert = self.check_performance_drift()
            if drift_alert:
                alerts.append(drift_alert)
            
            # Check prediction volume
            volume_alert = self.check_prediction_volume()
            if volume_alert:
                alerts.append(volume_alert)
            
            # Check data freshness
            freshness_alert = self.check_data_freshness()
            if freshness_alert:
                alerts.append(freshness_alert)
            
        except Exception as e:
            self.logger.error(f"Error checking performance alerts: {e}")
            alerts.append({
                'type': 'SYSTEM_ERROR',
                'severity': 'HIGH',
                'message': f'Error in alert system: {str(e)}',
                'timestamp': datetime.now()
            })
        
        return alerts
    
    def check_rmse_threshold(self) -> Optional[Dict[str, Any]]:
        """Check if RMSE exceeds threshold"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get latest RMSE
            query = '''
                SELECT metric_value, timestamp FROM performance_metrics 
                WHERE metric_name = 'rmse' 
                ORDER BY timestamp DESC LIMIT 1
            '''
            result = pd.read_sql_query(query, conn)
            
            if not result.empty:
                latest_rmse = result.iloc[0]['metric_value']
                
                if latest_rmse > self.config.rmse_threshold:
                    return {
                        'type': 'RMSE_THRESHOLD',
                        'severity': 'HIGH' if latest_rmse > self.config.rmse_threshold * 1.5 else 'MEDIUM',
                        'message': f'RMSE ({latest_rmse:.3f}) exceeds threshold ({self.config.rmse_threshold:.3f})',
                        'value': latest_rmse,
                        'threshold': self.config.rmse_threshold,
                        'timestamp': datetime.now()
                    }
        finally:
            conn.close()
        
        return None
    
    def check_accuracy_threshold(self) -> Optional[Dict[str, Any]]:
        """Check if accuracy falls below threshold"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            query = '''
                SELECT metric_value, timestamp FROM performance_metrics 
                WHERE metric_name = 'accuracy' 
                ORDER BY timestamp DESC LIMIT 1
            '''
            result = pd.read_sql_query(query, conn)
            
            if not result.empty:
                latest_accuracy = result.iloc[0]['metric_value']
                
                if latest_accuracy < self.config.accuracy_threshold:
                    return {
                        'type': 'ACCURACY_THRESHOLD',
                        'severity': 'HIGH' if latest_accuracy < self.config.accuracy_threshold * 0.9 else 'MEDIUM',
                        'message': f'Accuracy ({latest_accuracy:.1f}%) below threshold ({self.config.accuracy_threshold:.1f}%)',
                        'value': latest_accuracy,
                        'threshold': self.config.accuracy_threshold,
                        'timestamp': datetime.now()
                    }
        finally:
            conn.close()
        
        return None
    
    def check_performance_drift(self) -> Optional[Dict[str, Any]]:
        """Check for significant performance drift"""
        try:
            drift = self.calculate_performance_drift()
            
            if abs(drift) > self.config.drift_threshold:
                severity = 'CRITICAL' if abs(drift) > self.config.drift_threshold * 2 else 'HIGH'
                direction = 'degraded' if drift > 0 else 'improved'
                
                return {
                    'type': 'PERFORMANCE_DRIFT',
                    'severity': severity,
                    'message': f'Performance has {direction} by {abs(drift):.1f}% (threshold: {self.config.drift_threshold:.1f}%)',
                    'value': drift,
                    'threshold': self.config.drift_threshold,
                    'timestamp': datetime.now()
                }
        except Exception as e:
            self.logger.error(f"Error calculating drift: {e}")
        
        return None
    
    def check_prediction_volume(self) -> Optional[Dict[str, Any]]:
        """Check if prediction volume is unusually low"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Check predictions in last 24 hours
            query = '''
                SELECT COUNT(*) as count FROM predictions 
                WHERE timestamp >= datetime('now', '-1 day')
            '''
            result = pd.read_sql_query(query, conn)
            
            if not result.empty:
                daily_count = result.iloc[0]['count']
                
                if daily_count < self.config.prediction_count_threshold:
                    return {
                        'type': 'LOW_PREDICTION_VOLUME',
                        'severity': 'MEDIUM' if daily_count > 0 else 'HIGH',
                        'message': f'Low prediction volume: {daily_count} predictions in last 24h (threshold: {self.config.prediction_count_threshold})',
                        'value': daily_count,
                        'threshold': self.config.prediction_count_threshold,
                        'timestamp': datetime.now()
                    }
        finally:
            conn.close()
        
        return None
    
    def check_data_freshness(self) -> Optional[Dict[str, Any]]:
        """Check if data is stale"""
        try:
            data_files = [
                'data/processed/players_processed.csv',
                'data/processed/fixtures_processed.csv',
                'models/fpl_points_model.joblib'
            ]
            
            stale_files = []
            for file_path in data_files:
                if os.path.exists(file_path):
                    mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    age_hours = (datetime.now() - mod_time).total_seconds() / 3600
                    
                    if age_hours > 48:  # 2 days threshold
                        stale_files.append(f"{file_path} ({age_hours:.1f}h old)")
            
            if stale_files:
                return {
                    'type': 'STALE_DATA',
                    'severity': 'MEDIUM',
                    'message': f'Stale data detected: {", ".join(stale_files)}',
                    'files': stale_files,
                    'timestamp': datetime.now()
                }
        except Exception as e:
            self.logger.error(f"Error checking data freshness: {e}")
        
        return None
    
    def calculate_performance_drift(self, lookback_days: int = 7) -> float:
        """Calculate performance drift over time"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            query = '''
                SELECT metric_value, timestamp FROM performance_metrics 
                WHERE metric_name = 'rmse' 
                AND timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp DESC
            '''.format(lookback_days * 2)
            
            df = pd.read_sql_query(query, conn)
            
            if len(df) < 10:
                return 0.0
            
            # Split into recent and historical
            mid_point = len(df) // 2
            recent_avg = df.iloc[:mid_point]['metric_value'].mean()
            historical_avg = df.iloc[mid_point:]['metric_value'].mean()
            
            # Calculate percentage change (positive = worse performance for RMSE)
            if historical_avg != 0:
                drift = ((recent_avg - historical_avg) / historical_avg) * 100
            else:
                drift = 0.0
            
            return drift
            
        finally:
            conn.close()
    
    def send_alerts(self, alerts: List[Dict[str, Any]]):
        """Send alerts via configured channels"""
        if not alerts:
            return
        
        for alert in alerts:
            self.log_alert_to_db(alert)
            
            if self.config.email_enabled:
                self.send_email_alert(alert)
            
            if self.config.slack_enabled:
                self.send_slack_alert(alert)
            
            if self.config.discord_enabled:
                self.send_discord_alert(alert)
            
            self.logger.info(f"Alert sent: {alert['type']} - {alert['message']}")
    
    def log_alert_to_db(self, alert: Dict[str, Any]):
        """Log alert to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Ensure alerts table exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    additional_info TEXT
                )
            ''')
            
            cursor.execute('''
                INSERT INTO alerts (timestamp, alert_type, severity, message)
                VALUES (?, ?, ?, ?)
            ''', (
                alert['timestamp'].isoformat(),
                alert['type'],
                alert['severity'],
                alert['message']
            ))
            conn.commit()
        finally:
            conn.close()
    
    def send_email_alert(self, alert: Dict[str, Any]):
        """Send alert via email"""
        try:
            # Email configuration (should be set via environment variables)
            smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
            smtp_port = int(os.getenv('SMTP_PORT', '587'))
            email_user = os.getenv('EMAIL_USER')
            email_password = os.getenv('EMAIL_PASSWORD')
            alert_recipients = os.getenv('ALERT_RECIPIENTS', '').split(',')
            
            if not email_user or not email_password or not alert_recipients:
                self.logger.warning("Email configuration incomplete. Skipping email alert.")
                return
            
            # Create email
            msg = MIMEMultipart()
            msg['From'] = email_user
            msg['To'] = ', '.join(alert_recipients)
            msg['Subject'] = f"🚨 FPL Model Alert: {alert['type']}"
            
            # Email body
            body = f"""
            FPL Model Performance Alert
            
            Alert Type: {alert['type']}
            Severity: {alert['severity']}
            Time: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
            
            Message: {alert['message']}
            
            Please check the performance dashboard for more details.
            
            --
            FPL Automated Monitoring System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(email_user, email_password)
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email alert sent for {alert['type']}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    def send_slack_alert(self, alert: Dict[str, Any]):
        """Send alert via Slack webhook"""
        try:
            webhook_url = os.getenv('SLACK_WEBHOOK_URL')
            if not webhook_url:
                self.logger.warning("Slack webhook URL not configured")
                return
            
            severity_emoji = {
                'LOW': '🟡',
                'MEDIUM': '🟠',
                'HIGH': '🔴',
                'CRITICAL': '🚨'
            }
            
            payload = {
                'text': f"{severity_emoji.get(alert['severity'], '📝')} FPL Model Alert",
                'attachments': [{
                    'color': 'danger' if alert['severity'] in ['HIGH', 'CRITICAL'] else 'warning',
                    'fields': [
                        {'title': 'Alert Type', 'value': alert['type'], 'short': True},
                        {'title': 'Severity', 'value': alert['severity'], 'short': True},
                        {'title': 'Message', 'value': alert['message'], 'short': False},
                        {'title': 'Time', 'value': alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S'), 'short': True}
                    ]
                }]
            }
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
            self.logger.info(f"Slack alert sent for {alert['type']}")
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")
    
    def send_discord_alert(self, alert: Dict[str, Any]):
        """Send alert via Discord webhook"""
        try:
            webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
            if not webhook_url:
                self.logger.warning("Discord webhook URL not configured")
                return
            
            severity_colors = {
                'LOW': 0xFFFF00,      # Yellow
                'MEDIUM': 0xFF8C00,    # Orange
                'HIGH': 0xFF0000,      # Red
                'CRITICAL': 0x8B0000   # Dark Red
            }
            
            embed = {
                'title': f"🚨 FPL Model Alert: {alert['type']}",
                'description': alert['message'],
                'color': severity_colors.get(alert['severity'], 0x808080),
                'fields': [
                    {'name': 'Severity', 'value': alert['severity'], 'inline': True},
                    {'name': 'Time', 'value': alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S'), 'inline': True}
                ],
                'timestamp': alert['timestamp'].isoformat()
            }
            
            payload = {'embeds': [embed]}
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
            self.logger.info(f"Discord alert sent for {alert['type']}")
            
        except Exception as e:
            self.logger.error(f"Failed to send Discord alert: {e}")
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check"""
        health_status = {
            'timestamp': datetime.now(),
            'overall_status': 'HEALTHY',
            'checks': {}
        }
        
        try:
            # Check database connectivity
            conn = sqlite3.connect(self.db_path)
            conn.execute('SELECT 1')
            conn.close()
            health_status['checks']['database'] = 'OK'
        except Exception as e:
            health_status['checks']['database'] = f'ERROR: {e}'
            health_status['overall_status'] = 'UNHEALTHY'
        
        # Check model file
        model_path = 'models/fpl_points_model.joblib'
        if os.path.exists(model_path):
            mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
            age_days = (datetime.now() - mod_time).days
            
            if age_days <= 7:
                health_status['checks']['model'] = 'OK'
            else:
                health_status['checks']['model'] = f'WARNING: Model is {age_days} days old'
                if health_status['overall_status'] == 'HEALTHY':
                    health_status['overall_status'] = 'WARNING'
        else:
            health_status['checks']['model'] = 'ERROR: Model file not found'
            health_status['overall_status'] = 'UNHEALTHY'
        
        # Check recent performance metrics
        alerts = self.check_performance_alerts()
        if alerts:
            critical_alerts = [a for a in alerts if a['severity'] == 'CRITICAL']
            high_alerts = [a for a in alerts if a['severity'] == 'HIGH']
            
            if critical_alerts:
                health_status['checks']['performance'] = f'CRITICAL: {len(critical_alerts)} critical alerts'
                health_status['overall_status'] = 'CRITICAL'
            elif high_alerts:
                health_status['checks']['performance'] = f'WARNING: {len(high_alerts)} high severity alerts'
                if health_status['overall_status'] in ['HEALTHY', 'WARNING']:
                    health_status['overall_status'] = 'WARNING'
            else:
                health_status['checks']['performance'] = 'OK'
        else:
            health_status['checks']['performance'] = 'OK'
        
        return health_status
    
    def run_monitoring_cycle(self):
        """Run a complete monitoring cycle"""
        self.logger.info("Starting monitoring cycle...")
        
        try:
            # Check for alerts
            alerts = self.check_performance_alerts()
            
            if alerts:
                self.logger.info(f"Found {len(alerts)} alerts")
                self.send_alerts(alerts)
            else:
                self.logger.info("No alerts detected")
            
            # Run health check
            health = self.run_health_check()
            self.logger.info(f"Health check completed: {health['overall_status']}")
            
            # Log health metrics
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_metrics 
                (timestamp, metric_name, metric_value, additional_info)
                VALUES (?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                'health_score',
                1.0 if health['overall_status'] == 'HEALTHY' else 0.5 if health['overall_status'] == 'WARNING' else 0.0,
                json.dumps(health['checks'])
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error in monitoring cycle: {e}")
            
            # Send system error alert
            system_alert = {
                'type': 'MONITORING_ERROR',
                'severity': 'HIGH',
                'message': f'Error in monitoring system: {str(e)}',
                'timestamp': datetime.now()
            }
            self.send_alerts([system_alert])

def main():
    """Main entry point for alert monitoring"""
    alert_manager = AlertManager()
    alert_manager.run_monitoring_cycle()

if __name__ == "__main__":
    main()