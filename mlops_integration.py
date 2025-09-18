"""
MLOps Integration Script - Complete Production Pipeline
Integrates all MLOps components: monitoring, alerts, retraining, and A/B testing
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Any
import subprocess

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alert_system import AlertManager
from ab_testing_framework import ABTestManager, ABTestConfig
from automated_retraining import AutomatedRetrainingSystem

class MLOpsPipeline:
    """Main MLOps pipeline coordinator"""
    
    def __init__(self, config_file: str = "mlops_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
        self.setup_logging()
        
        # Initialize components
        self.alert_manager = AlertManager()
        self.ab_test_manager = ABTestManager()
        self.retraining_system = AutomatedRetrainingSystem()
        
    def load_config(self) -> Dict[str, Any]:
        """Load MLOps configuration"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            self.logger.warning(f"Config file not found: {self.config_file}")
            return {}
    
    def setup_logging(self):
        """Setup logging for MLOps pipeline"""
        os.makedirs('logs', exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/mlops_pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('MLOpsPipeline')
    
    def setup_environment(self):
        """Setup MLOps environment"""
        self.logger.info("🔧 Setting up MLOps environment...")
        
        # Create necessary directories
        directories = [
            'logs',
            'models',
            'data/processed',
            'reports',
            'ab_tests'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"📁 Created directory: {directory}")
        
        # Set environment variables for email (if configured)
        email_config = self.config.get('email_settings', {})
        if email_config and self.config.get('notification_channels', {}).get('email_enabled'):
            self.logger.info("📧 Email notifications configured")
        
        self.logger.info("✅ Environment setup completed")
    
    def setup_crontab(self):
        """Setup automated scheduling"""
        self.logger.info("⏰ Setting up automated scheduling...")
        
        if not self.config.get('retraining_schedule', {}).get('enabled', True):
            self.logger.info("ℹ️ Automated retraining is disabled")
            return
        
        try:
            # Run the crontab setup script
            setup_script = "./setup_crontab.sh"
            
            if os.path.exists(setup_script):
                result = subprocess.run([setup_script, "setup"], 
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.logger.info("✅ Crontab setup completed successfully")
                    self.logger.info("📅 Weekly retraining scheduled for Monday 23:00")
                else:
                    self.logger.error(f"❌ Crontab setup failed: {result.stderr}")
            else:
                self.logger.warning("⚠️ Crontab setup script not found")
                
        except Exception as e:
            self.logger.error(f"❌ Error setting up crontab: {e}")
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive system health check"""
        self.logger.info("🏥 Running system health check...")
        
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'HEALTHY',
            'components': {}
        }
        
        # Check alert system
        try:
            alert_health = self.alert_manager.run_health_check()
            health_status['components']['alerts'] = alert_health['overall_status']
        except Exception as e:
            health_status['components']['alerts'] = f'ERROR: {e}'
            health_status['overall_status'] = 'UNHEALTHY'
        
        # Check A/B testing system
        try:
            ab_tests = self.ab_test_manager.list_all_tests()
            health_status['components']['ab_testing'] = 'OK'
            health_status['components']['active_tests'] = len(ab_tests[ab_tests['status'] == 'RUNNING'])
        except Exception as e:
            health_status['components']['ab_testing'] = f'ERROR: {e}'
        
        # Check retraining system
        try:
            retrain_status = self.retraining_system.check_prerequisites()
            health_status['components']['retraining'] = 'OK' if retrain_status['ready'] else 'WARNING'
        except Exception as e:
            health_status['components']['retraining'] = f'ERROR: {e}'
        
        # Check model files
        model_files = [
            'models/fpl_points_model.joblib',
            'models/fpl_gw_model.joblib'
        ]
        
        missing_models = []
        for model_file in model_files:
            if not os.path.exists(model_file):
                missing_models.append(model_file)
        
        if missing_models:
            health_status['components']['models'] = f'WARNING: Missing {len(missing_models)} models'
            if health_status['overall_status'] == 'HEALTHY':
                health_status['overall_status'] = 'WARNING'
        else:
            health_status['components']['models'] = 'OK'
        
        self.logger.info(f"🏥 Health check completed: {health_status['overall_status']}")
        return health_status
    
    def run_monitoring_cycle(self):
        """Run a complete monitoring cycle"""
        self.logger.info("📊 Starting monitoring cycle...")
        
        try:
            # Run alert monitoring
            self.alert_manager.run_monitoring_cycle()
            
            # Analyze active A/B tests
            active_tests = self.ab_test_manager.get_active_tests()
            for test in active_tests:
                self.logger.info(f"🧪 Analyzing A/B test: {test['test_name']}")
                result = self.ab_test_manager.analyze_test(test['test_name'])
                
                if result and result.is_significant:
                    self.logger.info(f"🎯 A/B test {test['test_name']} has significant results: {result.winner} wins!")
            
            self.logger.info("✅ Monitoring cycle completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error in monitoring cycle: {e}")
    
    def create_sample_ab_test(self):
        """Create a sample A/B test for demonstration"""
        if not self.config.get('ab_testing', {}).get('enabled', True):
            self.logger.info("ℹ️ A/B testing is disabled")
            return
        
        try:
            config = ABTestConfig(
                test_name=f"performance_test_{datetime.now().strftime('%Y%m%d')}",
                description="Weekly performance comparison test",
                start_date=datetime.now(),
                end_date=datetime.now(),
                traffic_split=self.config.get('ab_testing', {}).get('default_traffic_split', 0.1),
                model_a_path="models/fpl_points_model.joblib",
                model_b_path="models/fpl_points_model_backup.joblib",
                success_metric="absolute_error"
            )
            
            success = self.ab_test_manager.create_test(config)
            if success:
                self.logger.info(f"🧪 Created A/B test: {config.test_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create A/B test: {e}")
    
    def start_dashboard(self):
        """Start the performance monitoring dashboard"""
        self.logger.info("📊 Starting performance dashboard...")
        
        try:
            import streamlit.web.cli as stcli
            import sys
            
            # Set streamlit arguments
            sys.argv = [
                "streamlit", 
                "run", 
                "performance_dashboard.py",
                "--server.port=8501",
                "--server.address=localhost"
            ]
            
            stcli.main()
            
        except ImportError:
            self.logger.error("❌ Streamlit not installed. Run: pip install streamlit")
        except Exception as e:
            self.logger.error(f"❌ Failed to start dashboard: {e}")
    
    def generate_report(self) -> str:
        """Generate MLOps status report"""
        self.logger.info("📋 Generating MLOps status report...")
        
        report = []
        report.append("# FPL MLOps Pipeline Status Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # System Health
        health = self.run_health_check()
        report.append("## 🏥 System Health")
        report.append(f"**Overall Status:** {health['overall_status']}")
        report.append("")
        
        for component, status in health['components'].items():
            report.append(f"- **{component.title()}:** {status}")
        report.append("")
        
        # A/B Tests
        try:
            ab_tests = self.ab_test_manager.list_all_tests()
            report.append("## 🧪 A/B Testing Status")
            
            if not ab_tests.empty:
                running_tests = ab_tests[ab_tests['status'] == 'RUNNING']
                completed_tests = ab_tests[ab_tests['status'] == 'COMPLETED']
                
                report.append(f"- **Running Tests:** {len(running_tests)}")
                report.append(f"- **Completed Tests:** {len(completed_tests)}")
                
                if len(running_tests) > 0:
                    report.append("\n**Active Tests:**")
                    for _, test in running_tests.iterrows():
                        report.append(f"- {test['test_name']}: {test['description']}")
            else:
                report.append("- No A/B tests found")
            
            report.append("")
        except Exception as e:
            report.append(f"- Error loading A/B tests: {e}")
            report.append("")
        
        # Recent Alerts
        try:
            recent_alerts = self.alert_manager.get_recent_alerts(days=7)
            report.append("## 🚨 Recent Alerts (Last 7 Days)")
            
            if not recent_alerts.empty:
                alert_counts = recent_alerts['severity'].value_counts()
                for severity, count in alert_counts.items():
                    report.append(f"- **{severity}:** {count} alerts")
                
                report.append("\n**Latest Alerts:**")
                for _, alert in recent_alerts.head(5).iterrows():
                    report.append(f"- {alert['alert_type']} ({alert['severity']}): {alert['message']}")
            else:
                report.append("- No alerts in the last 7 days ✅")
            
            report.append("")
        except Exception as e:
            report.append(f"- Error loading alerts: {e}")
            report.append("")
        
        # Configuration
        report.append("## ⚙️ Configuration")
        report.append(f"- **Email Alerts:** {'Enabled' if self.config.get('notification_channels', {}).get('email_enabled') else 'Disabled'}")
        report.append(f"- **Auto Retraining:** {'Enabled' if self.config.get('retraining_schedule', {}).get('enabled') else 'Disabled'}")
        report.append(f"- **A/B Testing:** {'Enabled' if self.config.get('ab_testing', {}).get('enabled') else 'Disabled'}")
        report.append("")
        
        # Save report
        report_content = "\n".join(report)
        report_file = f"reports/mlops_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        os.makedirs('reports', exist_ok=True)
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"📋 Report saved: {report_file}")
        return report_file

def main():
    """Main entry point for MLOps pipeline"""
    parser = argparse.ArgumentParser(description="FPL MLOps Pipeline Management")
    parser.add_argument('command', choices=[
        'setup', 'health-check', 'monitor', 'dashboard', 
        'report', 'ab-test', 'crontab'
    ], help='Command to execute')
    parser.add_argument('--config', default='mlops_config.json', 
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MLOpsPipeline(args.config)
    
    if args.command == 'setup':
        print("🚀 Setting up MLOps environment...")
        pipeline.setup_environment()
        pipeline.setup_crontab()
        print("✅ MLOps setup completed!")
        print("")
        print("🎯 Next steps:")
        print("1. Configure email settings in mlops_config.json")
        print("2. Run health check: python mlops_integration.py health-check")
        print("3. Start monitoring: python mlops_integration.py monitor")
        print("4. View dashboard: python mlops_integration.py dashboard")
        
    elif args.command == 'health-check':
        print("🏥 Running system health check...")
        health = pipeline.run_health_check()
        print(f"📊 Overall Status: {health['overall_status']}")
        print("")
        print("📋 Component Status:")
        for component, status in health['components'].items():
            print(f"  - {component.title()}: {status}")
        
    elif args.command == 'monitor':
        print("📊 Running monitoring cycle...")
        pipeline.run_monitoring_cycle()
        print("✅ Monitoring cycle completed")
        
    elif args.command == 'dashboard':
        print("📊 Starting performance dashboard...")
        print("🌐 Dashboard will be available at: http://localhost:8501")
        pipeline.start_dashboard()
        
    elif args.command == 'report':
        print("📋 Generating MLOps status report...")
        report_file = pipeline.generate_report()
        print(f"📄 Report saved: {report_file}")
        
    elif args.command == 'ab-test':
        print("🧪 Creating sample A/B test...")
        pipeline.create_sample_ab_test()
        print("✅ A/B test setup completed")
        
    elif args.command == 'crontab':
        print("⏰ Setting up automated scheduling...")
        pipeline.setup_crontab()
        print("✅ Crontab setup completed")

if __name__ == "__main__":
    main()