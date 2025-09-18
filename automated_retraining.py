#!/usr/bin/env python3
"""
Automated Weekly Model Retraining System
Runs every Monday night at 23:00 to retrain models with latest gameweek data
"""

import sys
import logging
import json
import smtplib
from datetime import datetime, timedelta
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import traceback

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from fpl_predictor import FPLPredictor

class AutomatedRetrainingSystem:
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.logs_dir = self.project_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load configuration
        self.config = self.load_config()
        
    def setup_logging(self):
        """Setup comprehensive logging for retraining process"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"retraining_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Starting automated retraining system")
        
    def load_config(self):
        """Load configuration for alerts and monitoring"""
        config_file = self.project_dir / "config" / "retraining_config.json"
        
        default_config = {
            "email_alerts": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": "",
                "sender_password": "",
                "recipient_emails": []
            },
            "performance_thresholds": {
                "min_rmse_improvement": 0.01,
                "max_acceptable_rmse": 1.0,
                "min_samples_for_training": 100
            },
            "retry_config": {
                "max_retries": 3,
                "retry_delay_minutes": 30
            }
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                self.logger.info("✅ Loaded configuration from file")
                return {**default_config, **config}
            except Exception as e:
                self.logger.warning(f"⚠️ Error loading config: {e}, using defaults")
                
        # Create config directory and default config
        config_file.parent.mkdir(exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        self.logger.info("📝 Created default configuration file")
        return default_config
    
    def check_prerequisites(self):
        """Check if system is ready for retraining"""
        self.logger.info("🔍 Checking retraining prerequisites...")
        
        try:
            # Check if FPL data is available
            predictor = FPLPredictor()
            
            # Check data availability
            data_files = [
                predictor.data_dir / 'players_latest.csv',
                predictor.data_dir / 'fixtures_latest.csv'
            ]
            
            missing_files = [f for f in data_files if not f.exists()]
            if missing_files:
                self.logger.error(f"❌ Missing data files: {missing_files}")
                return False
                
            # Check if new gameweek data is available
            if not predictor._should_retrain_models():
                self.logger.info("ℹ️ No retraining needed - models are current")
                return False
                
            self.logger.info("✅ Prerequisites check passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Prerequisites check failed: {e}")
            return False
    
    def run_data_update(self):
        """Update FPL data before retraining"""
        self.logger.info("🔄 Updating FPL data...")
        
        try:
            predictor = FPLPredictor()
            success = predictor.update_data()
            
            if success:
                self.logger.info("✅ Data update completed successfully")
                return True
            else:
                self.logger.error("❌ Data update failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Data update failed with exception: {e}")
            return False
    
    def run_model_retraining(self):
        """Execute the model retraining process"""
        self.logger.info("🚀 Starting model retraining...")
        
        try:
            predictor = FPLPredictor()
            
            # Record start time
            start_time = datetime.now()
            
            # Run retraining
            success = predictor._retrain_models_with_new_data()
            
            # Record end time and duration
            end_time = datetime.now()
            duration = end_time - start_time
            
            if success:
                self.logger.info(f"✅ Model retraining completed successfully in {duration}")
                
                # Log retraining statistics
                self.log_retraining_stats(predictor, duration)
                return True
            else:
                self.logger.error("❌ Model retraining failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Model retraining failed with exception: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def log_retraining_stats(self, predictor, duration):
        """Log detailed retraining statistics"""
        try:
            # Load latest model metadata
            metadata_files = list(predictor.models_dir.glob('model_metadata_*.json'))
            if metadata_files:
                latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
                with open(latest_metadata, 'r') as f:
                    metadata = json.load(f)
                
                dataset_info = metadata.get('dataset_info', {})
                model_performance = metadata.get('model_performance', {})
                
                self.logger.info("📊 Retraining Statistics:")
                self.logger.info(f"   Duration: {duration}")
                self.logger.info(f"   Training Samples: {dataset_info.get('training_samples', 'Unknown')}")
                self.logger.info(f"   Features: {dataset_info.get('features_count', 'Unknown')}")
                self.logger.info(f"   Gameweeks: {dataset_info.get('gameweeks_trained', 'Unknown')}")
                
                for model_name, metrics in model_performance.items():
                    rmse = metrics.get('train_rmse', 'Unknown')
                    mae = metrics.get('train_mae', 'Unknown')
                    self.logger.info(f"   {model_name}: RMSE={rmse}, MAE={mae}")
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Could not log retraining stats: {e}")
    
    def run_health_check(self):
        """Run post-retraining health check"""
        self.logger.info("🏥 Running post-retraining health check...")
        
        try:
            predictor = FPLPredictor()
            health_status = predictor._run_model_health_check()
            
            if isinstance(health_status, dict) and 'error' not in health_status:
                passed_checks = sum(health_status.values())
                total_checks = len(health_status)
                health_score = (passed_checks / total_checks) * 100
                
                self.logger.info(f"📊 Model Health Score: {health_score:.1f}%")
                
                if health_score >= 80:
                    self.logger.info("✅ Health check passed")
                    return True
                else:
                    self.logger.error(f"❌ Health check failed: {health_score:.1f}% < 80%")
                    return False
            else:
                self.logger.error("❌ Health check failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Health check failed with exception: {e}")
            return False
    
    def send_alert_email(self, subject, message, is_success=True):
        """Send email alert about retraining status"""
        if not self.config['email_alerts']['enabled']:
            return
            
        try:
            email_config = self.config['email_alerts']
            
            msg = MimeMultipart()
            msg['From'] = email_config['sender_email']
            msg['To'] = ', '.join(email_config['recipient_emails'])
            msg['Subject'] = f"[FPL ML] {subject}"
            
            # Add timestamp and status emoji
            status_emoji = "✅" if is_success else "❌"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            email_body = f"""
{status_emoji} FPL Model Retraining Report
Time: {timestamp}

{message}

---
This is an automated message from the FPL Analytics MLOps system.
            """
            
            msg.attach(MimeText(email_body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['sender_email'], email_config['sender_password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info("📧 Alert email sent successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to send alert email: {e}")
    
    def run_automated_retraining(self):
        """Main automated retraining process"""
        self.logger.info("🤖 Starting automated weekly retraining process")
        
        success_steps = []
        failed_steps = []
        
        try:
            # Step 1: Check prerequisites
            if self.check_prerequisites():
                success_steps.append("Prerequisites check")
            else:
                failed_steps.append("Prerequisites check")
                self.send_alert_email(
                    "Retraining Skipped - No Update Needed",
                    "Model retraining was skipped because models are already current or prerequisites not met.",
                    is_success=True
                )
                return True  # Not an error condition
            
            # Step 2: Update data
            if self.run_data_update():
                success_steps.append("Data update")
            else:
                failed_steps.append("Data update")
                
            # Step 3: Retrain models
            if not failed_steps and self.run_model_retraining():
                success_steps.append("Model retraining")
            else:
                failed_steps.append("Model retraining")
                
            # Step 4: Health check
            if not failed_steps and self.run_health_check():
                success_steps.append("Health check")
            else:
                failed_steps.append("Health check")
            
            # Send summary email
            if not failed_steps:
                self.logger.info("🎉 Automated retraining completed successfully!")
                self.send_alert_email(
                    "Retraining Completed Successfully",
                    f"All steps completed successfully:\n• " + "\n• ".join(success_steps),
                    is_success=True
                )
                return True
            else:
                self.logger.error(f"❌ Automated retraining failed at: {', '.join(failed_steps)}")
                self.send_alert_email(
                    "Retraining Failed",
                    f"Failed steps: {', '.join(failed_steps)}\nSuccessful steps: {', '.join(success_steps)}",
                    is_success=False
                )
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Automated retraining system error: {e}")
            self.logger.error(traceback.format_exc())
            self.send_alert_email(
                "Retraining System Error",
                f"System error occurred: {str(e)}\n\nCheck logs for details.",
                is_success=False
            )
            return False

def main():
    """Main entry point for automated retraining"""
    print("🤖 FPL Automated Weekly Retraining System")
    print("=" * 50)
    
    system = AutomatedRetrainingSystem()
    success = system.run_automated_retraining()
    
    # Exit with appropriate code for cron monitoring
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
