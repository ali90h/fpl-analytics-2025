#!/usr/bin/env python3
"""
FPL Feature Monitoring Setup Script
Automates the setup and initial configuration of the feature monitoring system
"""

import os
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'scipy', 'scikit-learn', 
        'streamlit', 'plotly', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ All packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please install manually:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    else:
        print("✅ All required packages are installed!")
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'logs',
        'data/performance_history',
        'data/processed',
        'models'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def create_default_configs():
    """Create default configuration files"""
    
    # Feature alert configuration
    feature_alert_config = {
        "drift_threshold": 0.05,
        "psi_threshold": 0.1,
        "missing_data_threshold": 5.0,
        "outlier_threshold": 10.0,
        "quality_score_threshold": 0.7,
        "consecutive_drift_threshold": 3,
        "drift_check_interval_hours": 12,
        "quality_check_interval_hours": 6
    }
    
    with open('feature_alert_config.json', 'w') as f:
        json.dump(feature_alert_config, f, indent=2)
    
    print("⚙️ Created feature_alert_config.json")
    
    # Main alert configuration
    alert_config = {
        "rmse_threshold": 0.3,
        "accuracy_threshold": 80.0,
        "drift_threshold": 5.0,
        "prediction_count_threshold": 100,
        "email_enabled": False,
        "slack_enabled": False,
        "discord_enabled": False,
        "check_interval_hours": 6
    }
    
    with open('alert_config.json', 'w') as f:
        json.dump(alert_config, f, indent=2)
    
    print("⚙️ Created alert_config.json")

def initialize_databases():
    """Initialize monitoring databases"""
    try:
        # Initialize feature monitoring database
        from feature_monitoring import FeatureMonitor
        monitor = FeatureMonitor()
        print("✅ Initialized feature_monitoring.db")
        
        # Initialize data quality database
        from data_quality_monitor import DataQualityMonitor
        quality_monitor = DataQualityMonitor()
        print("✅ Initialized data_quality_monitoring.db")
        
        # Initialize feature alerts database
        from enhanced_alert_integration import FeatureAlertManager
        alert_manager = FeatureAlertManager()
        print("✅ Initialized feature_alerts.db")
        
    except Exception as e:
        print(f"❌ Error initializing databases: {e}")
        return False
    
    return True

def create_sample_data():
    """Create sample data if FPL data doesn't exist"""
    import pandas as pd
    import numpy as np
    
    sample_data_path = "data/performance_history/model_features.csv"
    
    if not os.path.exists(sample_data_path):
        print("📊 Creating sample FPL data...")
        
        # Create sample player data
        np.random.seed(42)
        n_players = 100
        
        sample_data = {
            'player_id': range(1, n_players + 1),
            'name': [f'Player_{i}' for i in range(1, n_players + 1)],
            'team': np.random.randint(1, 21, n_players),
            'position': np.random.randint(1, 5, n_players),
            'gameweeks_played': np.random.randint(1, 39, n_players),
            'avg_PTS': np.random.normal(5, 2, n_players).clip(0, 20),
            'avg_MP': np.random.normal(70, 20, n_players).clip(0, 90),
            'avg_GS': np.random.exponential(0.3, n_players).clip(0, 3),
            'avg_A': np.random.exponential(0.2, n_players).clip(0, 2),
            'avg_xG': np.random.exponential(0.3, n_players).clip(0, 2),
            'avg_xA': np.random.exponential(0.2, n_players).clip(0, 1.5),
            'avg_CS': np.random.beta(2, 8, n_players),
            'avg_T': np.random.normal(2, 1, n_players).clip(0, 10),
            'avg_CBI': np.random.normal(10, 5, n_players).clip(0, 30),
            'avg_R': np.random.normal(5, 2, n_players).clip(0, 15),
            'avg_BP': np.random.exponential(1, n_players).clip(0, 5),
            'avg_BPS': np.random.normal(25, 15, n_players).clip(0, 80),
            'avg_I': np.random.normal(8, 4, n_players).clip(0, 25),
            'avg_C': np.random.normal(6, 3, n_players).clip(0, 15),
            'avg_T_threat': np.random.normal(30, 20, n_players).clip(0, 100),
            'avg_II': np.random.normal(8, 4, n_players).clip(0, 20),
        }
        
        # Add trend features (optional)
        trend_features = ['PTS', 'MP', 'GS', 'A', 'xG', 'xA', 'CS', 'T', 'CBI', 'R', 'BP', 'BPS', 'I', 'C', 'T_threat', 'II']
        for feature in trend_features:
            sample_data[f'trend_{feature}'] = np.random.normal(0, 0.1, n_players)
        
        df = pd.DataFrame(sample_data)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(sample_data_path), exist_ok=True)
        
        # Save sample data
        df.to_csv(sample_data_path, index=False)
        print(f"✅ Created sample data: {sample_data_path}")
    else:
        print("✅ FPL data already exists!")

def test_system():
    """Test that the monitoring system works"""
    try:
        print("🧪 Testing feature monitoring system...")
        
        # Test feature monitoring
        from feature_monitoring import FeatureMonitor
        import pandas as pd
        
        # Load data
        df = pd.read_csv("data/performance_history/model_features.csv")
        
        # Run monitoring
        monitor = FeatureMonitor()
        results = monitor.run_comprehensive_monitoring(df)
        
        print(f"✅ Feature monitoring test passed!")
        print(f"   • Features monitored: {results['summary']['features_monitored']}")
        print(f"   • Drift detected: {results['summary']['drift_detected']}")
        
        # Test data quality
        from data_quality_monitor import DataQualityMonitor
        quality_monitor = DataQualityMonitor()
        quality_report = quality_monitor.run_comprehensive_quality_check(df, 'fpl_players')
        
        print(f"✅ Data quality test passed!")
        print(f"   • Quality score: {quality_report['quality_score']:.3f}")
        print(f"   • Quality grade: {quality_report['quality_grade']}")
        
        # Test alerts
        from enhanced_alert_integration import FeatureAlertManager
        alert_manager = FeatureAlertManager()
        alert_manager.run_comprehensive_feature_monitoring()
        
        print(f"✅ Alert system test passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "="*60)
    print("🎉 FPL Feature Monitoring System Setup Complete!")
    print("="*60)
    print()
    print("📋 Next Steps:")
    print()
    print("1. 🚀 Launch the monitoring dashboard:")
    print("   streamlit run feature_monitoring_dashboard.py")
    print()
    print("2. 🔧 Configure alerts (optional):")
    print("   • Edit feature_alert_config.json for drift thresholds")
    print("   • Edit alert_config.json for notification settings")
    print("   • Set environment variables for email/Slack notifications")
    print()
    print("3. 📊 Replace sample data with your FPL data:")
    print("   • Update data/performance_history/model_features.csv")
    print("   • Ensure data follows the expected schema")
    print()
    print("4. ⚙️ Integrate with your MLOps pipeline:")
    print("   • Add monitoring calls to your training pipeline")
    print("   • Set up automated monitoring jobs")
    print("   • Configure CI/CD integration")
    print()
    print("5. 📚 Read the documentation:")
    print("   • See FEATURE_MONITORING_GUIDE.md for detailed usage")
    print("   • Check individual Python files for API documentation")
    print()
    print("🌐 Dashboard will be available at: http://localhost:8501")
    print("📧 For support, check the documentation or create an issue")
    print()

def main():
    """Main setup function"""
    print("🔧 FPL Feature Monitoring System Setup")
    print("="*50)
    print()
    
    # Step 1: Check requirements
    print("1️⃣ Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    print()
    
    # Step 2: Create directories
    print("2️⃣ Creating directories...")
    create_directories()
    print()
    
    # Step 3: Create configuration files
    print("3️⃣ Creating configuration files...")
    create_default_configs()
    print()
    
    # Step 4: Initialize databases
    print("4️⃣ Initializing databases...")
    if not initialize_databases():
        sys.exit(1)
    print()
    
    # Step 5: Create sample data
    print("5️⃣ Setting up data...")
    create_sample_data()
    print()
    
    # Step 6: Test system
    print("6️⃣ Testing system...")
    if not test_system():
        print("⚠️ Some tests failed, but setup is mostly complete.")
        print("Check the error messages above and refer to the documentation.")
    print()
    
    # Step 7: Show next steps
    print_next_steps()

if __name__ == "__main__":
    main()