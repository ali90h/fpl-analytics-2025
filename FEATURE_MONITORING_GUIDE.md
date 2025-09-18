# FPL Feature Monitoring System

## Overview

The FPL Feature Monitoring System is a comprehensive solution for tracking feature drift and data quality in Fantasy Premier League (FPL) prediction models. It provides real-time monitoring, statistical analysis, alerting, and visualization capabilities to ensure your ML models continue to perform optimally over time.

## 🎯 Key Features

### 📊 Feature Drift Detection
- **Statistical Tests**: Kolmogorov-Smirnov tests for distribution changes
- **Population Stability Index (PSI)**: Measures distribution shifts over time
- **Correlation Analysis**: Detects changes in feature relationships
- **Threshold-based Alerting**: Configurable sensitivity levels

### 🔍 Data Quality Monitoring
- **Schema Validation**: Ensures data conforms to expected structure
- **Completeness Metrics**: Tracks missing data patterns
- **Outlier Detection**: Identifies anomalous data points
- **Quality Scoring**: Overall data health assessment (A-F grading)

### 📈 Interactive Dashboard
- **Real-time Visualization**: Streamlit-based monitoring interface
- **Historical Trends**: Track changes over time
- **Alert Management**: View and manage active alerts
- **Feature Analysis**: Detailed drift and quality breakdowns

### 🚨 Integrated Alerting
- **Multi-channel Notifications**: Email, Slack, Discord support
- **Alert Suppression**: Prevent notification fatigue
- **Severity Levels**: LOW, MEDIUM, HIGH, CRITICAL classifications
- **Historical Tracking**: Alert audit trail and resolution tracking

## 🛠 System Architecture

```
FPL Feature Monitoring System
├── Core Components
│   ├── feature_monitoring.py          # Main monitoring engine
│   ├── data_quality_monitor.py        # Data quality assessment
│   └── enhanced_alert_integration.py  # Alert management
├── Visualization
│   └── feature_monitoring_dashboard.py # Streamlit dashboard
├── Databases
│   ├── feature_monitoring.db          # Feature statistics & drift
│   ├── data_quality_monitoring.db     # Quality metrics
│   └── feature_alerts.db             # Alert history
└── Configuration
    ├── feature_alert_config.json      # Alert thresholds
    └── alert_config.json             # Notification settings
```

## 🚀 Quick Start Guide

### 1. Installation

```bash
# Install required dependencies
pip install pandas numpy scipy scikit-learn streamlit plotly sqlite3

# Ensure you have the FPL MLOps system installed
# (This includes the base alert_system.py and model data)
```

### 2. Basic Setup

```python
from feature_monitoring import FeatureMonitor
from data_quality_monitor import DataQualityMonitor
import pandas as pd

# Load your FPL feature data
df = pd.read_csv("data/performance_history/model_features.csv")

# Initialize monitors
feature_monitor = FeatureMonitor()
quality_monitor = DataQualityMonitor()

# Run comprehensive monitoring
results = feature_monitor.run_comprehensive_monitoring(df)
quality_report = quality_monitor.run_comprehensive_quality_check(df, 'fpl_players')

print(f"Features monitored: {results['summary']['features_monitored']}")
print(f"Quality grade: {quality_report['quality_grade']}")
```

### 3. Launch Dashboard

```bash
streamlit run feature_monitoring_dashboard.py
```

Access the dashboard at `http://localhost:8501`

## ⚙️ Configuration

### Feature Drift Configuration

```python
from feature_monitoring import FeatureDriftConfig

config = FeatureDriftConfig(
    ks_threshold=0.05,              # KS test p-value threshold
    psi_threshold=0.1,              # PSI threshold for drift detection
    missing_data_threshold=5.0,     # % missing data threshold
    outlier_threshold=2.5,          # Z-score for outlier detection
    reference_window_days=30,       # Historical data window
    comparison_window_days=7        # Recent data window
)

monitor = FeatureMonitor(config=config)
```

### Alert Configuration

```python
from enhanced_alert_integration import FeatureAlertConfig

alert_config = FeatureAlertConfig(
    drift_threshold=0.05,                    # Drift detection sensitivity
    quality_score_threshold=0.7,             # Minimum quality score
    consecutive_drift_threshold=3,           # Alerts after N consecutive drifts
    drift_check_interval_hours=12            # How often to check for drift
)
```

### Email Alerts Setup

Set environment variables for email notifications:

```bash
export EMAIL_USER="your_email@gmail.com"
export EMAIL_PASSWORD="your_app_password"
export ALERT_RECIPIENTS="recipient1@email.com,recipient2@email.com"
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="587"
```

### Slack Integration

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
```

## 📋 Usage Examples

### 1. Basic Feature Drift Detection

```python
from feature_monitoring import FeatureMonitor
import pandas as pd

# Initialize monitor
monitor = FeatureMonitor()

# Load current data
df = pd.read_csv("current_features.csv")

# Detect drift
drift_results = monitor.detect_feature_drift(df)

# Check results
for feature, result in drift_results.items():
    if result.drift_detected:
        print(f"⚠️ Drift detected in {feature}: {result.drift_score:.4f}")
```

### 2. Data Quality Assessment

```python
from data_quality_monitor import DataQualityMonitor

# Initialize quality monitor
quality_monitor = DataQualityMonitor()

# Run quality check
report = quality_monitor.run_comprehensive_quality_check(df, 'fpl_players')

print(f"Quality Score: {report['quality_score']:.3f}")
print(f"Quality Grade: {report['quality_grade']}")

# Review recommendations
for rec in report['recommendations']:
    print(f"• {rec}")
```

### 3. Automated Monitoring Pipeline

```python
from enhanced_alert_integration import FeatureAlertManager

# Initialize alert manager
alert_manager = FeatureAlertManager()

# Run comprehensive monitoring
alert_manager.run_comprehensive_feature_monitoring()

# Get alert summary
summary = alert_manager.get_alert_summary(days=7)
print(f"Total alerts in last 7 days: {summary['total_alerts']}")
```

### 4. Custom Schema Definition

```python
from data_quality_monitor import SchemaDefinition, DataQualityMonitor

# Define custom schema
custom_schema = [
    SchemaDefinition('player_id', 'int', False, 1, 1000),
    SchemaDefinition('avg_points', 'float', True, 0, 30),
    SchemaDefinition('position', 'int', False, 1, 4),
    # Add more fields as needed
]

# Use with quality monitor
monitor = DataQualityMonitor()
monitor.schema_definitions['custom_dataset'] = custom_schema
```

## 📊 Dashboard Features

### Overview Section
- **System Health**: Overall monitoring status
- **Key Metrics**: Quality scores, drift counts, data completeness
- **Recent Alerts**: Latest issues requiring attention
- **Quick Actions**: Run monitoring, refresh data

### Feature Drift Analysis
- **Drift Timeline**: Historical drift detection results
- **Feature Heatmap**: Visual representation of drift patterns
- **Individual Feature Analysis**: Detailed drift scores and PSI values
- **Threshold Configuration**: Adjustable sensitivity settings

### Data Quality Monitoring
- **Quality Score Trends**: Track data health over time
- **Violation Breakdown**: Schema, completeness, and consistency issues
- **Grade Distribution**: A-F quality grade analysis
- **Metric Deep Dive**: Missing data, outliers, duplicates

### Alert Management
- **Active Alerts**: Current issues requiring attention
- **Alert History**: Historical alert patterns
- **Suppression Rules**: Manage notification frequency
- **Resolution Tracking**: Monitor issue resolution

### Historical Trends
- **Combined Timeline**: Integrated view of quality and drift
- **Pattern Analysis**: Identify recurring issues
- **Trend Statistics**: Statistical analysis of monitoring data
- **Performance Correlation**: Link monitoring to model performance

## 🔧 Advanced Configuration

### Custom Drift Detection

```python
class CustomDriftDetector:
    def __init__(self, monitor):
        self.monitor = monitor
    
    def detect_custom_drift(self, feature_data, baseline_data):
        # Implement your custom drift detection logic
        # Return DriftResult object
        pass

# Integrate with main monitor
monitor = FeatureMonitor()
monitor.custom_detector = CustomDriftDetector(monitor)
```

### Alert Suppression

```python
from enhanced_alert_integration import FeatureAlertManager

alert_manager = FeatureAlertManager()

# Suppress specific alert for 24 hours
alert_manager.suppress_alert(
    alert_type='FEATURE_DRIFT',
    feature_name='avg_points',
    hours=24,
    reason='Expected drift due to gameweek transition'
)
```

### Database Customization

```python
# Custom database path
monitor = FeatureMonitor(db_path="custom_monitoring.db")

# Custom table schemas can be modified in init_database() method
```

## 📈 Performance Monitoring

### Key Metrics to Track

1. **Drift Detection Rate**: Percentage of features showing drift
2. **Quality Score Trends**: Overall data health progression
3. **Alert Resolution Time**: How quickly issues are addressed
4. **False Positive Rate**: Accuracy of drift detection
5. **Data Completeness**: Percentage of complete records

### Monitoring Best Practices

1. **Regular Baseline Updates**: Refresh reference data periodically
2. **Seasonal Adjustments**: Account for FPL season patterns
3. **Feature Importance Weighting**: Focus on critical features
4. **Automated Responses**: Implement auto-remediation for common issues
5. **Documentation**: Maintain logs of all configuration changes

## 🚨 Troubleshooting

### Common Issues

#### High False Positive Rate
```python
# Adjust sensitivity thresholds
config = FeatureDriftConfig(
    ks_threshold=0.01,  # More strict (fewer false positives)
    psi_threshold=0.2   # Less sensitive
)
```

#### Missing Database Tables
```bash
# Remove and recreate databases
rm feature_monitoring.db data_quality_monitoring.db feature_alerts.db
python feature_monitoring.py  # Recreates tables
```

#### Dashboard Connection Issues
```python
# Check database connections
import sqlite3
conn = sqlite3.connect('feature_monitoring.db')
print(conn.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall())
conn.close()
```

#### Alert System Not Working
```bash
# Check environment variables
echo $EMAIL_USER
echo $SLACK_WEBHOOK_URL

# Test email configuration
python -c "from alert_system import AlertManager; AlertManager().send_email_alert({'type': 'TEST', 'message': 'Test', 'severity': 'LOW', 'timestamp': datetime.now()})"
```

### Performance Optimization

1. **Database Indexing**: Add indexes on frequently queried columns
2. **Data Sampling**: Use representative samples for large datasets
3. **Parallel Processing**: Run feature analysis in parallel
4. **Caching**: Cache expensive calculations

## 🔗 Integration with Existing Systems

### MLOps Pipeline Integration

```python
# Add to existing automated retraining pipeline
from enhanced_alert_integration import FeatureAlertManager

def retrain_with_monitoring():
    # Load new data
    df = load_latest_data()
    
    # Run feature monitoring
    alert_manager = FeatureAlertManager()
    alert_manager.run_comprehensive_feature_monitoring()
    
    # Proceed with retraining if no critical issues
    # ... existing retraining logic
```

### CI/CD Integration

```yaml
# GitHub Actions example
name: Feature Monitoring
on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours

jobs:
  monitor:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Feature Monitoring
        run: python enhanced_alert_integration.py
      - name: Check Exit Code
        run: exit $?
```

## 📚 API Reference

### FeatureMonitor Class

```python
class FeatureMonitor:
    def __init__(self, db_path: str, config: FeatureDriftConfig)
    def calculate_feature_statistics(self, df: pd.DataFrame) -> Dict[str, FeatureStats]
    def detect_feature_drift(self, df: pd.DataFrame) -> Dict[str, DriftResult]
    def run_comprehensive_monitoring(self, df: pd.DataFrame) -> Dict[str, Any]
```

### DataQualityMonitor Class

```python
class DataQualityMonitor:
    def __init__(self, db_path: str)
    def validate_schema(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]
    def calculate_quality_metrics(self, df: pd.DataFrame, dataset_name: str) -> QualityMetrics
    def run_comprehensive_quality_check(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]
```

### FeatureAlertManager Class

```python
class FeatureAlertManager:
    def __init__(self, config_file: str)
    def check_feature_drift_alerts(self) -> List[Dict[str, Any]]
    def check_data_quality_alerts(self) -> List[Dict[str, Any]]
    def run_comprehensive_feature_monitoring(self) -> None
    def suppress_alert(self, alert_type: str, hours: int, reason: str) -> None
```

## 🤝 Contributing

### Development Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd football-analytics-2025

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 *.py
black *.py
```

### Adding New Features

1. Create feature branch
2. Implement changes with tests
3. Update documentation
4. Submit pull request

### Reporting Issues

Please include:
- Error messages and stack traces
- Configuration settings
- Sample data (anonymized)
- Steps to reproduce

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Fantasy Premier League API**: Data source for player statistics
- **scikit-learn**: Statistical analysis tools
- **Streamlit**: Dashboard framework
- **Plotly**: Visualization library

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Email: support@fpl-analytics.com
- Documentation: See README files in each module

---

*Last updated: 2024-01-XX*
*Version: 1.0.0*