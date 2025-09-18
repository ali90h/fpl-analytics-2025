# FPL MLOps Pipeline - Complete Production System

A comprehensive MLOps pipeline for Fantasy Premier League model management, featuring automated retraining, performance monitoring, alerting, and A/B testing.

## 🎯 Features

### 🤖 Automated Retraining
- **Weekly Schedule**: Automatically retrain models every Monday at 23:00
- **Intelligent Updates**: Only retrain when new data is available
- **Health Validation**: Comprehensive model health checks after retraining
- **Email Notifications**: Get notified of retraining success/failure
- **Rollback Support**: Automatic backup and restore capabilities

### 📊 Performance Monitoring
- **Real-time Dashboard**: Streamlit-based monitoring interface
- **Key Metrics**: RMSE, accuracy, prediction volume tracking
- **Performance Drift Detection**: Automatic detection of model degradation
- **Historical Trends**: Track performance over time
- **System Health**: Monitor data freshness and system status

### 🚨 Automated Alerting
- **Smart Thresholds**: Configurable performance thresholds
- **Multiple Channels**: Email, Slack, Discord notifications
- **Severity Levels**: LOW, MEDIUM, HIGH, CRITICAL alert levels
- **Drift Detection**: Alert on significant performance changes
- **System Monitoring**: Data freshness and pipeline health alerts

### 🧪 A/B Testing Framework
- **Model Comparison**: Test different model versions
- **Statistical Analysis**: Proper significance testing
- **Traffic Splitting**: Configurable percentage-based routing
- **Performance Tracking**: Track success metrics for each variant
- **Automated Analysis**: Generate test results with recommendations

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Setup MLOps environment
python mlops_integration.py setup
```

### 2. Configure Settings
Edit `mlops_config.json`:
```json
{
  "email_settings": {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "alert_recipients": ["your-email@example.com"]
  },
  "alert_thresholds": {
    "rmse_threshold": 0.3,
    "accuracy_threshold": 80.0,
    "drift_threshold": 5.0
  }
}
```

### 3. Set Environment Variables
```bash
# For email notifications
export EMAIL_USER="your-email@gmail.com"
export EMAIL_PASSWORD="your-app-password"

# For Slack notifications (optional)
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."

# For Discord notifications (optional)
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
```

### 4. Start Monitoring
```bash
# Run health check
python mlops_integration.py health-check

# Start monitoring dashboard
python mlops_integration.py dashboard

# Run monitoring cycle
python mlops_integration.py monitor
```

## 📋 Available Commands

### MLOps Pipeline Management
```bash
# Setup complete MLOps environment
python mlops_integration.py setup

# Run system health check
python mlops_integration.py health-check

# Execute monitoring cycle
python mlops_integration.py monitor

# Start performance dashboard
python mlops_integration.py dashboard

# Generate status report
python mlops_integration.py report

# Create A/B test
python mlops_integration.py ab-test

# Setup crontab automation
python mlops_integration.py crontab
```

### Crontab Management
```bash
# Setup automated retraining
./setup_crontab.sh setup

# Check current status
./setup_crontab.sh status

# Test retraining script
./setup_crontab.sh test

# Remove automation
./setup_crontab.sh remove
```

### Individual Components
```bash
# Run automated retraining
python automated_retraining.py

# Run alert monitoring
python alert_system.py

# Start performance dashboard
streamlit run performance_dashboard.py

# Run A/B testing
python ab_testing_framework.py
```

## 🏗️ Architecture

### Component Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Pipeline │    │  Model Training │    │   Predictions   │
│                 │────│                 │────│                 │
│ • FPL API       │    │ • XGBoost       │    │ • Player Points │
│ • Data Cleaning │    │ • Feature Eng   │    │ • GW Scores     │
│ • Validation    │    │ • Validation    │    │ • Optimization  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────────────────────────────────────────┐
         │              MLOps Pipeline                         │
         │                                                     │
         │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
         │  │ Monitoring  │  │   Alerts    │  │ A/B Testing │ │
         │  │             │  │             │  │             │ │
         │  │ • Dashboard │  │ • Email     │  │ • Traffic   │ │
         │  │ • Metrics   │  │ • Slack     │  │   Splitting │ │
         │  │ • Health    │  │ • Discord   │  │ • Analysis  │ │
         │  └─────────────┘  └─────────────┘  └─────────────┘ │
         │                                                     │
         │  ┌─────────────┐                 ┌─────────────┐   │
         │  │ Retraining  │                 │   Storage   │   │
         │  │             │                 │             │   │
         │  │ • Weekly    │                 │ • SQLite    │   │
         │  │   Schedule  │                 │ • Logs      │   │
         │  │ • Validation│                 │ • Models    │   │
         │  └─────────────┘                 └─────────────┘   │
         └─────────────────────────────────────────────────────┘
```

### Data Flow
1. **Data Collection**: FPL API → Raw Data
2. **Processing**: Cleaning → Feature Engineering → Validation
3. **Training**: Model Training → Validation → Deployment
4. **Monitoring**: Performance Tracking → Alert Generation
5. **Retraining**: Weekly Automation → Model Updates
6. **Testing**: A/B Framework → Performance Comparison

## 📊 Monitoring Dashboard

The Streamlit dashboard provides:

### Key Metrics Cards
- **Current RMSE**: Latest model accuracy
- **Accuracy %**: Prediction accuracy percentage
- **Performance Drift**: Change in performance over time
- **Predictions Made**: Volume of recent predictions

### Performance Charts
- **Accuracy Trends**: RMSE and accuracy over time
- **Error Distribution**: Histogram of prediction errors
- **System Health**: Component status indicators

### Detailed Views
- **Metrics Table**: Recent performance metrics
- **Alerts Panel**: Current and recent alerts
- **Health Status**: Model, data, and automation status

## 🚨 Alert System

### Alert Types
- **Performance Alerts**: RMSE/accuracy thresholds
- **Drift Alerts**: Significant performance changes
- **System Alerts**: Data freshness, model availability
- **Volume Alerts**: Low prediction activity

### Notification Channels
- **Email**: SMTP-based notifications
- **Slack**: Webhook-based alerts
- **Discord**: Rich embed notifications
- **Database**: All alerts logged for analysis

### Configuration
```json
{
  "alert_thresholds": {
    "rmse_threshold": 0.3,
    "accuracy_threshold": 80.0,
    "drift_threshold": 5.0,
    "prediction_count_threshold": 100
  }
}
```

## 🧪 A/B Testing

### Test Setup
```python
from ab_testing_framework import ABTestManager, ABTestConfig

config = ABTestConfig(
    test_name="model_v2_comparison",
    description="Test new feature engineering approach",
    traffic_split=0.1,  # 10% to variant B
    model_a_path="models/current_model.joblib",
    model_b_path="models/experimental_model.joblib",
    success_metric="absolute_error"
)

manager = ABTestManager()
manager.create_test(config)
manager.start_test("model_v2_comparison")
```

### Analysis
- **Statistical Testing**: Mann-Whitney U, t-tests
- **Effect Size**: Cohen's d calculation
- **Confidence Intervals**: 95% confidence bounds
- **Power Analysis**: Statistical power calculation
- **Recommendations**: Automated test conclusions

## ⏰ Automated Scheduling

### Crontab Setup
The system automatically configures crontab for:
- **Weekly Retraining**: Every Monday at 23:00
- **Log Rotation**: Automatic log management
- **Health Checks**: Periodic system validation

### Manual Scheduling
```bash
# Every Monday at 23:00
0 23 * * 1 cd /path/to/project && python automated_retraining.py

# Every 6 hours for monitoring
0 */6 * * * cd /path/to/project && python alert_system.py
```

## 📁 File Structure

```
football-analytics-2025/
├── mlops_integration.py          # Main MLOps coordinator
├── automated_retraining.py       # Weekly retraining system
├── alert_system.py              # Alert monitoring
├── performance_dashboard.py      # Streamlit dashboard
├── ab_testing_framework.py       # A/B testing system
├── setup_crontab.sh             # Crontab automation
├── mlops_config.json            # Configuration
├── fpl_predictor.py             # Enhanced predictor
├── model_validation_suite.py    # Validation tests
├── logs/                        # System logs
├── models/                      # Model storage
├── reports/                     # Generated reports
├── data/                        # Data storage
└── ab_tests/                    # A/B test data
```

## 🔧 Configuration Files

### mlops_config.json
Main configuration for all MLOps components

### alert_config.json
Specific alert thresholds and settings

### Environment Variables
- `EMAIL_USER`: SMTP username
- `EMAIL_PASSWORD`: SMTP password
- `SLACK_WEBHOOK_URL`: Slack integration
- `DISCORD_WEBHOOK_URL`: Discord integration

## 📈 Performance Metrics

### Model Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **Accuracy**: Predictions within threshold
- **R²**: Coefficient of determination

### System Metrics
- **Prediction Volume**: Daily prediction count
- **Response Time**: Prediction latency
- **Data Freshness**: Age of training data
- **Model Age**: Time since last training

### Business Metrics
- **User Engagement**: Dashboard usage
- **A/B Test Coverage**: Traffic split coverage
- **Alert Response**: Time to resolution
- **System Uptime**: Availability percentage

## 🛠️ Troubleshooting

### Common Issues

#### Crontab Not Working
```bash
# Check crontab status
./setup_crontab.sh status

# Test retraining script
./setup_crontab.sh test

# Check logs
tail -f logs/cron_retraining.log
```

#### Email Alerts Not Sending
```bash
# Check environment variables
echo $EMAIL_USER
echo $EMAIL_PASSWORD

# Test SMTP connection
python -c "
import smtplib
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
"
```

#### Dashboard Not Loading
```bash
# Install Streamlit
pip install streamlit

# Check port availability
lsof -i :8501

# Start dashboard manually
streamlit run performance_dashboard.py
```

#### Database Errors
```bash
# Check database files
ls -la *.db

# Validate database schema
sqlite3 model_performance.db ".schema"
```

### Health Check Commands
```bash
# Full system health check
python mlops_integration.py health-check

# Component-specific checks
python alert_system.py
python ab_testing_framework.py
python automated_retraining.py
```

## 📞 Support

### Logs Location
- **MLOps Pipeline**: `logs/mlops_pipeline.log`
- **Retraining**: `logs/retraining.log`
- **Alerts**: `logs/alerts.log`
- **A/B Testing**: `logs/ab_testing.log`
- **Cron Jobs**: `logs/cron_retraining.log`

### Monitoring Commands
```bash
# Watch all logs
tail -f logs/*.log

# Monitor system resources
top -p $(pgrep -f "python.*mlops")

# Check disk usage
df -h
du -sh logs/ models/ data/
```

## 🚀 Production Deployment

### Pre-deployment Checklist
- [ ] Configure email credentials
- [ ] Set alert thresholds
- [ ] Test crontab setup
- [ ] Validate model paths
- [ ] Check disk space
- [ ] Configure firewalls (port 8501)
- [ ] Set up log rotation
- [ ] Test all notification channels

### Deployment Steps
1. **Environment Setup**: Run `python mlops_integration.py setup`
2. **Configuration**: Update `mlops_config.json`
3. **Testing**: Run `python mlops_integration.py health-check`
4. **Automation**: Execute `./setup_crontab.sh setup`
5. **Monitoring**: Start `python mlops_integration.py dashboard`

### Maintenance
- **Weekly**: Review performance reports
- **Monthly**: Update alert thresholds
- **Quarterly**: Review A/B test results
- **Yearly**: System architecture review

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📧 Contact

For questions and support, please check the logs first, then create an issue in the repository.