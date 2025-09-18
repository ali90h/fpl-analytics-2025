# ⚽ FPL Analytics 2025 - Maximum Accuracy Fantasy Premier League Predictions

**The ultimate terminal-based FPL analysis system with XGBoost machine learning and personalized team intelligence.**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![RMSE 0.827](https://img.shields.io/badge/RMSE-0.827-green.svg)](#)
[![Personalized](https://img.shields.io/badge/Team-5135491-purple.svg)](#)

---

## 🚀 **Quick Start**

```bash
# Activate environment
source .venv/bin/activate

# Analyze your current FPL team
python fpl_predictor.py analyze-team

# Get intelligent transfer suggestions  
python fpl_predictor.py suggest-transfers --gameweeks 5-10

# Predict any player's points
python fpl_predictor.py predict --player "Salah"
```

---

## 🎯 **What This System Does**

### **🧠 Maximum Accuracy ML Predictions**
- **0.827 RMSE** XGBoost model trained on 94 engineered features
- **Real-time analysis** of 2,960+ player records
- **Ensemble backup** with 4-model redundancy for reliability

### **🎯 Personalized Team Analysis**
- **Your actual FPL team** (ID: 5135491) with current predictions
- **Starting XI optimization** with 195.9 predicted points
- **Intelligent transfer suggestions** based on fixture difficulty
- **Budget-aware recommendations** within your £0.1m constraints

### **📅 Strategic Fixture Intelligence**
- **3-5 gameweek planning** with team strength analysis
- **Easy/Medium/Hard classifications** for all 20 Premier League teams
- **Transfer timing optimization** for maximum point gains
- **Captain selection guidance** from fixture-favorable teams

---

## 🏆 **Core Features**

| Feature | Description | Command |
|---------|-------------|---------|
| **Team Analysis** | Analyze your current FPL squad with ML predictions | `python fpl_predictor.py analyze-team` |
| **Transfer Suggestions** | Get personalized recommendations based on fixtures | `python fpl_predictor.py suggest-transfers --gameweeks 5-10` |
| **Player Predictions** | Predict points for any Premier League player | `python fpl_predictor.py predict --player "Name"` |
| **Fixture Analysis** | Analyze team difficulty for upcoming gameweeks | `python fpl_predictor.py fixtures --gameweeks 5-10` |
| **Team Optimization** | Build optimal XI within budget constraints | `python fpl_predictor.py optimize-team --budget 100.0` |
| **Top Picks** | Find best players by position and value | `python fpl_predictor.py top-picks --position MID` |
| **Chip Strategy** | Automated recommendations for optimal chip timing | `python fpl_predictor.py chip-advice` |

---

## 📊 **Your Current Team Status**

```
📋 Team Analysis (GW4) - 195.9 Predicted Points
💰 Bank: £0.1m | 🔄 Transfers: 0 available

TOP PERFORMERS:
⭐ João Pedro (CHE) - 26.0 pts | Easy fixtures  
⭐ Van de Ven (TOT) - 25.5 pts | Easy fixtures
⭐ M.Salah (LIV) - 25.6 pts (C) | Easy fixtures

TRANSFER PRIORITIES:
⚠️ Wood (NFO) - 0.0 pts | Hard fixtures → Immediate transfer
⚠️ N.Williams (NFO) - 2.9 pts | Hard fixtures → Upgrade target
```

---

## 🎮 **Essential Commands**

### **Personalized Analysis**
```bash
# Analyze your current FPL team
python fpl_predictor.py analyze-team

# Get intelligent transfer suggestions
python fpl_predictor.py suggest-transfers --gameweeks 5-10

# Fetch latest team data from FPL API
python fpl_predictor.py fetch-team --team-id 5135491

# Get intelligent chip strategy recommendations
python fpl_predictor.py chip-advice

# Analyze optimal timing for specific chips
python fpl_predictor.py chip-timing --chip wildcard
```

### **Player & Team Research**  
```bash
# Individual player prediction
python fpl_predictor.py predict --player "Haaland"

# Best players by position
python fpl_predictor.py top-picks --position FWD --limit 8

# Optimal team building
python fpl_predictor.py optimize-team --budget 100.0 --formation "3-4-3"
```

### **Strategic Planning**
```bash
# Fixture difficulty analysis
python fpl_predictor.py fixtures --gameweeks 5-10

# Players from easy fixture teams
python fpl_predictor.py fixture-picks --gameweeks 5-10 --position MID

# Value-focused picks
python fpl_predictor.py top-picks --sort-by value --limit 15
```

---

## 📈 **Results & Intelligence**

### **🎯 Current Transfer Recommendations**
1. **Wood → Haaland**: +25.6 points, Save £6.2m
2. **N.Williams → Calafiori**: +24.1 points, Save £4.4m  
3. **Wood → João Pedro**: +26.0 points, Save £6.8m

### **📊 Fixture Advantage (GW5-10)**
- **🟢 Easy**: ARS (7.28), TOT (6.74), CHE (6.49), LIV (6.18)
- **🔴 Hard**: NFO (3.45), AVL (3.16), BHA (3.12) ← Avoid these teams

### **⭐ Your Squad Strengths**
- Strong fixture positioning with TOT, CHE, LIV assets
- Salah excellent captain choice with Liverpool's easy run
- Defensive value with Van de Ven and Muñoz

---

## 📁 **Project Structure**

```
football-analytics-2025/
├── 🤖 fpl_predictor.py              # Main terminal interface
├── 📊 01_comprehensive_model_development.ipynb  # ML training pipeline
├── 📈 data/
│   ├── enhanced_fpl_features.csv    # 94 engineered features
│   ├── current_team.json            # Your FPL team data  
│   └── fpl_data.db                  # Historical database
├── 🧠 models/production/            # Trained XGBoost models
├── 📚 Documentation/
│   ├── PROJECT_DIRECTORY.md         # Comprehensive guide
│   ├── COMMAND_REFERENCE.md         # All available commands
│   ├── FIXTURE_ANALYSIS_GUIDE.md    # Fixture difficulty usage
│   └── PERSONALIZED_ANALYSIS_GUIDE.md # Your team analysis
└── ⚙️ .venv/                        # Python 3.11 environment
```

---

## 🔧 **Technical Specifications**

### **Machine Learning Architecture**
- **Algorithm**: XGBoost Gradient Boosting Regressor
- **Performance**: 0.827 RMSE (maximum accuracy achieved)
- **Features**: 94 engineered features from FPL data
- **Training Set**: 2,960 player records across 4 gameweeks
- **Preprocessing**: Advanced feature engineering, label encoding

### **System Requirements**
- **Python**: 3.11 (optimal ML library compatibility)
- **Dependencies**: XGBoost, pandas, numpy, scikit-learn, requests
- **Environment**: Isolated virtual environment with all dependencies
- **Platform**: macOS/Linux/Windows compatible

---

## 💡 **Strategic Workflows**

### **Weekly Transfer Planning**
1. **Analyze current team** → Check predicted points for all players
2. **Review fixtures** → Identify teams with easy/hard upcoming games  
3. **Get transfer suggestions** → See data-driven recommendations
4. **Compare alternatives** → Validate specific player swaps
5. **Execute optimal transfer** → Make informed decision

### **Captain Selection**
1. **Check next gameweek fixtures** → Find easiest opponents
2. **Review attacking options** → Focus on MID/FWD from easy teams
3. **Compare your players** → See predictions for current squad
4. **Make data-driven choice** → Select highest predicted scorer

---

## 🏆 **Competitive Advantages**

✅ **Maximum Accuracy**: 0.827 RMSE beats industry standards  
✅ **Personalized Analysis**: Your actual team (5135491), not generic advice  
✅ **Fixture Intelligence**: 3-5 gameweek strategic planning  
✅ **Budget Optimization**: Works within your exact financial constraints  
✅ **Terminal Speed**: Instant results without web interface delays  
✅ **Transfer Timing**: Optimal moments for squad changes  

---

## 🚀 **Get Started Now**

```bash
# 1. Activate the environment
source .venv/bin/activate

# 2. See your current team analysis
python fpl_predictor.py analyze-team

# 3. Get your next transfer recommendations  
python fpl_predictor.py suggest-transfers --gameweeks 5-10

# 4. Check fixture difficulty for planning
python fpl_predictor.py fixtures --gameweeks 5-10
```

---

## 📚 **Documentation**

- **[Complete Project Directory](PROJECT_DIRECTORY.md)** - Comprehensive feature guide
- **[Command Reference](COMMAND_REFERENCE.md)** - All available commands  
- **[Fixture Analysis Guide](FIXTURE_ANALYSIS_GUIDE.md)** - Strategic fixture usage
- **[Personalized Analysis](PERSONALIZED_ANALYSIS_GUIDE.md)** - Your team insights

---

**Transform your FPL experience from guesswork to data-driven excellence!** 🏆⚽

*Powered by XGBoost ML • Personalized for Team 5135491 • Built for Championship Success* 🤖📊

## 📋 System Requirements

- Python 3.8+
- Virtual environment recommended
- Internet connection for API access

## 🚀 Installation & Setup

1. **Clone/Navigate to project directory**
   ```bash
   cd football-analytics-2025
   ```

2. **Set up Python virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install click pandas numpy requests
   ```

4. **Verify installation**
   ```bash
   python fpl_tool.py --help
   ```

## 🛠️ Commands Reference

### Basic Commands

#### `status` - System Overview
```bash
python fpl_tool.py status
```
Shows current data status, gameweeks collected, data quality metrics.

#### `collect` - Data Collection
```bash
# Collect current gameweek
python fpl_tool.py collect

# Collect specific gameweek
python fpl_tool.py collect --gameweek 5

# Force re-collection (overwrite existing)
python fpl_tool.py collect --gameweek 5 --force
```

#### `update` - Incremental Updates
```bash
# Update from current gameweek
python fpl_tool.py update

# Update specific range
python fpl_tool.py update --from-gameweek 3 --to-gameweek 5
```

### Analysis Commands

#### `analyze` - Player & Trend Analysis
```bash
# Position trends analysis
python fpl_tool.py analyze

# Specific player analysis
python fpl_tool.py analyze --player 123

# Detailed player analysis
python fpl_tool.py analyze --player 123 --detailed

# Gameweek top performers
python fpl_tool.py analyze --gameweek 5

# Find value picks
python fpl_tool.py analyze --value-picks --max-price 7.0
```

#### `predict` - Performance Predictions
```bash
# Player points prediction
python fpl_tool.py predict --player 123

# Compare multiple players
python fpl_tool.py predict --compare "123,456,789"
```

## 📊 Data Sources

### Primary: Official FPL API
- **Bootstrap data**: Teams, players, positions, prices
- **Live data**: Real-time gameweek performance
- **Player history**: Historical performance data
- **Rate limited**: Respectful API usage (2 seconds between requests)

### Secondary: Football-Data.org API
- **Match fixtures**: Premier League match schedules
- **Team data**: Squad information and statistics
- **Mock fallback**: When API key not configured

## �️ Data Organization

```
data/
├── organized/
│   └── 2025-26/
│       ├── fpl/              # Raw FPL data by gameweek
│       ├── matches/          # Match context data
│       ├── processed/        # Analysis-ready data
│       └── backups/          # Data backups
└── football_data_2025_26.db # SQLite database
```

### Database Schema
- **Players table**: Current player data with form metrics
- **Gameweek performances**: Historical performance by gameweek
- **Data quality**: Quality metrics and validation logs

## 🔄 Typical Workflow

### Season Start Setup
```bash
# 1. Check system status
python fpl_tool.py status

# 2. Collect initial data
python fpl_tool.py collect --gameweek 1

# 3. Verify collection
python fpl_tool.py status
```

### Weekly Updates (After Each Gameweek)
```bash
# 1. Update with latest gameweek
python fpl_tool.py update --from-gameweek 5

# 2. Analyze performance trends
python fpl_tool.py analyze

# 3. Find value picks for next gameweek
python fpl_tool.py analyze --value-picks --max-price 7.5

# 4. Compare transfer targets
python fpl_tool.py predict --compare "123,456,789"
```

### Transfer Decision Support
```bash
# Analyze current player
python fpl_tool.py analyze --player 123 --detailed

# Get prediction for upcoming gameweeks
python fpl_tool.py predict --player 123

# Compare with alternatives
python fpl_tool.py predict --compare "123,456"

# Check value picks in price range
python fpl_tool.py analyze --value-picks --max-price 8.0
```

## � Analysis Features

### Player Analysis
- **Points per game**: Average scoring rate
- **Consistency score**: Performance reliability
- **Form trend**: Recent performance direction (IMPROVING/STABLE/DECLINING)
- **Value rating**: Points per price efficiency
- **Injury risk**: Based on minutes played patterns
- **Recommendation**: BUY/HOLD/SELL guidance

### Position Trends
- **Average points by position**: Recent gameweek averages
- **High scorers count**: Players with 8+ points
- **Consistency metrics**: Position reliability
- **Trend analysis**: Position-wide performance direction

### Value Picks
- **Price filtering**: Find players under specified budget
- **Value scoring**: Comprehensive value assessment
- **Form consideration**: Recent performance weighting
- **Recommendation strength**: STRONG/MODERATE buy signals

### Predictions
- **Points prediction**: Expected performance
- **Confidence rating**: Prediction reliability
- **Form multipliers**: Recent trend adjustments
- **Transfer recommendations**: BUY/HOLD/SELL guidance

## ⚙️ Configuration

### API Keys (Optional)
Create `.env` file for Football-Data.org API:
```
FOOTBALL_DATA_API_KEY=your_api_key_here
```

### Season Configuration
Default season is 2025-26, can be changed:
```bash
python fpl_tool.py --season 2024-25 status
```

## 🔧 Troubleshooting

### Common Issues

#### Import Errors
```bash
pip install click pandas numpy requests
```

#### No Data Available
- Ensure internet connection
- Check if FPL API is accessible
- Verify season dates (tool optimized for 2025/26)

#### Low Data Quality
- Re-collect recent gameweeks: `python fpl_tool.py collect --gameweek X --force`
- Check API response status in logs

#### Analysis Shows No Results
- Ensure sufficient gameweek data collected
- Check if season has started (need performance data for analysis)

### Logging
Logs are displayed in terminal with timestamps:
- **INFO**: Normal operations
- **WARNING**: Data quality or API issues
- **ERROR**: Operation failures

## 🎯 Best Practices

### Data Collection
- **Collect immediately after gameweek completion** for most accurate data
- **Use `--force` sparingly** to avoid unnecessary API calls
- **Monitor data quality scores** in status output

### Analysis Workflow
- **Start with position trends** for general market overview
- **Use value picks** to identify budget-friendly options
- **Employ player comparison** for final transfer decisions
- **Check predictions** for captaincy and future planning

### Performance Optimization
- **Collect only needed gameweeks** to minimize storage
- **Use specific player analysis** instead of bulk operations
- **Monitor API rate limits** in logs

## � Data Quality

### Quality Metrics
- **Player data validation**: Required fields, price ranges, position validity
- **Performance data validation**: Points ranges, minutes validation
- **Overall quality score**: Percentage of valid records

### Quality Thresholds
- **High quality**: 80%+ validation pass rate
- **Medium quality**: 60-79% validation pass rate  
- **Low quality**: <60% validation pass rate

### Mock Data Fallbacks
When APIs are unavailable:
- FPL API: Basic structure with placeholder data
- Football-Data.org: Mock match fixtures and team data

## 🚀 Advanced Usage

### Batch Analysis
```bash
# Analyze multiple gameweeks
for gw in {1..5}; do
  python fpl_tool.py analyze --gameweek $gw
done
```

### Season-long Data Collection
```bash
# Collect full season (when available)
python fpl_tool.py update --from-gameweek 1 --to-gameweek 38
```

### Export Analysis Results
Results are displayed in terminal - redirect to files:
```bash
python fpl_tool.py analyze --value-picks > value_picks.txt
python fpl_tool.py analyze --gameweek 5 > gw5_analysis.txt
```

## 📞 Support

### Getting Help
```bash
# General help
python fpl_tool.py --help

# Command-specific help
python fpl_tool.py analyze --help
python fpl_tool.py collect --help
```

### System Information
```bash
# Check Python version
python --version

# Check installed packages
pip list

# Verify tool functionality
python fpl_tool.py status
```

---

**Built for 2025/26 Season** | **Terminal-focused** | **Official FPL Data** | **High Quality Analysis**