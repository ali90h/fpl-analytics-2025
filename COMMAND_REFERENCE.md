# 🛠️ FPL Analytics 2025 - Complete Command Reference Guide

## 📋 **Available Commands**

### **🎯 Personalized Team Analysis**
```bash
# Analyze your current team with ML predictions
python fpl_predictor.py analyze-team

# Get intelligent transfer suggestions based on fixtures  
python fpl_predictor.py suggest-transfers --gameweeks 5-10

# Fetch latest team data from FPL API
python fpl_predictor.py fetch-team --team-id 5135491
# Shows starting XI, bench, predicted points, bank balance, and transfers made
```

### **🔮 Player Predictions**
```bash
# Predict points for a specific player
python fpl_predictor.py predict --player "M.Salah"
# Returns XGBoost model prediction with confidence interval

# Get detailed player analysis with all features
python fpl_predictor.py predict --player "Haaland" --verbose
# Shows feature breakdown and model reasoning
```

### **🏆 Top Player Rankings**
```bash
# Get top players across all positions (default: top 20)
python fpl_predictor.py top-picks
# Returns highest predicted point scorers with value analysis

# Get top players by position
python fpl_predictor.py top-picks --position MID --limit 10
# Filter by: GKP, DEF, MID, FWD

# Find best value players (points per £1m)
python fpl_predictor.py top-picks --sort-by value --limit 15
# Sorts by predicted points per million spent
```

### **⚽ Team Optimization**
```bash
# Build optimal 11-player team within budget
python fpl_predictor.py optimize-team --budget 100.0
# Creates best possible XI under budget constraints

# Optimize with formation constraints
python fpl_predictor.py optimize-team --budget 100.0 --formation "3-5-2"
# Available formations: 3-4-3, 3-5-2, 4-3-3, 4-4-2, 4-5-1, 5-3-2, 5-4-1

# Optimize starting XI only (no bench)
python fpl_predictor.py optimize-team --budget 83.0 --starting-xi-only
# Focuses budget on 11 players for maximum points
```

### **📅 Fixture Analysis**
```bash
# Analyze team fixture difficulty for gameweek range
python fpl_predictor.py fixtures --gameweeks 5-10
# Shows Easy/Medium/Hard fixture classifications for all teams

# Get player picks from teams with favorable fixtures
python fpl_predictor.py fixture-picks --gameweeks 5-10
# Returns top players from teams with easiest upcoming fixtures

# Position-specific fixture picks
python fpl_predictor.py fixture-picks --gameweeks 5-10 --position DEF --limit 8
# Combines fixture analysis with position filtering

# Short-term fixture planning (next 3 gameweeks)
python fpl_predictor.py fixtures --gameweeks 4-6
# Ideal for weekly transfer decisions
```

### **🎯 Strategic Planning Commands**
```bash
# Wildcard team planning
python fpl_predictor.py optimize-team --budget 100.0 --formation "3-4-3"
python fpl_predictor.py fixture-picks --gameweeks 8-13 --limit 20
# Combine optimization with fixture analysis for wildcard weeks

# Captain choice analysis
python fpl_predictor.py fixture-picks --gameweeks 5-6 --position MID --limit 5
python fpl_predictor.py fixture-picks --gameweeks 5-6 --position FWD --limit 5
# Find best captaincy options from easy fixture teams

# Transfer planning workflow
python fpl_predictor.py fixtures --gameweeks 6-11
python fpl_predictor.py fixture-picks --gameweeks 6-11 --position MID --limit 8
python fpl_predictor.py predict --player "Current Player Name"
# Compare current players vs fixture-favorable alternatives

# Bench boost strategy
python fpl_predictor.py fixture-picks --gameweeks 7-7 --position DEF --limit 10
# Find cheap players from teams with good single gameweek fixtures
```

### **📊 Data Analysis Commands**
```bash
# Compare multiple players
python fpl_predictor.py predict --player "Salah"
python fpl_predictor.py predict --player "Son"
python fpl_predictor.py predict --player "Sterling"
# Run sequential predictions for comparison

# Analyze team strength trends
python fpl_predictor.py fixtures --gameweeks 1-38
# Full season fixture difficulty overview

# Position depth analysis
python fpl_predictor.py top-picks --position GKP --limit 10
python fpl_predictor.py top-picks --position DEF --limit 15
python fpl_predictor.py top-picks --position MID --limit 20
python fpl_predictor.py top-picks --position FWD --limit 12
# Understand player options by position
```

## 🎮 **Command Parameters**

### **Common Options:**
- `--gameweeks X-Y`: Specify gameweek range (e.g., 5-10, 1-38)
- `--position POS`: Filter by position (GKP/DEF/MID/FWD)
- `--limit N`: Number of results to return
- `--budget X.X`: Budget constraint in £millions
- `--formation "X-Y-Z"`: Team formation (3-4-3, 4-3-3, etc.)
- `--verbose`: Show detailed analysis
- `--sort-by value`: Sort by points per £1m instead of raw points
- `--starting-xi-only`: Optimize 11 players only (no bench)

### **Example Workflows:**

#### **Weekly Transfer Decision:**
```bash
# 1. Check current team strength
python fpl_predictor.py fixtures --gameweeks 5-7

# 2. Find alternatives in your problem positions
python fpl_predictor.py fixture-picks --gameweeks 5-7 --position MID --limit 8

# 3. Compare specific players
python fpl_predictor.py predict --player "Current Player"
python fpl_predictor.py predict --player "Transfer Target"
```

#### **Wildcard Planning:**
```bash
# 1. Identify best fixture periods
python fpl_predictor.py fixtures --gameweeks 8-13

# 2. Build optimal team for that period
python fpl_predictor.py optimize-team --budget 100.0 --formation "3-4-3"

# 3. Get fixture-friendly alternatives
python fpl_predictor.py fixture-picks --gameweeks 8-13 --limit 15
```

#### **Captain Selection:**
```bash
# 1. Find teams with easiest next fixture
python fpl_predictor.py fixtures --gameweeks 5-5

# 2. Get top attackers from those teams
python fpl_predictor.py fixture-picks --gameweeks 5-5 --position MID --limit 5
python fpl_predictor.py fixture-picks --gameweeks 5-5 --position FWD --limit 5
```

## 🚀 **Pro Tips:**

1. **Combine Commands**: Use multiple commands together for comprehensive analysis
2. **Regular Updates**: Run weekly to adapt to form changes and injuries
3. **Budget Planning**: Use optimization to stay within FPL budget constraints
4. **Fixture Timing**: Plan transfers 2-3 gameweeks ahead using fixture analysis
5. **Value Hunting**: Use `--sort-by value` to find the best points per £1m
6. **Formation Testing**: Try different formations to maximize team potential

**Your complete FPL command arsenal for the 2025/26 season!** ⚽🏆