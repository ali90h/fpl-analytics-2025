#!/bin/bash
# Crontab Setup Script for FPL Model Retraining
# Sets up automated Monday night retraining at 23:00

# Script configuration
PROJECT_DIR="/Users/ali/football-analytics-2025"
PYTHON_PATH="$PROJECT_DIR/.venv/bin/python"
SCRIPT_PATH="$PROJECT_DIR/automated_retraining.py"
LOG_DIR="$PROJECT_DIR/logs"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to display current crontab
show_current_crontab() {
    echo "📋 Current crontab entries:"
    echo "=========================="
    crontab -l 2>/dev/null || echo "No crontab entries found"
    echo ""
}

# Function to add/update crontab entry
setup_crontab() {
    echo "⚙️ Setting up automated weekly retraining..."
    
    # Create temporary crontab file
    TEMP_CRON=$(mktemp)
    
    # Get existing crontab (excluding our entry)
    crontab -l 2>/dev/null | grep -v "fpl.*retraining" > "$TEMP_CRON"
    
    # Add our crontab entry for Monday 23:00
    echo "# FPL Model Retraining - Every Monday at 23:00" >> "$TEMP_CRON"
    echo "0 23 * * 1 cd $PROJECT_DIR && $PYTHON_PATH $SCRIPT_PATH >> $LOG_DIR/cron_retraining.log 2>&1" >> "$TEMP_CRON"
    echo "" >> "$TEMP_CRON"
    
    # Install the new crontab
    if crontab "$TEMP_CRON"; then
        echo "✅ Crontab updated successfully!"
        echo ""
        echo "📅 Scheduled: Every Monday at 23:00"
        echo "📝 Logs: $LOG_DIR/cron_retraining.log"
        echo "🔧 Script: $SCRIPT_PATH"
    else
        echo "❌ Failed to update crontab"
        rm -f "$TEMP_CRON"
        exit 1
    fi
    
    # Cleanup
    rm -f "$TEMP_CRON"
}

# Function to remove crontab entry
remove_crontab() {
    echo "🗑️ Removing automated retraining from crontab..."
    
    # Create temporary crontab file
    TEMP_CRON=$(mktemp)
    
    # Get existing crontab (excluding our entry)
    crontab -l 2>/dev/null | grep -v "fpl.*retraining" | grep -v "FPL Model Retraining" > "$TEMP_CRON"
    
    # Install the new crontab
    if crontab "$TEMP_CRON"; then
        echo "✅ Crontab entry removed successfully!"
    else
        echo "❌ Failed to remove crontab entry"
        rm -f "$TEMP_CRON"
        exit 1
    fi
    
    # Cleanup
    rm -f "$TEMP_CRON"
}

# Function to test the retraining script
test_retraining() {
    echo "🧪 Testing automated retraining script..."
    echo "========================================"
    
    if [ ! -f "$PYTHON_PATH" ]; then
        echo "❌ Python virtual environment not found at: $PYTHON_PATH"
        echo "💡 Please ensure the virtual environment is set up correctly"
        exit 1
    fi
    
    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "❌ Retraining script not found at: $SCRIPT_PATH"
        exit 1
    fi
    
    echo "📍 Running test execution..."
    cd "$PROJECT_DIR"
    
    # Run the script in test mode
    if "$PYTHON_PATH" "$SCRIPT_PATH"; then
        echo "✅ Test execution completed successfully!"
    else
        echo "❌ Test execution failed!"
        echo "💡 Check the script and dependencies"
        exit 1
    fi
}

# Function to show help
show_help() {
    echo "🤖 FPL Automated Retraining - Crontab Setup"
    echo "==========================================="
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  setup     Set up automated Monday night retraining"
    echo "  remove    Remove automated retraining from crontab"
    echo "  status    Show current crontab status"
    echo "  test      Test the retraining script"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup     # Set up weekly automation"
    echo "  $0 test      # Test retraining script"
    echo "  $0 status    # Check current crontab"
    echo ""
}

# Function to show status
show_status() {
    echo "📊 FPL Retraining Automation Status"
    echo "==================================="
    echo ""
    
    # Check if crontab entry exists
    if crontab -l 2>/dev/null | grep -q "fpl.*retraining"; then
        echo "✅ Automated retraining is ENABLED"
        echo ""
        echo "📅 Schedule:"
        crontab -l 2>/dev/null | grep -A1 -B1 "fpl.*retraining"
    else
        echo "❌ Automated retraining is NOT ENABLED"
        echo "💡 Run '$0 setup' to enable automation"
    fi
    
    echo ""
    echo "📁 Project Directory: $PROJECT_DIR"
    echo "🐍 Python Path: $PYTHON_PATH"
    echo "📝 Log Directory: $LOG_DIR"
    
    # Check if recent logs exist
    if [ -f "$LOG_DIR/cron_retraining.log" ]; then
        echo ""
        echo "📜 Recent log entries:"
        echo "====================="
        tail -n 10 "$LOG_DIR/cron_retraining.log"
    fi
}

# Main script logic
case "${1:-help}" in
    "setup")
        show_current_crontab
        setup_crontab
        echo ""
        echo "🎯 Next steps:"
        echo "1. Test the setup: $0 test"
        echo "2. Check status: $0 status"
        echo "3. Monitor logs: tail -f $LOG_DIR/cron_retraining.log"
        ;;
    "remove")
        remove_crontab
        ;;
    "status")
        show_status
        ;;
    "test")
        test_retraining
        ;;
    "help"|"--help"|"-h")
        show_help
        ;;
    *)
        echo "❌ Unknown option: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
