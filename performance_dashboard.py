"""
Performance Monitoring Dashboard for FPL Model
Real-time monitoring and visualization of model performance metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class PerformanceMonitor:
    """Monitor and track model performance metrics over time"""
    
    def __init__(self, db_path: str = "model_performance.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize performance tracking database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                model_version TEXT,
                data_version TEXT,
                additional_info TEXT
            )
        ''')
        
        # Model predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                player_id INTEGER NOT NULL,
                predicted_points REAL NOT NULL,
                actual_points REAL,
                absolute_error REAL,
                gameweek INTEGER,
                model_version TEXT
            )
        ''')
        
        # System alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                resolved BOOLEAN DEFAULT FALSE,
                resolved_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_metric(self, metric_name: str, value: float, 
                   model_version: str = None, additional_info: str = None):
        """Log a performance metric"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance_metrics 
            (timestamp, metric_name, metric_value, model_version, additional_info)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now().isoformat(), metric_name, value, model_version, additional_info))
        
        conn.commit()
        conn.close()
    
    def log_predictions(self, predictions_df: pd.DataFrame, gameweek: int, 
                       model_version: str = None):
        """Log batch predictions for performance tracking"""
        conn = sqlite3.connect(self.db_path)
        
        predictions_df['timestamp'] = datetime.now().isoformat()
        predictions_df['gameweek'] = gameweek
        predictions_df['model_version'] = model_version
        
        predictions_df.to_sql('predictions', conn, if_exists='append', index=False)
        conn.close()
    
    def log_alert(self, alert_type: str, severity: str, message: str):
        """Log a system alert"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (timestamp, alert_type, severity, message)
            VALUES (?, ?, ?, ?)
        ''', (datetime.now().isoformat(), alert_type, severity, message))
        
        conn.commit()
        conn.close()
    
    def get_metrics(self, metric_name: str = None, days: int = 30) -> pd.DataFrame:
        """Retrieve performance metrics"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM performance_metrics 
            WHERE timestamp >= datetime('now', '-{} days')
        '''.format(days)
        
        if metric_name:
            query += f" AND metric_name = '{metric_name}'"
        
        query += " ORDER BY timestamp DESC"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def get_recent_alerts(self, days: int = 7) -> pd.DataFrame:
        """Get recent alerts"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM alerts 
            WHERE timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def calculate_performance_drift(self, metric_name: str, 
                                   lookback_days: int = 7) -> float:
        """Calculate performance drift over time"""
        df = self.get_metrics(metric_name, days=lookback_days * 2)
        
        if len(df) < 10:
            return 0.0
        
        # Split into recent and historical
        mid_point = len(df) // 2
        recent = df.iloc[:mid_point]['metric_value'].mean()
        historical = df.iloc[mid_point:]['metric_value'].mean()
        
        # Calculate percentage change
        if historical != 0:
            drift = ((recent - historical) / historical) * 100
        else:
            drift = 0.0
        
        return drift

class DashboardApp:
    """Streamlit dashboard for performance monitoring"""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.setup_page()
    
    def setup_page(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="FPL Model Performance Dashboard",
            page_icon="⚽",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def run(self):
        """Main dashboard application"""
        st.title("⚽ FPL Model Performance Dashboard")
        st.markdown("---")
        
        # Sidebar controls
        with st.sidebar:
            st.header("📊 Dashboard Controls")
            
            # Time range selector
            time_range = st.selectbox(
                "Time Range",
                ["Last 7 days", "Last 30 days", "Last 90 days"],
                index=1
            )
            
            days_map = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}
            selected_days = days_map[time_range]
            
            # Refresh data
            if st.button("🔄 Refresh Data"):
                st.rerun()
            
            # Auto-refresh
            auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
            if auto_refresh:
                st.rerun()
        
        # Main dashboard layout
        col1, col2, col3, col4 = st.columns(4)
        
        # Key metrics cards
        self.render_metric_cards(col1, col2, col3, col4, selected_days)
        
        # Performance charts
        st.markdown("---")
        st.header("📈 Performance Trends")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            self.render_accuracy_trend(selected_days)
        
        with chart_col2:
            self.render_prediction_distribution()
        
        # Detailed metrics
        st.markdown("---")
        st.header("🔍 Detailed Metrics")
        
        detail_col1, detail_col2 = st.columns(2)
        
        with detail_col1:
            self.render_metric_table(selected_days)
        
        with detail_col2:
            self.render_alerts_panel(selected_days)
        
        # System health
        st.markdown("---")
        st.header("🏥 System Health")
        self.render_health_status()
    
    def render_metric_cards(self, col1, col2, col3, col4, days: int):
        """Render key performance metric cards"""
        
        # Get recent metrics
        rmse_df = self.monitor.get_metrics("rmse", days)
        accuracy_df = self.monitor.get_metrics("accuracy", days)
        drift_df = self.monitor.get_metrics("performance_drift", days)
        
        # RMSE Card
        with col1:
            if not rmse_df.empty:
                current_rmse = rmse_df.iloc[0]['metric_value']
                rmse_change = self.calculate_metric_change(rmse_df)
                
                st.metric(
                    label="🎯 Current RMSE",
                    value=f"{current_rmse:.3f}",
                    delta=f"{rmse_change:+.3f}" if rmse_change else None
                )
            else:
                st.metric("🎯 Current RMSE", "No data")
        
        # Accuracy Card
        with col2:
            if not accuracy_df.empty:
                current_acc = accuracy_df.iloc[0]['metric_value']
                acc_change = self.calculate_metric_change(accuracy_df)
                
                st.metric(
                    label="✅ Accuracy %",
                    value=f"{current_acc:.1f}%",
                    delta=f"{acc_change:+.1f}%" if acc_change else None
                )
            else:
                st.metric("✅ Accuracy %", "No data")
        
        # Performance Drift Card
        with col3:
            current_drift = self.monitor.calculate_performance_drift("rmse", 7)
            drift_color = "🔴" if abs(current_drift) > 5 else "🟢"
            
            st.metric(
                label=f"{drift_color} Performance Drift",
                value=f"{current_drift:+.1f}%",
                delta=None
            )
        
        # Predictions Count Card
        with col4:
            conn = sqlite3.connect(self.monitor.db_path)
            pred_count = pd.read_sql_query(
                f"SELECT COUNT(*) as count FROM predictions WHERE timestamp >= datetime('now', '-{days} days')",
                conn
            )
            conn.close()
            
            total_predictions = pred_count.iloc[0]['count'] if not pred_count.empty else 0
            
            st.metric(
                label="📊 Predictions Made",
                value=f"{total_predictions:,}",
                delta=None
            )
    
    def calculate_metric_change(self, df: pd.DataFrame) -> Optional[float]:
        """Calculate change in metric over time"""
        if len(df) < 2:
            return None
        
        current = df.iloc[0]['metric_value']
        previous = df.iloc[1]['metric_value']
        
        return current - previous
    
    def render_accuracy_trend(self, days: int):
        """Render accuracy trend chart"""
        st.subheader("📈 Accuracy Trend")
        
        rmse_df = self.monitor.get_metrics("rmse", days)
        accuracy_df = self.monitor.get_metrics("accuracy", days)
        
        if not rmse_df.empty or not accuracy_df.empty:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('RMSE Over Time', 'Accuracy Over Time'),
                vertical_spacing=0.1
            )
            
            # RMSE trend
            if not rmse_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=rmse_df['timestamp'],
                        y=rmse_df['metric_value'],
                        mode='lines+markers',
                        name='RMSE',
                        line=dict(color='red')
                    ),
                    row=1, col=1
                )
            
            # Accuracy trend
            if not accuracy_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=accuracy_df['timestamp'],
                        y=accuracy_df['metric_value'],
                        mode='lines+markers',
                        name='Accuracy %',
                        line=dict(color='green')
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                height=400,
                showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance data available for the selected time range.")
    
    def render_prediction_distribution(self):
        """Render prediction error distribution"""
        st.subheader("📊 Prediction Distribution")
        
        conn = sqlite3.connect(self.monitor.db_path)
        predictions_df = pd.read_sql_query('''
            SELECT * FROM predictions 
            WHERE actual_points IS NOT NULL 
            AND timestamp >= datetime('now', '-30 days')
            ORDER BY timestamp DESC
            LIMIT 1000
        ''', conn)
        conn.close()
        
        if not predictions_df.empty:
            # Calculate errors
            predictions_df['error'] = (
                predictions_df['predicted_points'] - predictions_df['actual_points']
            )
            
            # Create histogram
            fig = px.histogram(
                predictions_df,
                x='error',
                nbins=30,
                title='Prediction Error Distribution',
                labels={'error': 'Prediction Error', 'count': 'Frequency'},
                color_discrete_sequence=['lightblue']
            )
            
            # Add vertical line at zero
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary stats
            error_stats = predictions_df['error'].describe()
            st.markdown(f"""
            **Error Statistics:**
            - Mean Error: {error_stats['mean']:.3f}
            - Std Dev: {error_stats['std']:.3f}
            - MAE: {predictions_df['error'].abs().mean():.3f}
            """)
        else:
            st.info("No prediction data available.")
    
    def render_metric_table(self, days: int):
        """Render detailed metrics table"""
        st.subheader("📋 Recent Metrics")
        
        all_metrics = self.monitor.get_metrics(days=days)
        
        if not all_metrics.empty:
            # Format the data
            display_df = all_metrics[['timestamp', 'metric_name', 'metric_value', 'model_version']]
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            display_df['metric_value'] = display_df['metric_value'].round(4)
            
            st.dataframe(
                display_df.head(20),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No metrics data available.")
    
    def render_alerts_panel(self, days: int):
        """Render alerts panel"""
        st.subheader("🚨 Recent Alerts")
        
        alerts_df = self.monitor.get_recent_alerts(days)
        
        if not alerts_df.empty:
            for _, alert in alerts_df.head(10).iterrows():
                severity_emoji = {
                    'LOW': '🟡',
                    'MEDIUM': '🟠', 
                    'HIGH': '🔴',
                    'CRITICAL': '🚨'
                }.get(alert['severity'], '📝')
                
                st.markdown(f"""
                {severity_emoji} **{alert['alert_type']}** ({alert['severity']})
                
                {alert['message']}
                
                *{alert['timestamp'].strftime('%Y-%m-%d %H:%M')}*
                
                ---
                """)
        else:
            st.success("🎉 No alerts in the selected time range!")
    
    def render_health_status(self):
        """Render system health status"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🔧 Model Status")
            try:
                # Check if model file exists and is recent
                model_path = "models/fpl_points_model.joblib"
                if os.path.exists(model_path):
                    mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
                    age_days = (datetime.now() - mod_time).days
                    
                    if age_days <= 7:
                        st.success(f"✅ Model is fresh ({age_days} days old)")
                    elif age_days <= 14:
                        st.warning(f"⚠️ Model is aging ({age_days} days old)")
                    else:
                        st.error(f"❌ Model is stale ({age_days} days old)")
                else:
                    st.error("❌ Model file not found")
            except Exception as e:
                st.error(f"❌ Error checking model: {str(e)}")
        
        with col2:
            st.markdown("### 📊 Data Pipeline")
            try:
                # Check data freshness
                data_path = "data/processed"
                if os.path.exists(data_path):
                    files = os.listdir(data_path)
                    if files:
                        st.success("✅ Data pipeline operational")
                    else:
                        st.warning("⚠️ No processed data found")
                else:
                    st.error("❌ Data directory not found")
            except Exception as e:
                st.error(f"❌ Error checking data: {str(e)}")
        
        with col3:
            st.markdown("### 🤖 Automation")
            try:
                # Check if crontab is set up
                import subprocess
                result = subprocess.run(['crontab', '-l'], 
                                      capture_output=True, text=True)
                
                if 'fpl' in result.stdout.lower():
                    st.success("✅ Automated retraining enabled")
                else:
                    st.warning("⚠️ No automation detected")
            except Exception as e:
                st.info("ℹ️ Cannot check automation status")

def main():
    """Main entry point for the dashboard"""
    dashboard = DashboardApp()
    dashboard.run()

if __name__ == "__main__":
    main()