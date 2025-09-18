"""
Feature Monitoring Dashboard
Comprehensive visualization for feature drift and data quality monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import os
from feature_monitoring import FeatureMonitor
from data_quality_monitor import DataQualityMonitor

class FeatureMonitoringDashboard:
    """Interactive dashboard for feature monitoring and data quality"""
    
    def __init__(self):
        self.feature_monitor = FeatureMonitor()
        self.quality_monitor = DataQualityMonitor()
        self.setup_page_config()
    
    def setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="FPL Feature Monitoring Dashboard",
            page_icon="⚽",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def main(self):
        """Main dashboard interface"""
        st.title("⚽ FPL Feature Monitoring Dashboard")
        st.markdown("### Real-time monitoring of feature drift and data quality")
        
        # Sidebar navigation
        with st.sidebar:
            st.header("🔧 Controls")
            
            # Dashboard sections
            section = st.selectbox(
                "📊 Dashboard Section",
                ["Overview", "Feature Drift", "Data Quality", "Alerts", "Historical Trends"]
            )
            
            # Data refresh
            if st.button("🔄 Refresh Data"):
                st.rerun()
            
            # Time range selection
            st.subheader("📅 Time Range")
            time_range = st.selectbox(
                "Select Period",
                ["Last 7 days", "Last 30 days", "Last 90 days", "All time"]
            )
            
            days_map = {
                "Last 7 days": 7,
                "Last 30 days": 30,
                "Last 90 days": 90,
                "All time": 365
            }
            days = days_map[time_range]
        
        # Main content based on selected section
        if section == "Overview":
            self.show_overview()
        elif section == "Feature Drift":
            self.show_feature_drift(days)
        elif section == "Data Quality":
            self.show_data_quality(days)
        elif section == "Alerts":
            self.show_alerts()
        elif section == "Historical Trends":
            self.show_historical_trends(days)
    
    def show_overview(self):
        """Show overview dashboard"""
        st.header("📈 System Overview")
        
        # Get latest metrics
        try:
            # Load current feature data
            feature_data_path = "data/performance_history/model_features.csv"
            if os.path.exists(feature_data_path):
                df = pd.read_csv(feature_data_path)
                
                # Run quick quality check
                quality_report = self.quality_monitor.run_comprehensive_quality_check(df, 'fpl_players')
                
                # Display key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "📊 Data Quality Score",
                        f"{quality_report['quality_score']:.3f}",
                        f"Grade: {quality_report['quality_grade']}"
                    )
                
                with col2:
                    st.metric(
                        "📋 Dataset Size",
                        f"{quality_report['summary']['total_rows']:,} rows",
                        f"{quality_report['summary']['total_columns']} features"
                    )
                
                with col3:
                    completeness = quality_report['summary']['completeness']
                    st.metric(
                        "✅ Data Completeness",
                        completeness,
                        "🟢 Good" if float(completeness.rstrip('%')) > 90 else "🟡 Moderate"
                    )
                
                with col4:
                    violations = len(quality_report['violations'])
                    st.metric(
                        "⚠️ Schema Violations",
                        violations,
                        "🟢 None" if violations == 0 else f"🔴 {violations} issues"
                    )
                
                # Feature drift overview
                st.subheader("🌊 Feature Drift Status")
                
                # Check for recent drift
                drift_results = self.get_recent_drift_results()
                
                if drift_results:
                    st.write("Recent drift detection results:")
                    
                    drift_df = pd.DataFrame(drift_results)
                    
                    # Create drift status chart
                    fig = px.bar(
                        drift_df, 
                        x='feature_name', 
                        y='drift_score',
                        color='has_drift',
                        title="Feature Drift Scores",
                        color_discrete_map={True: 'red', False: 'green'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("🔍 No recent drift detection results available. Run feature monitoring to generate drift analysis.")
                
                # Quality recommendations
                st.subheader("💡 Quality Recommendations")
                for rec in quality_report['recommendations']:
                    st.write(f"• {rec}")
            
            else:
                st.error("❌ Feature data not found. Please ensure model_features.csv exists.")
        
        except Exception as e:
            st.error(f"❌ Error loading overview: {e}")
    
    def show_feature_drift(self, days: int):
        """Show feature drift analysis"""
        st.header("🌊 Feature Drift Analysis")
        
        try:
            # Get drift results from database
            drift_data = self.get_drift_data(days)
            
            if not drift_data.empty:
                # Drift summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_features = drift_data['feature_name'].nunique()
                    st.metric("📊 Total Features Monitored", total_features)
                
                with col2:
                    drifted_features = drift_data[drift_data['has_drift'] == 1]['feature_name'].nunique()
                    st.metric("🚨 Features with Drift", drifted_features)
                
                with col3:
                    avg_drift_score = drift_data['drift_score'].mean()
                    st.metric("📈 Average Drift Score", f"{avg_drift_score:.3f}")
                
                # Drift timeline
                st.subheader("📅 Drift Detection Timeline")
                
                drift_data['timestamp'] = pd.to_datetime(drift_data['timestamp'])
                daily_drift = drift_data.groupby([
                    drift_data['timestamp'].dt.date, 
                    'has_drift'
                ]).size().reset_index(name='count')
                
                fig = px.line(
                    daily_drift, 
                    x='timestamp', 
                    y='count',
                    color='has_drift',
                    title="Daily Drift Detection Results",
                    color_discrete_map={0: 'green', 1: 'red'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature-wise drift analysis
                st.subheader("🔍 Feature-wise Drift Analysis")
                
                # Select feature for detailed analysis
                features = sorted(drift_data['feature_name'].unique())
                selected_feature = st.selectbox("Select Feature for Analysis", features)
                
                if selected_feature:
                    feature_data = drift_data[drift_data['feature_name'] == selected_feature]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Drift score over time
                        fig = px.line(
                            feature_data, 
                            x='timestamp', 
                            y='drift_score',
                            title=f"Drift Score Timeline - {selected_feature}",
                            markers=True
                        )
                        fig.add_hline(y=0.05, line_dash="dash", line_color="red", 
                                    annotation_text="Drift Threshold")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # PSI score over time
                        if 'psi_score' in feature_data.columns:
                            fig = px.line(
                                feature_data, 
                                x='timestamp', 
                                y='psi_score',
                                title=f"PSI Score Timeline - {selected_feature}",
                                markers=True
                            )
                            fig.add_hline(y=0.1, line_dash="dash", line_color="red", 
                                        annotation_text="PSI Threshold")
                            st.plotly_chart(fig, use_container_width=True)
                
                # Drift heatmap
                st.subheader("🔥 Feature Drift Heatmap")
                
                # Create pivot table for heatmap
                drift_pivot = drift_data.pivot_table(
                    values='drift_score',
                    index='feature_name',
                    columns=drift_data['timestamp'].dt.date,
                    fill_value=0
                )
                
                if not drift_pivot.empty:
                    fig = px.imshow(
                        drift_pivot.values,
                        x=drift_pivot.columns.astype(str),
                        y=drift_pivot.index,
                        title="Feature Drift Heatmap",
                        aspect="auto",
                        color_continuous_scale="Reds"
                    )
                    fig.update_layout(height=max(400, len(drift_pivot) * 25))
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("📊 No drift data available for the selected time period.")
                
                # Option to run drift detection
                if st.button("🔍 Run Feature Drift Detection"):
                    with st.spinner("Running drift detection..."):
                        try:
                            # Load current data and run drift detection
                            feature_data_path = "data/performance_history/model_features.csv"
                            if os.path.exists(feature_data_path):
                                df = pd.read_csv(feature_data_path)
                                results = self.feature_monitor.run_comprehensive_monitoring(df)
                                st.success("✅ Drift detection completed!")
                                st.rerun()
                            else:
                                st.error("❌ Feature data not found.")
                        except Exception as e:
                            st.error(f"❌ Error running drift detection: {e}")
        
        except Exception as e:
            st.error(f"❌ Error loading drift analysis: {e}")
    
    def show_data_quality(self, days: int):
        """Show data quality dashboard"""
        st.header("🔍 Data Quality Analysis")
        
        try:
            # Get quality metrics from database
            quality_data = self.get_quality_data(days)
            
            if not quality_data.empty:
                # Quality metrics over time
                st.subheader("📈 Quality Score Timeline")
                
                quality_data['timestamp'] = pd.to_datetime(quality_data['timestamp'])
                
                fig = go.Figure()
                
                # Add quality score line
                fig.add_trace(go.Scatter(
                    x=quality_data['timestamp'],
                    y=quality_data['quality_score'],
                    mode='lines+markers',
                    name='Quality Score',
                    line=dict(color='blue', width=3)
                ))
                
                # Add quality threshold lines
                fig.add_hline(y=0.9, line_dash="dash", line_color="green", 
                            annotation_text="Excellent (A)")
                fig.add_hline(y=0.8, line_dash="dash", line_color="orange", 
                            annotation_text="Good (B)")
                fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                            annotation_text="Poor (C)")
                
                fig.update_layout(
                    title="Data Quality Score Over Time",
                    xaxis_title="Date",
                    yaxis_title="Quality Score",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Quality metrics breakdown
                st.subheader("📊 Quality Metrics Breakdown")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Missing data percentage over time
                    fig = px.line(
                        quality_data, 
                        x='timestamp', 
                        y='missing_values_percentage',
                        title="Missing Data Percentage",
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Schema violations over time
                    fig = px.line(
                        quality_data, 
                        x='timestamp', 
                        y='schema_violations',
                        title="Schema Violations Count",
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Quality grade distribution
                st.subheader("🏆 Quality Grade Distribution")
                
                grade_counts = quality_data['quality_grade'].value_counts()
                
                fig = px.pie(
                    values=grade_counts.values,
                    names=grade_counts.index,
                    title="Distribution of Quality Grades",
                    color_discrete_map={
                        'A': 'green',
                        'B': 'lightgreen',
                        'C': 'orange',
                        'D': 'red',
                        'F': 'darkred'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Latest quality report
                st.subheader("📋 Latest Quality Report")
                
                latest_metrics = quality_data.iloc[-1]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Overall Score", f"{latest_metrics['quality_score']:.3f}")
                    st.metric("Grade", latest_metrics['quality_grade'])
                
                with col2:
                    st.metric("Missing Data", f"{latest_metrics['missing_values_percentage']:.1f}%")
                    st.metric("Complete Rows", f"{latest_metrics['complete_rows_percentage']:.1f}%")
                
                with col3:
                    st.metric("Schema Violations", int(latest_metrics['schema_violations']))
                    st.metric("Duplicate Rows", f"{latest_metrics['duplicate_percentage']:.1f}%")
                
                with col4:
                    st.metric("Outliers", f"{latest_metrics['outliers_percentage']:.1f}%")
                    st.metric("Total Rows", f"{int(latest_metrics['total_rows']):,}")
            
            else:
                st.info("📊 No quality data available for the selected time period.")
                
                # Option to run quality check
                if st.button("🔍 Run Data Quality Check"):
                    with st.spinner("Running quality assessment..."):
                        try:
                            # Load current data and run quality check
                            feature_data_path = "data/performance_history/model_features.csv"
                            if os.path.exists(feature_data_path):
                                df = pd.read_csv(feature_data_path)
                                quality_report = self.quality_monitor.run_comprehensive_quality_check(df, 'fpl_players')
                                st.success("✅ Quality assessment completed!")
                                st.rerun()
                            else:
                                st.error("❌ Feature data not found.")
                        except Exception as e:
                            st.error(f"❌ Error running quality check: {e}")
        
        except Exception as e:
            st.error(f"❌ Error loading quality analysis: {e}")
    
    def show_alerts(self):
        """Show alerts and notifications"""
        st.header("🚨 Monitoring Alerts")
        
        try:
            # Get recent alerts
            quality_alerts = self.get_quality_alerts()
            drift_alerts = self.get_drift_alerts()
            
            # Alert summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🔍 Quality Alerts", len(quality_alerts))
            
            with col2:
                st.metric("🌊 Drift Alerts", len(drift_alerts))
            
            with col3:
                total_alerts = len(quality_alerts) + len(drift_alerts)
                st.metric("📊 Total Active Alerts", total_alerts)
            
            # Quality alerts
            if quality_alerts:
                st.subheader("🔍 Data Quality Alerts")
                
                for alert in quality_alerts[:10]:  # Show latest 10
                    with st.expander(f"⚠️ {alert['alert_type']} - {alert['severity']}"):
                        st.write(f"**Dataset:** {alert['dataset_name']}")
                        st.write(f"**Timestamp:** {alert['timestamp']}")
                        st.write(f"**Message:** {alert['message']}")
                        if alert['details']:
                            st.write(f"**Details:** {alert['details']}")
            
            # Drift alerts
            if drift_alerts:
                st.subheader("🌊 Feature Drift Alerts")
                
                for alert in drift_alerts[:10]:  # Show latest 10
                    with st.expander(f"🚨 Drift Detected - {alert['feature_name']}"):
                        st.write(f"**Feature:** {alert['feature_name']}")
                        st.write(f"**Timestamp:** {alert['timestamp']}")
                        st.write(f"**Drift Score:** {alert['drift_score']:.4f}")
                        st.write(f"**PSI Score:** {alert.get('psi_score', 'N/A')}")
            
            if not quality_alerts and not drift_alerts:
                st.success("✅ No active alerts - all systems operating normally!")
        
        except Exception as e:
            st.error(f"❌ Error loading alerts: {e}")
    
    def show_historical_trends(self, days: int):
        """Show historical trends and analysis"""
        st.header("📈 Historical Trends Analysis")
        
        try:
            # Get combined trend data
            quality_data = self.get_quality_data(days)
            drift_data = self.get_drift_data(days)
            
            if not quality_data.empty or not drift_data.empty:
                # Combined timeline
                st.subheader("📅 Combined Monitoring Timeline")
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=['Data Quality Score', 'Feature Drift Incidents'],
                    shared_xaxes=True,
                    vertical_spacing=0.1
                )
                
                # Quality score timeline
                if not quality_data.empty:
                    quality_data['timestamp'] = pd.to_datetime(quality_data['timestamp'])
                    fig.add_trace(
                        go.Scatter(
                            x=quality_data['timestamp'],
                            y=quality_data['quality_score'],
                            mode='lines+markers',
                            name='Quality Score',
                            line=dict(color='blue')
                        ),
                        row=1, col=1
                    )
                
                # Drift incidents timeline
                if not drift_data.empty:
                    drift_data['timestamp'] = pd.to_datetime(drift_data['timestamp'])
                    daily_drift = drift_data[drift_data['has_drift'] == 1].groupby(
                        drift_data['timestamp'].dt.date
                    ).size().reset_index(name='drift_count')
                    daily_drift['timestamp'] = pd.to_datetime(daily_drift['timestamp'])
                    
                    fig.add_trace(
                        go.Scatter(
                            x=daily_drift['timestamp'],
                            y=daily_drift['drift_count'],
                            mode='lines+markers',
                            name='Drift Incidents',
                            line=dict(color='red')
                        ),
                        row=2, col=1
                    )
                
                fig.update_layout(height=600, title="Monitoring Trends Overview")
                st.plotly_chart(fig, use_container_width=True)
                
                # Trend statistics
                st.subheader("📊 Trend Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if not quality_data.empty:
                        st.write("**Data Quality Trends:**")
                        avg_quality = quality_data['quality_score'].mean()
                        quality_trend = "Improving" if quality_data['quality_score'].iloc[-1] > quality_data['quality_score'].iloc[0] else "Declining"
                        st.write(f"• Average Quality Score: {avg_quality:.3f}")
                        st.write(f"• Overall Trend: {quality_trend}")
                        st.write(f"• Best Grade: {quality_data['quality_grade'].mode().iloc[0]}")
                
                with col2:
                    if not drift_data.empty:
                        st.write("**Feature Drift Trends:**")
                        total_drift_incidents = len(drift_data[drift_data['has_drift'] == 1])
                        most_drifted_feature = drift_data[drift_data['has_drift'] == 1]['feature_name'].mode()
                        avg_drift_score = drift_data['drift_score'].mean()
                        
                        st.write(f"• Total Drift Incidents: {total_drift_incidents}")
                        st.write(f"• Average Drift Score: {avg_drift_score:.4f}")
                        if len(most_drifted_feature) > 0:
                            st.write(f"• Most Drifted Feature: {most_drifted_feature.iloc[0]}")
            
            else:
                st.info("📊 No historical data available for the selected time period.")
        
        except Exception as e:
            st.error(f"❌ Error loading trends analysis: {e}")
    
    # Database query methods
    def get_drift_data(self, days: int) -> pd.DataFrame:
        """Get feature drift data from database"""
        try:
            conn = sqlite3.connect(self.feature_monitor.db_path)
            query = '''
                SELECT * FROM feature_drift 
                WHERE timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp DESC
            '''.format(days)
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except:
            return pd.DataFrame()
    
    def get_quality_data(self, days: int) -> pd.DataFrame:
        """Get data quality metrics from database"""
        try:
            conn = sqlite3.connect(self.quality_monitor.db_path)
            query = '''
                SELECT * FROM quality_metrics 
                WHERE timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp DESC
            '''.format(days)
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except:
            return pd.DataFrame()
    
    def get_recent_drift_results(self) -> List[Dict]:
        """Get recent drift detection results"""
        try:
            conn = sqlite3.connect(self.feature_monitor.db_path)
            query = '''
                SELECT feature_name, drift_score, has_drift, psi_score 
                FROM feature_drift 
                WHERE timestamp >= datetime('now', '-1 day')
                ORDER BY drift_score DESC
                LIMIT 20
            '''
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'feature_name': row[0],
                    'drift_score': row[1],
                    'has_drift': bool(row[2]),
                    'psi_score': row[3]
                }
                for row in results
            ]
        except:
            return []
    
    def get_quality_alerts(self) -> List[Dict]:
        """Get recent quality alerts"""
        try:
            conn = sqlite3.connect(self.quality_monitor.db_path)
            query = '''
                SELECT * FROM quality_rules 
                WHERE enabled = 1
                ORDER BY updated_at DESC
                LIMIT 10
            '''
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'alert_type': 'Quality Rule',
                    'dataset_name': row[2] or 'Global',
                    'timestamp': row[6],
                    'severity': 'Medium',
                    'message': row[1],
                    'details': row[4]
                }
                for row in results
            ]
        except:
            return []
    
    def get_drift_alerts(self) -> List[Dict]:
        """Get recent drift alerts"""
        try:
            conn = sqlite3.connect(self.feature_monitor.db_path)
            query = '''
                SELECT feature_name, timestamp, drift_score, psi_score 
                FROM feature_drift 
                WHERE has_drift = 1 
                AND timestamp >= datetime('now', '-7 days')
                ORDER BY timestamp DESC
                LIMIT 10
            '''
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'feature_name': row[0],
                    'timestamp': row[1],
                    'drift_score': row[2],
                    'psi_score': row[3]
                }
                for row in results
            ]
        except:
            return []

def main():
    """Run the feature monitoring dashboard"""
    dashboard = FeatureMonitoringDashboard()
    dashboard.main()

if __name__ == "__main__":
    main()