"""
Enhanced Data Quality Monitoring System
Comprehensive data validation, schema checking, and quality metrics
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SchemaDefinition:
    """Definition of expected data schema"""
    column_name: str
    data_type: str  # 'int', 'float', 'string', 'bool'
    nullable: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[str]] = None
    pattern: Optional[str] = None  # regex pattern for strings
    description: str = ""

@dataclass
class QualityMetrics:
    """Data quality metrics for a dataset"""
    timestamp: datetime
    dataset_name: str
    total_rows: int
    total_columns: int
    
    # Completeness metrics
    missing_values_count: int
    missing_values_percentage: float
    complete_rows_count: int
    complete_rows_percentage: float
    
    # Validity metrics
    schema_violations: int
    data_type_errors: int
    range_violations: int
    pattern_violations: int
    
    # Consistency metrics
    duplicate_rows: int
    duplicate_percentage: float
    
    # Accuracy metrics (statistical)
    outliers_count: int
    outliers_percentage: float
    
    # Overall quality score (0-1)
    quality_score: float
    
    # Quality grade (A, B, C, D, F)
    quality_grade: str

class DataQualityMonitor:
    """Comprehensive data quality monitoring system"""
    
    def __init__(self, db_path: str = "data_quality_monitoring.db"):
        self.db_path = db_path
        self.schema_definitions = {}
        self.init_database()
        self.setup_logging()
        self.load_schema_definitions()
    
    def setup_logging(self):
        """Setup logging for data quality monitoring"""
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/data_quality.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DataQualityMonitor')
    
    def init_database(self):
        """Initialize data quality monitoring database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Quality metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                total_rows INTEGER,
                total_columns INTEGER,
                missing_values_count INTEGER,
                missing_values_percentage REAL,
                complete_rows_count INTEGER,
                complete_rows_percentage REAL,
                schema_violations INTEGER,
                data_type_errors INTEGER,
                range_violations INTEGER,
                pattern_violations INTEGER,
                duplicate_rows INTEGER,
                duplicate_percentage REAL,
                outliers_count INTEGER,
                outliers_percentage REAL,
                quality_score REAL,
                quality_grade TEXT,
                additional_metrics TEXT
            )
        ''')
        
        # Schema validation results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS schema_violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                column_name TEXT NOT NULL,
                violation_type TEXT NOT NULL,
                violation_count INTEGER,
                sample_values TEXT,
                severity TEXT
            )
        ''')
        
        # Data quality rules
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_name TEXT UNIQUE NOT NULL,
                dataset_name TEXT,
                column_name TEXT,
                rule_type TEXT NOT NULL,
                rule_definition TEXT NOT NULL,
                enabled BOOLEAN DEFAULT TRUE,
                created_at TEXT NOT NULL,
                updated_at TEXT
            )
        ''')
        
        # Quality trends
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_trends (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                trend_direction TEXT,
                change_percentage REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_schema_definitions(self):
        """Load or create schema definitions for FPL data"""
        
        # FPL Player Data Schema
        self.schema_definitions['fpl_players'] = [
            SchemaDefinition('player_id', 'int', False, 1, 1000, description="FPL Player ID"),
            SchemaDefinition('name', 'string', False, description="Player name"),
            SchemaDefinition('team', 'int', False, 1, 20, description="Team ID"),
            SchemaDefinition('position', 'int', False, 1, 4, description="Position type (1-4)"),
            SchemaDefinition('gameweeks_played', 'int', True, 0, 38, description="Games played"),
            SchemaDefinition('avg_PTS', 'float', True, -5, 30, description="Average points"),
            SchemaDefinition('avg_MP', 'float', True, 0, 90, description="Average minutes"),
            SchemaDefinition('avg_GS', 'float', True, 0, 10, description="Average goals scored"),
            SchemaDefinition('avg_A', 'float', True, 0, 10, description="Average assists"),
            SchemaDefinition('avg_xG', 'float', True, 0, 5, description="Average expected goals"),
            SchemaDefinition('avg_xA', 'float', True, 0, 5, description="Average expected assists"),
            SchemaDefinition('avg_CS', 'float', True, 0, 1, description="Average clean sheets"),
            SchemaDefinition('avg_T', 'float', True, 0, 20, description="Average transfers"),
            SchemaDefinition('avg_CBI', 'float', True, 0, 50, description="Average combined index"),
            SchemaDefinition('avg_R', 'float', True, 0, 20, description="Average recoveries"),
            SchemaDefinition('avg_BP', 'float', True, 0, 10, description="Average bonus points"),
            SchemaDefinition('avg_BPS', 'float', True, 0, 100, description="Average bonus point system"),
            SchemaDefinition('avg_I', 'float', True, 0, 20, description="Average influence"),
            SchemaDefinition('avg_C', 'float', True, 0, 20, description="Average creativity"),
            SchemaDefinition('avg_T_threat', 'float', True, 0, 100, description="Average threat"),
            SchemaDefinition('avg_II', 'float', True, 0, 50, description="Average ict index")
        ]
        
        # Trend features schema (optional features that can be missing)
        trend_features = [
            'trend_PTS', 'trend_MP', 'trend_GS', 'trend_A', 'trend_xG', 'trend_xA',
            'trend_CS', 'trend_T', 'trend_CBI', 'trend_R', 'trend_BP', 'trend_BPS',
            'trend_I', 'trend_C', 'trend_T_threat', 'trend_II'
        ]
        
        for feature in trend_features:
            self.schema_definitions['fpl_players'].append(
                SchemaDefinition(feature, 'float', True, -100, 100, description=f"Trend for {feature.replace('trend_', '')}")
            )
    
    def validate_schema(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Validate dataset against expected schema"""
        
        if dataset_name not in self.schema_definitions:
            self.logger.warning(f"No schema definition found for dataset: {dataset_name}")
            return {'schema_valid': True, 'violations': []}
        
        schema = self.schema_definitions[dataset_name]
        violations = []
        
        for schema_def in schema:
            col_name = schema_def.column_name
            
            # Check if column exists
            if col_name not in df.columns:
                if not schema_def.nullable:
                    violations.append({
                        'column': col_name,
                        'type': 'MISSING_REQUIRED_COLUMN',
                        'count': 1,
                        'severity': 'HIGH',
                        'message': f"Required column '{col_name}' is missing"
                    })
                continue
            
            column_data = df[col_name]
            
            # Check data type compatibility
            type_violations = self._check_data_type(column_data, schema_def)
            violations.extend(type_violations)
            
            # Check nullable constraint
            if not schema_def.nullable and column_data.isnull().any():
                null_count = column_data.isnull().sum()
                violations.append({
                    'column': col_name,
                    'type': 'NULL_IN_NON_NULLABLE',
                    'count': null_count,
                    'severity': 'HIGH',
                    'message': f"Found {null_count} null values in non-nullable column '{col_name}'"
                })
            
            # Check value ranges (for numeric data)
            if schema_def.min_value is not None or schema_def.max_value is not None:
                range_violations = self._check_value_ranges(column_data, schema_def)
                violations.extend(range_violations)
            
            # Check allowed values
            if schema_def.allowed_values:
                value_violations = self._check_allowed_values(column_data, schema_def)
                violations.extend(value_violations)
            
            # Check patterns (for string data)
            if schema_def.pattern:
                pattern_violations = self._check_patterns(column_data, schema_def)
                violations.extend(pattern_violations)
        
        return {
            'schema_valid': len(violations) == 0,
            'violations': violations,
            'total_violations': len(violations),
            'high_severity_violations': len([v for v in violations if v['severity'] == 'HIGH'])
        }
    
    def _check_data_type(self, column_data: pd.Series, schema_def: SchemaDefinition) -> List[Dict]:
        """Check data type compatibility"""
        violations = []
        
        expected_type = schema_def.data_type
        col_name = schema_def.column_name
        
        # Skip null values for type checking
        non_null_data = column_data.dropna()
        
        if len(non_null_data) == 0:
            return violations
        
        if expected_type == 'int':
            # Check if values can be converted to int
            try:
                pd.to_numeric(non_null_data, errors='raise')
                # Check if all values are whole numbers
                float_values = pd.to_numeric(non_null_data)
                non_int_count = sum(float_values != float_values.astype(int))
                if non_int_count > 0:
                    violations.append({
                        'column': col_name,
                        'type': 'DATA_TYPE_MISMATCH',
                        'count': non_int_count,
                        'severity': 'MEDIUM',
                        'message': f"Found {non_int_count} non-integer values in integer column '{col_name}'"
                    })
            except:
                violations.append({
                    'column': col_name,
                    'type': 'DATA_TYPE_MISMATCH',
                    'count': len(non_null_data),
                    'severity': 'HIGH',
                    'message': f"Cannot convert column '{col_name}' to integer"
                })
        
        elif expected_type == 'float':
            try:
                pd.to_numeric(non_null_data, errors='raise')
            except:
                violations.append({
                    'column': col_name,
                    'type': 'DATA_TYPE_MISMATCH',
                    'count': len(non_null_data),
                    'severity': 'HIGH',
                    'message': f"Cannot convert column '{col_name}' to numeric"
                })
        
        elif expected_type == 'string':
            # Check if all values can be converted to string
            non_string_count = sum(~non_null_data.astype(str).notna())
            if non_string_count > 0:
                violations.append({
                    'column': col_name,
                    'type': 'DATA_TYPE_MISMATCH',
                    'count': non_string_count,
                    'severity': 'LOW',
                    'message': f"Found {non_string_count} values that cannot be converted to string in '{col_name}'"
                })
        
        return violations
    
    def _check_value_ranges(self, column_data: pd.Series, schema_def: SchemaDefinition) -> List[Dict]:
        """Check if values are within expected ranges"""
        violations = []
        
        # Only check numeric data
        if schema_def.data_type not in ['int', 'float']:
            return violations
        
        try:
            numeric_data = pd.to_numeric(column_data, errors='coerce').dropna()
            
            if len(numeric_data) == 0:
                return violations
            
            range_violations_count = 0
            
            if schema_def.min_value is not None:
                below_min = (numeric_data < schema_def.min_value).sum()
                range_violations_count += below_min
            
            if schema_def.max_value is not None:
                above_max = (numeric_data > schema_def.max_value).sum()
                range_violations_count += above_max
            
            if range_violations_count > 0:
                violations.append({
                    'column': schema_def.column_name,
                    'type': 'VALUE_RANGE_VIOLATION',
                    'count': range_violations_count,
                    'severity': 'MEDIUM' if range_violations_count < len(numeric_data) * 0.1 else 'HIGH',
                    'message': f"Found {range_violations_count} values outside expected range [{schema_def.min_value}, {schema_def.max_value}] in '{schema_def.column_name}'"
                })
        
        except Exception as e:
            self.logger.warning(f"Error checking value ranges for {schema_def.column_name}: {e}")
        
        return violations
    
    def _check_allowed_values(self, column_data: pd.Series, schema_def: SchemaDefinition) -> List[Dict]:
        """Check if values are in allowed set"""
        violations = []
        
        non_null_data = column_data.dropna()
        if len(non_null_data) == 0:
            return violations
        
        invalid_values = ~non_null_data.isin(schema_def.allowed_values)
        invalid_count = invalid_values.sum()
        
        if invalid_count > 0:
            violations.append({
                'column': schema_def.column_name,
                'type': 'INVALID_VALUE',
                'count': invalid_count,
                'severity': 'MEDIUM',
                'message': f"Found {invalid_count} invalid values in '{schema_def.column_name}' (not in allowed set)"
            })
        
        return violations
    
    def _check_patterns(self, column_data: pd.Series, schema_def: SchemaDefinition) -> List[Dict]:
        """Check if string values match expected patterns"""
        violations = []
        
        # This would need regex pattern matching implementation
        # For now, return empty violations
        return violations
    
    def calculate_quality_metrics(self, df: pd.DataFrame, dataset_name: str) -> QualityMetrics:
        """Calculate comprehensive quality metrics for dataset"""
        
        timestamp = datetime.now()
        total_rows = len(df)
        total_columns = len(df.columns)
        
        # Completeness metrics
        missing_values_count = df.isnull().sum().sum()
        total_cells = total_rows * total_columns
        missing_values_percentage = (missing_values_count / total_cells * 100) if total_cells > 0 else 0
        
        complete_rows_count = len(df.dropna())
        complete_rows_percentage = (complete_rows_count / total_rows * 100) if total_rows > 0 else 0
        
        # Schema validation
        schema_validation = self.validate_schema(df, dataset_name)
        schema_violations = schema_validation['total_violations']
        
        # Specific violation counts
        violations = schema_validation['violations']
        data_type_errors = len([v for v in violations if v['type'] == 'DATA_TYPE_MISMATCH'])
        range_violations = len([v for v in violations if v['type'] == 'VALUE_RANGE_VIOLATION'])
        pattern_violations = len([v for v in violations if v['type'] == 'PATTERN_VIOLATION'])
        
        # Consistency metrics
        duplicate_rows = df.duplicated().sum()
        duplicate_percentage = (duplicate_rows / total_rows * 100) if total_rows > 0 else 0
        
        # Outlier detection (for numeric columns)
        outliers_count = 0
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in ['player_id', 'team', 'position']:  # Skip ID columns
                continue
            
            clean_data = df[col].dropna()
            if len(clean_data) > 0:
                Q1 = clean_data.quantile(0.25)
                Q3 = clean_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = clean_data[(clean_data < Q1 - 1.5 * IQR) | (clean_data > Q3 + 1.5 * IQR)]
                outliers_count += len(outliers)
        
        outliers_percentage = (outliers_count / total_cells * 100) if total_cells > 0 else 0
        
        # Calculate overall quality score (0-1)
        quality_score = self._calculate_quality_score(
            missing_values_percentage, complete_rows_percentage, schema_violations,
            duplicate_percentage, outliers_percentage, total_rows
        )
        
        # Assign quality grade
        quality_grade = self._assign_quality_grade(quality_score)
        
        return QualityMetrics(
            timestamp=timestamp,
            dataset_name=dataset_name,
            total_rows=total_rows,
            total_columns=total_columns,
            missing_values_count=missing_values_count,
            missing_values_percentage=missing_values_percentage,
            complete_rows_count=complete_rows_count,
            complete_rows_percentage=complete_rows_percentage,
            schema_violations=schema_violations,
            data_type_errors=data_type_errors,
            range_violations=range_violations,
            pattern_violations=pattern_violations,
            duplicate_rows=duplicate_rows,
            duplicate_percentage=duplicate_percentage,
            outliers_count=outliers_count,
            outliers_percentage=outliers_percentage,
            quality_score=quality_score,
            quality_grade=quality_grade
        )
    
    def _calculate_quality_score(self, missing_pct: float, complete_rows_pct: float,
                                schema_violations: int, duplicate_pct: float,
                                outliers_pct: float, total_rows: int) -> float:
        """Calculate overall quality score (0-1)"""
        
        # Completeness score (30% weight)
        completeness_score = max(0, 1 - missing_pct / 100)
        
        # Validity score (25% weight)
        validity_score = max(0, 1 - (schema_violations / max(total_rows, 1)))
        
        # Consistency score (25% weight)
        consistency_score = max(0, 1 - duplicate_pct / 100)
        
        # Accuracy score (20% weight) - based on outliers
        accuracy_score = max(0, 1 - outliers_pct / 100)
        
        # Weighted average
        overall_score = (
            completeness_score * 0.30 +
            validity_score * 0.25 +
            consistency_score * 0.25 +
            accuracy_score * 0.20
        )
        
        return min(1.0, max(0.0, overall_score))
    
    def _assign_quality_grade(self, quality_score: float) -> str:
        """Assign letter grade based on quality score"""
        if quality_score >= 0.9:
            return 'A'
        elif quality_score >= 0.8:
            return 'B'
        elif quality_score >= 0.7:
            return 'C'
        elif quality_score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def save_quality_metrics(self, metrics: QualityMetrics):
        """Save quality metrics to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO quality_metrics (
                timestamp, dataset_name, total_rows, total_columns,
                missing_values_count, missing_values_percentage,
                complete_rows_count, complete_rows_percentage,
                schema_violations, data_type_errors, range_violations, pattern_violations,
                duplicate_rows, duplicate_percentage,
                outliers_count, outliers_percentage,
                quality_score, quality_grade
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp.isoformat(), metrics.dataset_name,
            metrics.total_rows, metrics.total_columns,
            metrics.missing_values_count, metrics.missing_values_percentage,
            metrics.complete_rows_count, metrics.complete_rows_percentage,
            metrics.schema_violations, metrics.data_type_errors,
            metrics.range_violations, metrics.pattern_violations,
            metrics.duplicate_rows, metrics.duplicate_percentage,
            metrics.outliers_count, metrics.outliers_percentage,
            metrics.quality_score, metrics.quality_grade
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Saved quality metrics for {metrics.dataset_name}: Grade {metrics.quality_grade} (Score: {metrics.quality_score:.3f})")
    
    def save_schema_violations(self, violations: List[Dict], dataset_name: str):
        """Save schema violations to database"""
        if not violations:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        for violation in violations:
            cursor.execute('''
                INSERT INTO schema_violations (
                    timestamp, dataset_name, column_name, violation_type,
                    violation_count, severity
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                timestamp, dataset_name, violation['column'],
                violation['type'], violation['count'], violation['severity']
            ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Saved {len(violations)} schema violations for {dataset_name}")
    
    def run_comprehensive_quality_check(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Run comprehensive data quality assessment"""
        self.logger.info(f"🔍 Running comprehensive quality check for {dataset_name}...")
        
        try:
            # Calculate quality metrics
            metrics = self.calculate_quality_metrics(df, dataset_name)
            
            # Save metrics to database
            self.save_quality_metrics(metrics)
            
            # Get schema validation details
            schema_validation = self.validate_schema(df, dataset_name)
            
            # Save violations
            self.save_schema_violations(schema_validation['violations'], dataset_name)
            
            # Generate quality report
            quality_report = {
                'dataset_name': dataset_name,
                'timestamp': metrics.timestamp.isoformat(),
                'quality_score': metrics.quality_score,
                'quality_grade': metrics.quality_grade,
                'summary': {
                    'total_rows': metrics.total_rows,
                    'total_columns': metrics.total_columns,
                    'completeness': f"{100 - metrics.missing_values_percentage:.1f}%",
                    'schema_compliance': schema_validation['schema_valid'],
                    'duplicates': f"{metrics.duplicate_percentage:.1f}%",
                    'outliers': f"{metrics.outliers_percentage:.1f}%"
                },
                'violations': schema_validation['violations'],
                'recommendations': self._generate_quality_recommendations(metrics, schema_validation)
            }
            
            self.logger.info(f"✅ Quality check completed: Grade {metrics.quality_grade} (Score: {metrics.quality_score:.3f})")
            return quality_report
        
        except Exception as e:
            self.logger.error(f"❌ Error in quality check: {e}")
            return {'error': str(e), 'dataset_name': dataset_name}
    
    def _generate_quality_recommendations(self, metrics: QualityMetrics, 
                                        schema_validation: Dict) -> List[str]:
        """Generate actionable quality improvement recommendations"""
        recommendations = []
        
        # Missing data recommendations
        if metrics.missing_values_percentage > 10:
            recommendations.append(f"🔍 High missing data ({metrics.missing_values_percentage:.1f}%) - investigate data collection issues")
        elif metrics.missing_values_percentage > 5:
            recommendations.append(f"⚠️ Moderate missing data ({metrics.missing_values_percentage:.1f}%) - consider imputation strategies")
        
        # Schema compliance recommendations
        if not schema_validation['schema_valid']:
            high_severity = schema_validation.get('high_severity_violations', 0)
            if high_severity > 0:
                recommendations.append(f"🚨 {high_severity} critical schema violations require immediate attention")
            else:
                recommendations.append("📋 Schema violations detected - review data validation rules")
        
        # Duplicate data recommendations
        if metrics.duplicate_percentage > 5:
            recommendations.append(f"🔄 High duplicate rate ({metrics.duplicate_percentage:.1f}%) - implement deduplication process")
        
        # Outlier recommendations
        if metrics.outliers_percentage > 5:
            recommendations.append(f"📊 High outlier rate ({metrics.outliers_percentage:.1f}%) - review outlier detection and handling")
        
        # Overall quality recommendations
        if metrics.quality_score < 0.7:
            recommendations.append("📉 Low overall quality score - comprehensive data cleanup needed")
        elif metrics.quality_score < 0.85:
            recommendations.append("📈 Moderate quality score - targeted improvements recommended")
        
        if not recommendations:
            recommendations.append("✅ Good data quality - maintain current data management practices")
        
        return recommendations
    
    def get_quality_trends(self, dataset_name: str, days: int = 30) -> Dict[str, Any]:
        """Get quality trends over time"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT timestamp, quality_score, quality_grade,
                   missing_values_percentage, duplicate_percentage,
                   outliers_percentage, schema_violations
            FROM quality_metrics 
            WHERE dataset_name = ? 
            AND timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp
        '''.format(days)
        
        df = pd.read_sql_query(query, conn, params=(dataset_name,))
        conn.close()
        
        if df.empty:
            return {'no_data': True}
        
        # Calculate trends
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        trends = {
            'dataset_name': dataset_name,
            'period_days': days,
            'data_points': len(df),
            'current_score': df['quality_score'].iloc[-1] if len(df) > 0 else None,
            'average_score': df['quality_score'].mean(),
            'score_trend': 'improving' if len(df) > 1 and df['quality_score'].iloc[-1] > df['quality_score'].iloc[0] else 'declining' if len(df) > 1 else 'stable',
            'grade_distribution': df['quality_grade'].value_counts().to_dict(),
            'recent_violations': df['schema_violations'].sum()
        }
        
        return trends

def main():
    """Main entry point for data quality monitoring"""
    monitor = DataQualityMonitor()
    
    # Test with current feature data
    try:
        feature_data_path = "data/performance_history/model_features.csv"
        if os.path.exists(feature_data_path):
            df = pd.read_csv(feature_data_path)
            
            # Run quality check
            quality_report = monitor.run_comprehensive_quality_check(df, 'fpl_players')
            
            print("📊 Data Quality Assessment Results:")
            print(f"📈 Quality Score: {quality_report['quality_score']:.3f}")
            print(f"🏆 Quality Grade: {quality_report['quality_grade']}")
            print(f"📋 Dataset: {quality_report['summary']['total_rows']} rows, {quality_report['summary']['total_columns']} columns")
            print(f"✅ Completeness: {quality_report['summary']['completeness']}")
            print(f"🔄 Duplicates: {quality_report['summary']['duplicates']}")
            print(f"📊 Outliers: {quality_report['summary']['outliers']}")
            
            if quality_report['violations']:
                print(f"\n⚠️ Schema Violations: {len(quality_report['violations'])}")
                for violation in quality_report['violations'][:3]:  # Show first 3
                    print(f"   • {violation['message']}")
            
            print(f"\n💡 Recommendations:")
            for rec in quality_report['recommendations']:
                print(f"   • {rec}")
        
        else:
            print("❌ Feature data not found")
    
    except Exception as e:
        print(f"❌ Error in data quality monitoring: {e}")

if __name__ == "__main__":
    main()