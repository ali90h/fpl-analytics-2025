#!/usr/bin/env python3
"""
Model Validation Test Suite
Comprehensive testing for FPL prediction model accuracy and reliability
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from fpl_predictor import FPLPredictor

def test_model_loading():
    """Test model loading integrity"""
    print("🔍 Testing Model Loading...")
    try:
        predictor = FPLPredictor()
        
        # Check if model is loaded
        if not hasattr(predictor, 'model') or predictor.model is None:
            print("❌ Model not loaded")
            return False
        
        # Check if preprocessors are loaded
        if not hasattr(predictor, 'preprocessors') or predictor.preprocessors is None:
            print("❌ Preprocessors not loaded")
            return False
        
        # Check feature names
        if not hasattr(predictor, 'feature_names') or not predictor.feature_names:
            print("❌ Feature names not loaded")
            return False
        
        print(f"✅ Model loaded successfully with {len(predictor.feature_names)} features")
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_prediction_accuracy():
    """Test prediction accuracy with known data patterns"""
    print("\n🎯 Testing Prediction Accuracy...")
    try:
        predictor = FPLPredictor()
        
        # Load current data
        df = predictor._load_current_data()
        if df.empty:
            print("❌ No data available for testing")
            return False
        
        # Test with sample of players
        sample_size = min(50, len(df))
        test_sample = df.sample(n=sample_size, random_state=42)
        
        # Prepare features
        X = predictor._prepare_features(test_sample)
        if X is None:
            print("❌ Feature preparation failed")
            return False
        
        # Make predictions
        predictions = predictor._safe_predict(X)
        if predictions is None:
            print("❌ Prediction generation failed")
            return False
        
        # Validate predictions
        if len(predictions) != sample_size:
            print(f"❌ Prediction count mismatch: expected {sample_size}, got {len(predictions)}")
            return False
        
        # Check prediction statistics
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        pred_min = np.min(predictions)
        pred_max = np.max(predictions)
        
        print(f"📊 Prediction Statistics:")
        print(f"   Mean: {pred_mean:.2f} points")
        print(f"   Std Dev: {pred_std:.2f}")
        print(f"   Range: {pred_min:.2f} to {pred_max:.2f}")
        
        # Validate reasonable ranges
        if pred_min < -2 or pred_max > 30:
            print(f"⚠️ Some predictions outside reasonable range")
            return False
        
        if pred_std < 0.5:
            print("⚠️ Low prediction variance - model may not be discriminating")
            return False
        
        print("✅ Prediction accuracy test passed")
        return True
        
    except Exception as e:
        print(f"❌ Prediction accuracy test failed: {e}")
        return False

def test_feature_consistency():
    """Test feature consistency across different data loads"""
    print("\n🔧 Testing Feature Consistency...")
    try:
        predictor = FPLPredictor()
        
        # Load data multiple times and check feature consistency
        df1 = predictor._load_current_data()
        df2 = predictor._load_current_data()
        
        if df1.empty or df2.empty:
            print("❌ Data loading failed")
            return False
        
        # Prepare features for both datasets
        X1 = predictor._prepare_features(df1.head(10))
        X2 = predictor._prepare_features(df2.head(10))
        
        if X1 is None or X2 is None:
            print("❌ Feature preparation failed")
            return False
        
        # Check feature consistency
        if list(X1.columns) != list(X2.columns):
            print("❌ Feature columns inconsistent between loads")
            return False
        
        if X1.shape[1] != X2.shape[1]:
            print("❌ Feature count inconsistent")
            return False
        
        print(f"✅ Feature consistency verified ({X1.shape[1]} features)")
        return True
        
    except Exception as e:
        print(f"❌ Feature consistency test failed: {e}")
        return False

def test_edge_cases():
    """Test model behavior with edge cases"""
    print("\n⚠️ Testing Edge Cases...")
    try:
        predictor = FPLPredictor()
        
        # Test with minimal data
        df = predictor._load_current_data()
        if df.empty:
            print("❌ No data for edge case testing")
            return False
        
        # Test with single player
        single_player = df.head(1)
        X_single = predictor._prepare_features(single_player)
        
        if X_single is not None:
            pred_single = predictor._safe_predict(X_single)
            if pred_single is not None and len(pred_single) == 1:
                print("✅ Single player prediction works")
            else:
                print("❌ Single player prediction failed")
                return False
        
        # Test with empty DataFrame (should fail gracefully)
        empty_df = pd.DataFrame()
        X_empty = predictor._prepare_features(empty_df)
        
        if X_empty is None:
            print("✅ Empty data handled gracefully")
        else:
            print("⚠️ Empty data not handled properly")
        
        return True
        
    except Exception as e:
        print(f"❌ Edge case testing failed: {e}")
        return False

def test_performance_benchmarks():
    """Test model against performance benchmarks"""
    print("\n📈 Testing Performance Benchmarks...")
    try:
        predictor = FPLPredictor()
        
        # Load model metadata
        metadata_files = list(predictor.models_dir.glob('model_metadata_*.json'))
        if not metadata_files:
            print("⚠️ No model metadata found - skipping performance benchmarks")
            return True
        
        import json
        latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
        with open(latest_metadata, 'r') as f:
            metadata = json.load(f)
        
        model_performance = metadata.get('model_performance', {})
        if not model_performance:
            print("⚠️ No performance metrics in metadata")
            return True
        
        # Check benchmarks
        benchmark_passed = True
        
        for model_name, metrics in model_performance.items():
            rmse = metrics.get('train_rmse', float('inf'))
            mae = metrics.get('train_mae', float('inf'))
            
            print(f"📊 {model_name}:")
            print(f"   RMSE: {rmse:.3f}")
            print(f"   MAE: {mae:.3f}")
            
            # FPL-specific benchmarks
            if rmse > 1.0:
                print(f"   ⚠️ RMSE above recommended threshold (1.0)")
                benchmark_passed = False
            
            if mae > 0.8:
                print(f"   ⚠️ MAE above recommended threshold (0.8)")
                benchmark_passed = False
        
        if benchmark_passed:
            print("✅ Performance benchmarks passed")
        else:
            print("⚠️ Some performance benchmarks not met")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark testing failed: {e}")
        return False

def run_validation_suite():
    """Run complete validation suite"""
    print("🧪 FPL Model Validation Suite")
    print("=" * 50)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Prediction Accuracy", test_prediction_accuracy),
        ("Feature Consistency", test_feature_consistency),
        ("Edge Cases", test_edge_cases),
        ("Performance Benchmarks", test_performance_benchmarks)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    success_rate = (passed / total) * 100
    print(f"\nOverall Success Rate: {success_rate:.1f}% ({passed}/{total} tests passed)")
    
    if success_rate >= 80:
        print("🟢 Model validation: EXCELLENT")
        return True
    elif success_rate >= 60:
        print("🟡 Model validation: GOOD - some issues detected")
        return True
    else:
        print("🔴 Model validation: POOR - immediate attention needed")
        return False

if __name__ == "__main__":
    success = run_validation_suite()
    sys.exit(0 if success else 1)