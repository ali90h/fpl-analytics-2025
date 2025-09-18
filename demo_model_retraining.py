#!/usr/bin/env python3
"""
Demo: Intelligent Weekly Model Retraining System
Demonstrates the automated ML pipeline for FPL prediction model updates
"""

from fpl_predictor import FPLPredictor
from datetime import datetime

def main():
    print("🚀 FPL Intelligent Model Retraining Demo")
    print("=" * 50)
    
    # Initialize predictor
    predictor = FPLPredictor()
    
    print("\n1️⃣ Checking Current Model Status")
    print("-" * 30)
    
    # Check if retraining is needed
    needs_retrain = predictor._should_retrain_models()
    
    if needs_retrain:
        print("✅ Retraining is recommended!")
        
        print("\n2️⃣ Starting Intelligent Model Retraining")
        print("-" * 40)
        
        # Perform retraining
        success = predictor._retrain_models_with_new_data()
        
        if success:
            print("\n🎉 Model Retraining Completed Successfully!")
            print("Your FPL predictions are now using the latest data!")
            
            print("\n3️⃣ Testing New Models")
            print("-" * 20)
            
            # Test a quick prediction
            try:
                predictions = predictor.predict_next_gameweek()
                if predictions is not None:
                    print(f"📊 Generated predictions for {len(predictions)} players")
                    print("🔝 Top 5 Predictions:")
                    for i, (_, player) in enumerate(predictions.head(5).iterrows()):
                        print(f"   {i+1}. {player['web_name']} - {player['predicted_points']:.1f} pts")
                else:
                    print("⚠️ Could not generate test predictions")
            except Exception as e:
                print(f"⚠️ Error testing predictions: {e}")
                
        else:
            print("\n❌ Model Retraining Failed")
            print("Current models will continue to be used.")
            
    else:
        print("✅ Current models are up to date!")
        print("No retraining needed at this time.")
        
        # Show current model info
        try:
            metadata_files = list(predictor.models_dir.glob('model_metadata_*.json'))
            if metadata_files:
                latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
                with open(latest_metadata, 'r') as f:
                    metadata = json.load(f)
                
                print(f"\n📊 Current Model Info:")
                print(f"   Training Data: {metadata.get('dataset_info', {}).get('gameweeks_trained', 'Unknown')}")
                print(f"   Model Type: XGBoost Ensemble")
                print(f"   Features: {metadata.get('dataset_info', {}).get('features_count', 'Unknown')}")
                print(f"   Last Updated: {metadata.get('timestamp', 'Unknown')}")
        except:
            pass
    
    print("\n🔄 Weekly Automation Schedule:")
    print("   • Check for new completed gameweeks")
    print("   • Validate model performance degradation")
    print("   • Retrain if improvement opportunity detected")
    print("   • Deploy new models only if significantly better")
    print("   • Maintain model versioning and rollback capability")
    
    print(f"\n💡 Manual Retraining:")
    print(f"   python fpl_predictor.py retrain-models")
    print(f"   python fpl_predictor.py retrain-models --force")

if __name__ == "__main__":
    import json
    main()