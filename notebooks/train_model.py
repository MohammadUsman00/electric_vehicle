"""
Model Training Script for EV Battery Health Prediction
Trains a baseline RandomForest or XGBoost model and saves it to model/
"""

import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import json

from utils import load_dataset, compute_battery_features, preprocess_data, save_model

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available, using RandomForest")


def train_model(use_xgboost: bool = False, test_size: float = 0.2, random_state: int = 42):
    """
    Train a battery health prediction model.
    
    Args:
        use_xgboost: If True, use XGBoost; otherwise use RandomForest
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
    """
    print("Loading dataset...")
    df = load_dataset()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Compute battery features
    print("Computing battery features...")
    df = compute_battery_features(df)
    
    print(f"Features computed. New shape: {df.shape}")
    print(f"New columns: {list(df.columns)}")
    
    # Preprocess data
    print("Preprocessing data...")
    X, y, feature_names = preprocess_data(df, target_col='SOH')
    
    print(f"Features: {feature_names}")
    print(f"Target (SOH) statistics:")
    print(f"  Mean: {y.mean():.2f}%")
    print(f"  Min: {y.min():.2f}%")
    print(f"  Max: {y.max():.2f}%")
    print(f"  Std: {y.std():.2f}%")
    
    # Split data
    print(f"\nSplitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print(f"\nTraining {'XGBoost' if use_xgboost and HAS_XGBOOST else 'RandomForest'} model...")
    
    if use_xgboost and HAS_XGBOOST:
        model = XGBRegressor(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    metrics = {
        'train': {
            'mae': float(train_mae),
            'rmse': float(train_rmse),
            'r2': float(train_r2)
        },
        'test': {
            'mae': float(test_mae),
            'rmse': float(test_rmse),
            'r2': float(test_r2)
        },
        'model_type': 'XGBoost' if use_xgboost and HAS_XGBOOST else 'RandomForest',
        'n_features': len(feature_names),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test)
    }
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"\nTraining Set:")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  R²:   {train_r2:.4f}")
    
    print(f"\nTest Set:")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    print("="*50)
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))
    
    # Save model
    print(f"\nSaving model to model/ directory...")
    save_model(model, scaler, feature_names, metrics)
    
    print("Model training complete!")
    print(f"Model saved to: model/ev_model.pkl")
    print(f"Metrics saved to: model/model_metrics.json")
    
    return model, scaler, feature_names, metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train EV Battery Health Prediction Model")
    parser.add_argument("--xgboost", action="store_true", help="Use XGBoost instead of RandomForest")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    train_model(
        use_xgboost=args.xgboost,
        test_size=args.test_size,
        random_state=args.random_state
    )

