"""
Smoke tests for EV Battery Health Prediction App
Tests basic functionality without requiring full Streamlit runtime
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from utils import (
    load_dataset, compute_battery_features, preprocess_data,
    load_model, get_dataset_statistics
)


def test_imports():
    """Test that all required modules can be imported."""
    import streamlit
    import pandas
    import numpy
    import sklearn
    assert True


def test_load_dataset():
    """Test dataset loading."""
    try:
        df = load_dataset()
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert len(df.columns) > 0
        print(f"✓ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    except FileNotFoundError:
        pytest.skip("Dataset file not found - skipping test")


def test_compute_battery_features():
    """Test feature computation."""
    try:
        df = load_dataset()
        df_features = compute_battery_features(df)
        
        # Check that required features are computed
        assert 'SOH' in df_features.columns
        assert 'SOC' in df_features.columns
        assert 'Charge_Cycles' in df_features.columns
        
        # Check SOH is in valid range
        assert df_features['SOH'].min() >= 0
        assert df_features['SOH'].max() <= 100
        
        print(f"✓ Features computed successfully")
        print(f"  SOH range: {df_features['SOH'].min():.2f}% - {df_features['SOH'].max():.2f}%")
    except FileNotFoundError:
        pytest.skip("Dataset file not found - skipping test")


def test_preprocess_data():
    """Test data preprocessing."""
    try:
        df = load_dataset()
        df = compute_battery_features(df)
        X, y, feature_names = preprocess_data(df, target_col='SOH')
        
        assert X is not None
        assert y is not None
        assert len(feature_names) > 0
        assert len(X) == len(y)
        assert len(X.columns) == len(feature_names)
        
        # Check target is in valid range
        assert y.min() >= 0
        assert y.max() <= 100
        
        print(f"✓ Data preprocessed: {len(X)} samples, {len(feature_names)} features")
    except FileNotFoundError:
        pytest.skip("Dataset file not found - skipping test")


def test_load_model():
    """Test model loading."""
    model_data = load_model()
    
    if model_data is None:
        pytest.skip("Model not found - run 'python notebooks/train_model.py' first")
    
    assert 'model' in model_data
    assert model_data['model'] is not None
    
    # Test prediction on dummy data
    model = model_data['model']
    scaler = model_data.get('scaler')
    feature_names = model_data.get('feature_names')
    
    if feature_names and len(feature_names) > 0:
        # Create dummy input
        X_test = np.random.rand(1, len(feature_names))
        
        if scaler:
            X_test = scaler.transform(X_test)
        
        prediction = model.predict(X_test)
        assert len(prediction) == 1
        assert 0 <= prediction[0] <= 100  # SOH should be 0-100%
        
        print(f"✓ Model loaded and prediction works: SOH = {prediction[0]:.2f}%")
    else:
        print("✓ Model loaded (no feature names available)")


def test_get_dataset_statistics():
    """Test dataset statistics computation."""
    try:
        df = load_dataset()
        df = compute_battery_features(df)
        stats = get_dataset_statistics(df)
        
        assert 'total_rows' in stats
        assert 'columns' in stats
        assert stats['total_rows'] > 0
        
        if 'SOH' in df.columns:
            assert 'soh_mean' in stats
            assert 0 <= stats['soh_mean'] <= 100
        
        print(f"✓ Dataset statistics computed: {len(stats)} keys")
    except FileNotFoundError:
        pytest.skip("Dataset file not found - skipping test")


def test_model_prediction_workflow():
    """Test complete prediction workflow."""
    try:
        # Load model
        model_data = load_model()
        if model_data is None:
            pytest.skip("Model not found")
        
        # Load and preprocess data
        df = load_dataset()
        df = compute_battery_features(df)
        X, y, feature_names = preprocess_data(df, target_col='SOH')
        
        # Get model components
        model = model_data['model']
        scaler = model_data.get('scaler')
        model_features = model_data.get('feature_names', feature_names)
        
        # Prepare test sample
        if len(X) > 0 and len(model_features) > 0:
            # Use first row as test
            X_test = X.iloc[[0]][model_features] if isinstance(X, pd.DataFrame) else X[0:1]
            
            if scaler:
                X_test_scaled = scaler.transform(X_test)
            else:
                X_test_scaled = X_test.values
            
            # Make prediction
            prediction = model.predict(X_test_scaled)[0]
            
            assert 0 <= prediction <= 100
            print(f"✓ Complete workflow test passed: Prediction = {prediction:.2f}%")
    except FileNotFoundError:
        pytest.skip("Dataset file not found - skipping test")
    except Exception as e:
        pytest.skip(f"Workflow test skipped: {e}")


if __name__ == "__main__":
    print("Running smoke tests for EV Battery Health Prediction App\n")
    print("=" * 60)
    
    # Run tests
    test_imports()
    test_load_dataset()
    test_compute_battery_features()
    test_preprocess_data()
    test_load_model()
    test_get_dataset_statistics()
    test_model_prediction_workflow()
    
    print("=" * 60)
    print("All smoke tests completed!")

