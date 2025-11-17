"""
Utility functions for EV Battery Health & Range Prediction
Includes data loading, preprocessing, model utilities, and Gemini API integration
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
import json
from typing import Tuple, Dict, Optional, List

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip .env loading

# Gemini API integration
try:
    import google.generativeai as genai
    _HAS_GEMINI = True
except ImportError:
    _HAS_GEMINI = False
    genai = None


def get_gemini_api_key() -> Optional[str]:
    """Get Gemini API key from environment or Streamlit secrets."""
    try:
        import streamlit as st
        # Try Streamlit secrets first
        if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
            return st.secrets['GEMINI_API_KEY']
    except:
        pass
    
    # Fallback to environment variable
    return os.environ.get('GEMINI_API_KEY')


def call_gemini(prompt: str, context: str = "") -> str:
    """
    Call Google Gemini API with a prompt and optional context.
    
    Args:
        prompt: User's question/prompt
        context: Additional context (e.g., dataset statistics) to include
    
    Returns:
        Response text from Gemini
    """
    api_key = get_gemini_api_key()
    
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Please set it as an environment variable "
            "or in Streamlit secrets (st.secrets['GEMINI_API_KEY'])."
        )
    
    if not _HAS_GEMINI:
        raise RuntimeError(
            "google-generativeai package not installed. "
            "Install it with: pip install google-generativeai"
        )
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        full_prompt = f"{context}\n\nUser Question: {prompt}\n\nPlease provide a helpful answer."
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        raise RuntimeError(f"Error calling Gemini API: {str(e)}")


def find_dataset_file(data_dir: str = "data") -> Optional[str]:
    """
    Auto-detect CSV dataset file in data directory or root.
    Looks for files with battery-related column names.
    
    Args:
        data_dir: Directory to search for dataset
        
    Returns:
        Path to dataset file or None if not found
    """
    data_path = Path(data_dir)
    
    # Check data directory first (preferred location)
    csv_files = []
    if data_path.exists():
        csv_files = list(data_path.glob("*.csv"))
    
    # Also check root directory
    root_csvs = list(Path(".").glob("*.csv"))
    
    all_csvs = csv_files + root_csvs  # Prefer data/ directory files
    
    if not all_csvs:
        return None
    
    # Look for battery-related keywords in column names
    battery_keywords = ['soh', 'soc', 'voltage', 'temperature', 'current', 'charge', 'cycle']
    
    for csv_file in all_csvs:
        try:
            df_sample = pd.read_csv(csv_file, nrows=5)
            cols_lower = [col.lower() for col in df_sample.columns]
            if any(keyword in ' '.join(cols_lower) for keyword in battery_keywords):
                return str(csv_file)
        except:
            continue
    
    # If no match, return first CSV found
    return str(all_csvs[0])


def load_dataset(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load the EV battery dataset.
    
    Args:
        filepath: Optional path to CSV file. If None, auto-detects.
        
    Returns:
        DataFrame with battery data
    """
    if filepath is None:
        filepath = find_dataset_file()
    
    if filepath is None:
        raise FileNotFoundError(
            "No dataset file found. Please place a CSV file in the 'data/' directory "
            "or in the root directory."
        )
    
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    return df


def compute_battery_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features from raw battery data.
    
    Features computed:
    - SOH (State of Health): Based on capacity degradation
    - SOC (State of Charge): Estimated from voltage
    - Charge_Cycles: Count of charge/discharge cycles
    - C_Rate: Current / nominal capacity
    - Energy: Voltage * Current (power)
    - Capacity: Integrated current over time
    
    Args:
        df: Raw dataframe with Time, Current, Voltage, Temperature columns
        
    Returns:
        DataFrame with additional computed features
    """
    df = df.copy()
    
    # Ensure numeric columns
    numeric_cols = ['Time', 'Current', 'Voltage', 'Temperature']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Forward fill missing values
    df = df.ffill().bfill()
    
    # Compute capacity from current integration (Ah)
    if 'Current' in df.columns and 'Time' in df.columns:
        # Convert time to hours if needed (assuming seconds)
        time_diff = df['Time'].diff().fillna(0)
        # If time values are large, assume seconds; convert to hours
        if len(time_diff) > 0 and time_diff.max() > 100:
            time_diff = time_diff / 3600  # Convert seconds to hours
        
        # Integrate current to get capacity (Ah)
        # Use absolute time differences to avoid issues with negative values
        time_diff = time_diff.abs()
        df['Capacity_Ah'] = (df['Current'].abs() * time_diff).cumsum()
        
        # Nominal capacity (estimate from max capacity seen, or use 2.5 Ah as default)
        nominal_capacity = abs(df['Capacity_Ah']).max() if 'Capacity_Ah' in df.columns else 2.5
        
        # Compute SOH (State of Health) - current capacity / initial capacity
        if 'Capacity_Ah' in df.columns:
            # Use rolling window to estimate current capacity
            window_size = min(100, len(df) // 10)
            if window_size > 1:
                current_capacity = df['Capacity_Ah'].rolling(window=window_size).max() - df['Capacity_Ah'].rolling(window=window_size).min()
                df['SOH'] = (current_capacity / nominal_capacity * 100).fillna(100).clip(0, 100)
            else:
                df['SOH'] = 100.0
        else:
            df['SOH'] = 100.0
        
        # Compute C-Rate (Current / Nominal Capacity)
        df['C_Rate'] = df['Current'] / nominal_capacity if nominal_capacity > 0 else 0
        
        # Count charge cycles (transitions from negative to positive current)
        if 'Current' in df.columns:
            current_sign = np.sign(df['Current'])
            sign_change = (current_sign != current_sign.shift()).astype(int)
            df['Charge_Cycles'] = sign_change.cumsum() // 2  # Each cycle = charge + discharge
        else:
            df['Charge_Cycles'] = 0
    else:
        df['Capacity_Ah'] = 0
        df['SOH'] = 100.0
        df['C_Rate'] = 0
        df['Charge_Cycles'] = 0
    
    # Compute SOC (State of Charge) from voltage (simplified model)
    if 'Voltage' in df.columns:
        # Simple linear model: SOC = (V - V_min) / (V_max - V_min) * 100
        v_min = df['Voltage'].min()
        v_max = df['Voltage'].max()
        if v_max > v_min:
            df['SOC'] = ((df['Voltage'] - v_min) / (v_max - v_min) * 100).clip(0, 100)
        else:
            df['SOC'] = 50.0
    else:
        df['SOC'] = 50.0
    
    # Compute Energy (Power = Voltage * Current)
    if 'Voltage' in df.columns and 'Current' in df.columns:
        df['Energy_W'] = df['Voltage'] * df['Current']
    else:
        df['Energy_W'] = 0
    
    # Compute Remaining Range (km) - simplified: Range = SOH * SOC * base_range / 100
    base_range_km = 400  # Assume 400 km base range
    if 'SOH' in df.columns and 'SOC' in df.columns:
        df['Remaining_Range_km'] = (df['SOH'] * df['SOC'] * base_range_km / 10000).clip(0, base_range_km)
    else:
        df['Remaining_Range_km'] = base_range_km
    
    return df


def preprocess_data(df: pd.DataFrame, target_col: str = 'SOH', 
                   feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Preprocess data for model training.
    
    Args:
        df: Input dataframe
        target_col: Target column name
        feature_cols: Optional list of feature columns. If None, auto-selects.
        
    Returns:
        Tuple of (X, y, feature_names)
    """
    df = df.copy()
    
    # Compute features if not already present
    if 'SOH' not in df.columns:
        df = compute_battery_features(df)
    
    # Handle missing values
    df = df.ffill().bfill().fillna(0)
    
    # Select features
    if feature_cols is None:
        # Auto-select numeric features (exclude target and metadata)
        exclude_cols = [target_col, 'Time', 'Remaining_Range_km']
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols 
                       and pd.api.types.is_numeric_dtype(df[col])]
    
    # Ensure target exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    # Extract features and target
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Remove rows with invalid target values
    valid_mask = y.notna() & (y >= 0) & (y <= 100)  # SOH should be 0-100%
    X = X[valid_mask]
    y = y[valid_mask]
    
    return X, y, feature_cols


def load_model(model_dir: str = "model") -> Optional[Dict]:
    """
    Load saved model and scaler from disk.
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        Dictionary with 'model', 'scaler', 'feature_names', 'metrics', or None if not found
    """
    model_path = Path(model_dir) / "ev_model.pkl"
    
    if not model_path.exists():
        # Try joblib format
        model_path = Path(model_dir) / "ev_model.joblib"
        if not model_path.exists():
            return None
    
    try:
        model_data = joblib.load(model_path)
        
        # Handle different model storage formats
        if isinstance(model_data, dict):
            return model_data
        else:
            # If just the model, return with defaults
            return {
                'model': model_data,
                'scaler': None,
                'feature_names': None,
                'metrics': {}
            }
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def save_model(model, scaler, feature_names: List[str], metrics: Dict, 
              model_dir: str = "model"):
    """
    Save model, scaler, and metadata to disk.
    
    Args:
        model: Trained model object
        scaler: Fitted scaler object
        feature_names: List of feature column names
        metrics: Dictionary of evaluation metrics
        model_dir: Directory to save model files
    """
    model_path = Path(model_dir)
    model_path.mkdir(exist_ok=True)
    
    # Save model bundle
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'metrics': metrics
    }
    
    joblib.dump(model_data, model_path / "ev_model.pkl")
    
    # Save metrics as JSON
    metrics_path = model_path / "model_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def get_dataset_statistics(df: pd.DataFrame) -> Dict:
    """
    Compute dataset statistics for chatbot context.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_rows': len(df),
        'columns': list(df.columns),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns)
    }
    
    # Compute feature statistics
    if 'SOH' in df.columns:
        stats['soh_mean'] = float(df['SOH'].mean())
        stats['soh_min'] = float(df['SOH'].min())
        stats['soh_max'] = float(df['SOH'].max())
        stats['soh_std'] = float(df['SOH'].std())
    
    if 'Voltage' in df.columns:
        stats['voltage_mean'] = float(df['Voltage'].mean())
        stats['voltage_range'] = [float(df['Voltage'].min()), float(df['Voltage'].max())]
    
    if 'Temperature' in df.columns:
        stats['temperature_mean'] = float(df['Temperature'].mean())
        stats['temperature_range'] = [float(df['Temperature'].min()), float(df['Temperature'].max())]
    
    if 'Charge_Cycles' in df.columns:
        stats['max_cycles'] = int(df['Charge_Cycles'].max())
    
    # Feature importance (correlation with SOH if available)
    if 'SOH' in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corrwith(df['SOH']).abs().sort_values(ascending=False)
        stats['feature_correlations'] = {
            col: float(corr) for col, corr in correlations.items() 
            if col != 'SOH' and not pd.isna(corr)
        }
    
    return stats

