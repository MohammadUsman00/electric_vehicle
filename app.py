"""
EV Battery Health & Range Prediction - Streamlit Web App
Complete production-style app with prediction, model performance, and AI chatbot
"""

import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

# Import utilities
from utils import (
    load_dataset, compute_battery_features, preprocess_data,
    load_model, get_dataset_statistics, call_gemini, get_gemini_api_key
)

# Page configuration
st.set_page_config(
    page_title="EV Battery Health & Range Prediction",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-excellent { color: #28a745; font-weight: bold; }
    .status-good { color: #17a2b8; font-weight: bold; }
    .status-moderate { color: #ffc107; font-weight: bold; }
    .status-poor { color: #dc3545; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'model_data' not in st.session_state:
    st.session_state.model_data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def load_data():
    """Load dataset and cache in session state."""
    if st.session_state.dataset is None:
        try:
            df = load_dataset()
            df = compute_battery_features(df)
            st.session_state.dataset = df
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return None
    return st.session_state.dataset


def load_model_data():
    """Load model and cache in session state."""
    if st.session_state.model_data is None:
        model_data = load_model()
        if model_data:
            st.session_state.model_data = model_data
        else:
            st.warning("No saved model found. Please train a model first using notebooks/train_model.py")
    return st.session_state.model_data


# Sidebar navigation
st.sidebar.title("🔋 EV Battery Health")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Navigate",
    ["Home", "Make Prediction", "Model Performance", "Chatbot", "About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Info")
if st.sidebar.button("Load Dataset"):
    df = load_data()
    if df is not None:
        st.sidebar.success(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "Home":
    st.markdown('<div class="main-header">🔋 EV Battery Health & Range Prediction</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the EV Battery Health Prediction System
    
    This application uses machine learning to predict battery health (State of Health - SOH) 
    and estimate remaining driving range for electric vehicles based on operational data.
    
    **Key Features:**
    - 🔮 **Predict Battery Health**: Input battery parameters to get SOH predictions
    - 📊 **Model Performance**: View detailed metrics and visualizations
    - 🤖 **AI Chatbot**: Ask questions about the project and dataset
    - 📈 **Data Analysis**: Explore battery data and feature importance
    
    **How to Use:**
    1. Navigate to **Make Prediction** to input battery parameters and get predictions
    2. Check **Model Performance** to see how well the model performs
    3. Use the **Chatbot** to ask questions about battery health, the dataset, or the model
    4. Visit **About** for more information about the project
    """)
    
    st.markdown("---")
    
    # Load and show dataset sample
    st.subheader("📊 Dataset Preview")
    df = load_data()
    
    if df is not None:
        st.success(f"✅ Dataset loaded successfully: **{len(df)}** rows, **{len(df.columns)}** columns")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            if 'SOH' in df.columns:
                st.metric("Avg SOH", f"{df['SOH'].mean():.1f}%")
        
        st.markdown("#### First 5 Rows")
        st.dataframe(df.head(), use_container_width=True)
        
        # Show basic statistics
        if st.checkbox("Show Dataset Statistics"):
            st.markdown("#### Dataset Statistics")
            st.dataframe(df.describe(), use_container_width=True)
    else:
        st.info("💡 Click 'Load Dataset' in the sidebar to load the battery dataset.")
    
    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("""
    - **GitHub Repository**: [Add your repo link here]
    - **Documentation**: See README.md for detailed instructions
    - **Model Training**: Run `python notebooks/train_model.py` to train a new model
    """)


# ============================================================================
# PREDICTION PAGE
# ============================================================================
elif page == "Make Prediction":
    st.markdown('<div class="main-header">🔮 Make Prediction</div>', unsafe_allow_html=True)
    
    model_data = load_model_data()
    
    if model_data is None:
        st.warning("""
        ⚠️ **No model found!**
        
        Please train a model first by running:
        ```bash
        python notebooks/train_model.py
        ```
        
        This will create a model file in the `model/` directory.
        """)
        st.stop()
    
    model = model_data['model']
    scaler = model_data.get('scaler')
    feature_names = model_data.get('feature_names', [])
    metrics = model_data.get('metrics', {})
    
    st.info(f"✅ Model loaded: **{metrics.get('model_type', 'Unknown')}** | Test R²: **{metrics.get('test', {}).get('r2', 0):.3f}**")
    
    st.markdown("---")
    
    # Prediction form
    st.subheader("Input Battery Parameters")
    
    # Get default values from dataset if available
    df = load_data()
    defaults = {}
    if df is not None:
        if 'Voltage' in df.columns:
            defaults['Voltage'] = float(df['Voltage'].median())
        if 'Temperature' in df.columns:
            defaults['Temperature'] = float(df['Temperature'].median())
        if 'Current' in df.columns:
            defaults['Current'] = float(df['Current'].median())
        if 'Charge_Cycles' in df.columns:
            defaults['Charge_Cycles'] = int(df['Charge_Cycles'].median())
    
    # Create input fields
    col1, col2 = st.columns(2)
    
    input_data = {}
    with col1:
        input_data['Voltage'] = st.number_input(
            "Voltage (V)", 
            min_value=0.0, 
            max_value=5.0, 
            value=defaults.get('Voltage', 3.0),
            step=0.1
        )
        input_data['Temperature'] = st.number_input(
            "Temperature (°C)", 
            min_value=-20.0, 
            max_value=60.0, 
            value=defaults.get('Temperature', 25.0),
            step=0.1
        )
    
    with col2:
        input_data['Current'] = st.number_input(
            "Current (A)", 
            min_value=-50.0, 
            max_value=50.0, 
            value=defaults.get('Current', 0.0),
            step=0.1
        )
        input_data['Charge_Cycles'] = st.number_input(
            "Charge Cycles", 
            min_value=0, 
            max_value=10000, 
            value=int(defaults.get('Charge_Cycles', 0)),
            step=1
        )
    
    # Add additional feature inputs if needed
    if feature_names:
        missing_features = [f for f in feature_names if f not in input_data]
        if missing_features:
            st.markdown("#### Additional Features")
            for feat in missing_features[:5]:  # Limit to 5 additional features
                if feat not in ['SOH', 'SOC', 'Remaining_Range_km', 'Time']:
                    input_data[feat] = st.number_input(f"{feat}", value=0.0, key=f"input_{feat}")
    
    st.markdown("---")
    
    if st.button("🔮 Predict Battery Health", type="primary", use_container_width=True):
        try:
            # Create a temporary dataframe for feature computation
            temp_df = pd.DataFrame([input_data])
            
            # Compute derived features
            if 'Time' not in temp_df.columns:
                temp_df['Time'] = 0
            
            temp_df = compute_battery_features(temp_df)
            
            # Prepare features for prediction
            if feature_names:
                # Use model's expected features
                X_pred = temp_df[feature_names].copy()
            else:
                # Fallback: use available numeric features
                numeric_cols = temp_df.select_dtypes(include=[np.number]).columns
                exclude_cols = ['SOH', 'SOC', 'Remaining_Range_km', 'Time']
                X_pred = temp_df[[c for c in numeric_cols if c not in exclude_cols]].copy()
            
            # Scale features if scaler available
            if scaler:
                X_pred_scaled = scaler.transform(X_pred)
            else:
                X_pred_scaled = X_pred.values
            
            # Make prediction
            soh_pred = model.predict(X_pred_scaled)[0]
            soh_pred = np.clip(soh_pred, 0, 100)  # Ensure SOH is between 0-100%
            
            # Estimate remaining range (simplified: Range = SOH * SOC * base_range / 100)
            base_range = 400  # km
            soc = temp_df['SOC'].iloc[0] if 'SOC' in temp_df.columns else 50
            range_pred = (soh_pred * soc * base_range / 10000)
            range_pred = np.clip(range_pred, 0, base_range)
            
            # Display results
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Battery Health (SOH)", f"{soh_pred:.2f}%")
            
            with col2:
                st.metric("Estimated Range", f"{range_pred:.1f} km")
            
            with col3:
                st.metric("State of Charge (SOC)", f"{soc:.1f}%")
            
            # Status indicator
            st.markdown("---")
            if soh_pred >= 80:
                status = "Excellent"
                status_class = "status-excellent"
                emoji = "🟢"
            elif soh_pred >= 60:
                status = "Good"
                status_class = "status-good"
                emoji = "🔵"
            elif soh_pred >= 40:
                status = "Moderate"
                status_class = "status-moderate"
                emoji = "🟡"
            else:
                status = "Poor"
                status_class = "status-poor"
                emoji = "🔴"
            
            st.markdown(f"### {emoji} Battery Status: <span class='{status_class}'>{status}</span>", unsafe_allow_html=True)
            
            # Feature importance visualization
            if hasattr(model, 'feature_importances_') and feature_names:
                st.markdown("---")
                st.markdown("#### 🔍 Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                top_features = importance_df.head(10)
                ax.barh(top_features['Feature'], top_features['Importance'])
                ax.set_xlabel('Importance')
                ax.set_title('Top 10 Feature Importance')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show which features contributed most to this prediction
                st.markdown("#### 📈 Feature Contributions (for this prediction)")
                feature_values = X_pred.iloc[0] if isinstance(X_pred, pd.DataFrame) else pd.Series(X_pred[0], index=feature_names)
                contrib_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Value': feature_values.values,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                st.dataframe(contrib_df.head(10), use_container_width=True)
        
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.exception(e)


# ============================================================================
# MODEL PERFORMANCE PAGE
# ============================================================================
elif page == "Model Performance":
    st.markdown('<div class="main-header">📊 Model Performance</div>', unsafe_allow_html=True)
    
    model_data = load_model_data()
    
    if model_data is None:
        st.warning("No model found. Please train a model first using `python notebooks/train_model.py`")
        st.stop()
    
    metrics = model_data.get('metrics', {})
    model_type = metrics.get('model_type', 'Unknown')
    
    st.info(f"**Model Type:** {model_type}")
    
    # Display metrics
    st.markdown("---")
    st.subheader("📈 Performance Metrics")
    
    train_metrics = metrics.get('train', {})
    test_metrics = metrics.get('test', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Test R² Score", f"{test_metrics.get('r2', 0):.4f}")
    with col2:
        st.metric("Test MAE", f"{test_metrics.get('mae', 0):.4f}")
    with col3:
        st.metric("Test RMSE", f"{test_metrics.get('rmse', 0):.4f}")
    with col4:
        st.metric("Training R²", f"{train_metrics.get('r2', 0):.4f}")
    
    # Load test data for visualization
    st.markdown("---")
    st.subheader("📉 Performance Visualizations")
    
    df = load_data()
    if df is not None:
        try:
            # Prepare data
            X, y, feature_names = preprocess_data(df, target_col='SOH')
            
            # Split data (same as training)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale
            scaler = model_data.get('scaler')
            if scaler:
                X_test_scaled = scaler.transform(X_test)
            else:
                X_test_scaled = X_test.values
            
            # Predictions
            model = model_data['model']
            y_pred = model.predict(X_test_scaled)
            
            # Predicted vs Actual plot
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Scatter plot
            axes[0].scatter(y_test, y_pred, alpha=0.5)
            axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[0].set_xlabel('Actual SOH (%)')
            axes[0].set_ylabel('Predicted SOH (%)')
            axes[0].set_title('Predicted vs Actual SOH')
            axes[0].grid(True, alpha=0.3)
            
            # Residual plot
            residuals = y_test - y_pred
            axes[1].scatter(y_pred, residuals, alpha=0.5)
            axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
            axes[1].set_xlabel('Predicted SOH (%)')
            axes[1].set_ylabel('Residuals (%)')
            axes[1].set_title('Residual Plot')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Error distribution
            st.markdown("#### Error Distribution")
            fig2, ax = plt.subplots(figsize=(10, 5))
            ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Residuals (Actual - Predicted)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Prediction Errors')
            ax.axvline(x=0, color='r', linestyle='--', lw=2)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig2)
            
        except Exception as e:
            st.error(f"Error generating visualizations: {str(e)}")
            st.exception(e)
    else:
        st.info("Load dataset to see performance visualizations.")


# ============================================================================
# CHATBOT PAGE
# ============================================================================
elif page == "Chatbot":
    st.markdown('<div class="main-header">🤖 AI Chatbot Assistant</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Ask questions about:
    - Battery health and degradation
    - The dataset and its features
    - Model performance and predictions
    - EV battery technology
    - How to use this application
    """)
    
    # Check for API key
    api_key = get_gemini_api_key()
    
    if not api_key:
        st.warning("""
        ⚠️ **Gemini API Key Not Found**
        
        To enable the chatbot, please set the `GEMINI_API_KEY` environment variable:
        
        **For local development:**
        ```bash
        # Windows (PowerShell)
        $env:GEMINI_API_KEY = "your_api_key_here"
        
        # Linux/Mac
        export GEMINI_API_KEY="your_api_key_here"
        ```
        
        **For Streamlit Cloud:**
        1. Go to your Streamlit Cloud dashboard
        2. Select your app
        3. Go to Settings → Secrets
        4. Add: `GEMINI_API_KEY = "your_api_key_here"`
        
        **For GitHub Actions:**
        Add `GEMINI_API_KEY` as a repository secret in GitHub Settings → Secrets.
        """)
        st.stop()
    
    st.success("✅ Gemini API key found. Chatbot is ready!")
    
    # Load dataset statistics for context
    df = load_data()
    context = ""
    if df is not None:
        stats = get_dataset_statistics(df)
        context = f"""
        Dataset Context:
        - Total rows: {stats['total_rows']}
        - Columns: {', '.join(stats['columns'][:10])}
        """
        if 'soh_mean' in stats:
            context += f"""
        - Average SOH: {stats['soh_mean']:.2f}%
        - SOH range: {stats['soh_min']:.2f}% - {stats['soh_max']:.2f}%
        """
        if 'voltage_mean' in stats:
            context += f"""
        - Average Voltage: {stats['voltage_mean']:.2f}V
        """
        if 'temperature_mean' in stats:
            context += f"""
        - Average Temperature: {stats['temperature_mean']:.2f}°C
        """
        if 'feature_correlations' in stats:
            top_feature = max(stats['feature_correlations'].items(), key=lambda x: x[1])
            context += f"""
        - Most correlated feature with SOH: {top_feature[0]} (correlation: {top_feature[1]:.3f})
        """
    
    # Chat interface
    st.markdown("---")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 💬 Conversation History")
        for i, (role, message) in enumerate(st.session_state.chat_history):
            if role == "user":
                st.markdown(f"**You:** {message}")
            else:
                st.markdown(f"**Assistant:** {message}")
            st.markdown("---")
    
    # User input
    user_query = st.text_input("Ask a question:", key="chat_input")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        send_button = st.button("Send", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    if send_button and user_query:
        # Add user message to history
        st.session_state.chat_history.append(("user", user_query))
        
        # Get response from Gemini
        with st.spinner("🤔 Thinking..."):
            try:
                # Build prompt with project context
                system_context = """You are an expert assistant for an EV Battery Health & Range Prediction project.
                You help users understand battery health, the dataset, model performance, and EV battery technology.
                When answering data-driven questions, cite specific statistics and features from the dataset context provided.
                Be concise, accurate, and helpful."""
                
                full_context = system_context + "\n\n" + context
                response = call_gemini(user_query, full_context)
                
                # Add assistant response to history
                st.session_state.chat_history.append(("assistant", response))
                
                # Show response
                st.markdown(f"**Assistant:** {response}")
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append(("assistant", error_msg))


# ============================================================================
# ABOUT PAGE
# ============================================================================
elif page == "About":
    st.markdown('<div class="main-header">📖 About</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## EV Battery Health & Range Prediction System
    
    This application uses machine learning to predict the State of Health (SOH) of electric vehicle batteries 
    and estimate remaining driving range based on operational parameters.
    
    ### 🎯 Project Goals
    
    1. **Predict Battery Health**: Estimate SOH percentage from voltage, temperature, current, and charge cycles
    2. **Estimate Range**: Calculate remaining driving range based on battery health and state of charge
    3. **Feature Analysis**: Identify which factors most influence battery degradation
    4. **User-Friendly Interface**: Provide an intuitive web interface for predictions and analysis
    
    ### 🔧 Technical Details
    
    **Model:**
    - Algorithm: RandomForest or XGBoost Regressor
    - Target: State of Health (SOH) percentage (0-100%)
    - Features: Voltage, Temperature, Current, Charge Cycles, and derived features
    
    **Features Computed:**
    - **SOH (State of Health)**: Current capacity / Initial capacity × 100%
    - **SOC (State of Charge)**: Estimated from voltage
    - **Charge Cycles**: Count of charge/discharge cycles
    - **C-Rate**: Current / Nominal capacity
    - **Energy**: Voltage × Current (power)
    
    ### 📊 Dataset
    
    The dataset contains experimental battery data with the following columns:
    - **Time**: Timestamp or time index
    - **Current**: Battery current (A)
    - **Voltage**: Battery voltage (V)
    - **Temperature**: Battery temperature (°C)
    
    Additional features are computed automatically during preprocessing.
    
    ### 🚀 Getting Started
    
    1. **Install Dependencies:**
       ```bash
       pip install -r requirements.txt
       ```
    
    2. **Train Model (if not already trained):**
       ```bash
       python notebooks/train_model.py
       ```
    
    3. **Run Application:**
       ```bash
       streamlit run app.py
       ```
    
    4. **Set Gemini API Key (for chatbot):**
       ```bash
       # Windows PowerShell
       $env:GEMINI_API_KEY = "your_key_here"
       
       # Linux/Mac
       export GEMINI_API_KEY="your_key_here"
       ```
    
    ### 🔒 Security
    
    - API keys are stored as environment variables or Streamlit secrets
    - Never commit API keys to the repository
    - Use `.gitignore` to exclude sensitive files
    
    ### 📝 License
    
    [Add your license information here]
    
    ### 👥 Contributors
    
    [Add contributor information here]
    
    ### 📧 Contact
    
    [Add contact information here]
    """)
    
    st.markdown("---")
    st.markdown("### 📚 Additional Resources")
    st.markdown("""
    - **README.md**: Detailed documentation and setup instructions
    - **Model Training**: See `notebooks/train_model.py` for training script
    - **Example Queries**: Check `example_queries.txt` for chatbot examples
    """)
