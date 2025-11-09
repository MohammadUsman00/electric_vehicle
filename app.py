# app.py

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# Try to import OpenAI SDK; handle gracefully if it's not installed
try:
    import openai
    _HAS_OPENAI = True
except Exception:
    openai = None
    _HAS_OPENAI = False

# Use environment variables for API keys (do NOT hard-code keys in source)
# Priority: GEMINI_API_KEY (if you want to keep a separate key name) then OPENAI_API_KEY
gemini_key = os.getenv("GEMINI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
if _HAS_OPENAI:
    if gemini_key:
        # NOTE: If your Gemini key is not OpenAI-compatible this will not magically
        # make it work — see README for full Gemini integration steps.
        openai.api_key = gemini_key
    elif openai_api_key:
        openai.api_key = openai_api_key
    else:
        # keep at None; we'll show a warning in the UI if missing
        openai.api_key = None
else:
    # openai package isn't installed; keep the env vars available for display/instructions
    # but avoid accessing openai attributes elsewhere.
    # gemini_key and openai_api_key are already set above from environment.
    pass

# Try to import XGBoost only if available; otherwise we will disable that option
try:
    from xgboost import XGBRegressor
    _HAS_XGBOOST = True
except Exception:
    XGBRegressor = None
    _HAS_XGBOOST = False

# Title
st.title("🔋 AI-Powered EV Range Estimation")

# Sidebar
st.sidebar.header("Upload Battery Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Chatbot
st.sidebar.markdown("### 🤖 Ask the AI Assistant")
user_input = st.sidebar.text_input("Ask a question about EV batteries or ML")

if user_input:
    if not _HAS_OPENAI:
        st.sidebar.error("The 'openai' Python package is not installed. Install dependencies with: pip install -r requirements.txt")
    elif not openai.api_key:
        st.sidebar.error("OpenAI API key not found. Set the OPENAI_API_KEY or GEMINI_API_KEY environment variable to enable the assistant.")
    else:
        with st.spinner("Thinking..."):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert in EV battery analytics and machine learning."},
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=400,
                )
                text = response['choices'][0]['message']['content']
            except Exception as e:
                text = f"Error calling OpenAI API: {e}"
            st.sidebar.success(text)

# Main App
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Raw Data Preview")
    st.write(df.head())

    st.subheader("🔍 Data Visualization")
    if st.checkbox("Show Correlation Heatmap"):
        corr = df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    st.subheader("⚙️ Model Training")
    target = st.selectbox("Select Target Variable", df.columns)
    features = st.multiselect("Select Feature Columns", [col for col in df.columns if col != target])

    if features and target:
        # Ensure selected features are numeric where possible
        X = df[features].apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(df[target], errors='coerce')
        valid_rows = X.notna().all(axis=1) & y.notna()
        if valid_rows.sum() < 2:
            st.error("Not enough valid numeric rows after converting selected columns. Check your data or choose different columns.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X[valid_rows], y[valid_rows], test_size=0.2, random_state=42)

            model_options = ["Linear Regression", "Random Forest"]
            if _HAS_XGBOOST:
                model_options.append("XGBoost")

            model_choice = st.selectbox("Choose Model", model_options)

            if st.button("Train Model"):
                if model_choice == "Linear Regression":
                    model = LinearRegression()
                elif model_choice == "Random Forest":
                    model = RandomForestRegressor()
                else:
                    # XGBoost is available only if imported successfully
                    model = XGBRegressor()

                with st.spinner("Training model..."):
                    model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                st.subheader("📈 Model Evaluation")
                st.write(f"R² Score: {r2_score(y_test, y_pred):.3f}")
                st.write(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.3f}")
                st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")

                st.subheader("📉 Predicted vs Actual")
                fig2, ax2 = plt.subplots()
                ax2.scatter(y_test, y_pred, alpha=0.7)
                ax2.set_xlabel("Actual")
                ax2.set_ylabel("Predicted")
                ax2.set_title("Actual vs Predicted")
                st.pyplot(fig2)

                # Feature importance if available
                if hasattr(model, "feature_importances_"):
                    st.subheader("🔍 Feature Importance")
                    importance_df = pd.DataFrame({
                        "Feature": features,
                        "Importance": model.feature_importances_
                    }).sort_values(by="Importance", ascending=False)
                    st.bar_chart(importance_df.set_index("Feature"))

                # Persist model and metadata in session_state
                st.session_state['model'] = model
                st.session_state['model_features'] = features
                st.session_state['model_target'] = target

                # Option to save model to disk
                if st.checkbox("Save trained model to disk"):
                    model_path = st.text_input("Model filename", value="trained_model.joblib")
                    if st.button("Save now"):
                        try:
                            joblib.dump({
                                'model': model,
                                'features': features,
                                'target': target
                            }, model_path)
                            st.success(f"Model saved to {model_path}")
                        except Exception as e:
                            st.error(f"Failed to save model: {e}")

    # Option to load a previously saved model
    st.markdown("### Load a saved model (optional)")
    load_file = st.file_uploader("Upload saved model (joblib or pkl)", type=["joblib", "pkl"], key="load_model")
    if load_file:
        try:
            loaded = joblib.load(load_file)
            if isinstance(loaded, dict) and 'model' in loaded:
                st.session_state['model'] = loaded['model']
                st.session_state['model_features'] = loaded.get('features', [])
                st.session_state['model_target'] = loaded.get('target', 'target')
            else:
                st.session_state['model'] = loaded
            st.success("Model loaded into session state. You can now make predictions.")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

    st.subheader("🔮 Make a Prediction")
    # Prediction UI uses the trained model in session_state
    if 'model' not in st.session_state:
        st.info("Train a model first (or load a saved model) to enable predictions.")
    else:
        model = st.session_state['model']
        model_features = st.session_state.get('model_features', [])
        model_target = st.session_state.get('model_target', 'target')

        input_data = {}
        for feature in model_features:
            input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

        if st.button("Predict Range/Health"):
            try:
                input_df = pd.DataFrame([input_data])[model_features]
                prediction = model.predict(input_df)[0]
                st.success(f"Predicted {model_target}: {prediction:.2f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
