"""Make Prediction page."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ev_app.sidebar import render_sidebar
from ev_app.state import init_session_state, load_data, load_model_data
from ev_app.ui import disclaimer_research_only, inject_custom_css
from utils import compute_battery_features

st.set_page_config(
    page_title="Make Prediction — EV Battery",
    page_icon="🔮",
    layout="wide",
)

inject_custom_css()
init_session_state()
render_sidebar()

st.markdown(
    '<div class="main-header">🔮 Make Prediction</div>', unsafe_allow_html=True
)

model_data = load_model_data()

if model_data is None:
    st.warning(
        """
**No model found.**

Train a model first:

```bash
python notebooks/train_model.py
```
"""
    )
    st.stop()

model = model_data["model"]
scaler = model_data.get("scaler")
feature_names = model_data.get("feature_names", [])
metrics = model_data.get("metrics", {})

r2 = metrics.get("test", {}).get("r2", 0)
mt = metrics.get("model_type", "Unknown")
st.info(f"Model: **{mt}** · Test R²: **{r2:.4f}**")

st.markdown("---")
st.subheader("Input battery parameters")

df = load_data()
defaults = {}
if df is not None:
    if "Voltage" in df.columns:
        defaults["Voltage"] = float(df["Voltage"].median())
    if "Temperature" in df.columns:
        defaults["Temperature"] = float(df["Temperature"].median())
    if "Current" in df.columns:
        defaults["Current"] = float(df["Current"].median())
    if "Charge_Cycles" in df.columns:
        defaults["Charge_Cycles"] = int(df["Charge_Cycles"].median())

col1, col2 = st.columns(2)
input_data = {}
with col1:
    input_data["Voltage"] = st.number_input(
        "Voltage (V)", min_value=0.0, max_value=5.0,
        value=defaults.get("Voltage", 3.0), step=0.1,
    )
    input_data["Temperature"] = st.number_input(
        "Temperature (°C)", min_value=-20.0, max_value=60.0,
        value=defaults.get("Temperature", 25.0), step=0.1,
    )
with col2:
    input_data["Current"] = st.number_input(
        "Current (A)", min_value=-50.0, max_value=50.0,
        value=defaults.get("Current", 0.0), step=0.1,
    )
    input_data["Charge_Cycles"] = st.number_input(
        "Charge cycles", min_value=0, max_value=10000,
        value=int(defaults.get("Charge_Cycles", 0)), step=1,
    )

if feature_names:
    missing_features = [f for f in feature_names if f not in input_data]
    if missing_features:
        st.markdown("#### Additional features")
        for feat in missing_features[:5]:
            if feat not in ("SOH", "SOC", "Remaining_Range_km", "Time"):
                input_data[feat] = st.number_input(feat, value=0.0, key=f"input_{feat}")

st.markdown("---")

if st.button("Predict battery health", type="primary", use_container_width=True):
    try:
        temp_df = pd.DataFrame([input_data])
        if "Time" not in temp_df.columns:
            temp_df["Time"] = 0
        temp_df = compute_battery_features(temp_df)

        if feature_names:
            X_pred = temp_df[feature_names].copy()
        else:
            numeric_cols = temp_df.select_dtypes(include=[np.number]).columns
            exclude_cols = {"SOH", "SOC", "Remaining_Range_km", "Time"}
            X_pred = temp_df[[c for c in numeric_cols if c not in exclude_cols]].copy()

        if scaler:
            X_pred_scaled = scaler.transform(X_pred)
        else:
            X_pred_scaled = X_pred.values

        soh_pred = float(model.predict(X_pred_scaled)[0])
        soh_pred = float(np.clip(soh_pred, 0, 100))

        base_range = 400.0
        soc = float(temp_df["SOC"].iloc[0]) if "SOC" in temp_df.columns else 50.0
        range_pred = float(np.clip(soh_pred * soc * base_range / 10000.0, 0, base_range))

        st.markdown("---")
        st.markdown("### Results")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("SOH", f"{soh_pred:.2f}%")
        with c2:
            st.metric("Estimated range", f"{range_pred:.1f} km")
        with c3:
            st.metric("SOC (estimate)", f"{soc:.1f}%")

        st.markdown("---")
        if soh_pred >= 80:
            status, status_class, emoji = "Excellent", "status-excellent", "🟢"
        elif soh_pred >= 60:
            status, status_class, emoji = "Good", "status-good", "🔵"
        elif soh_pred >= 40:
            status, status_class, emoji = "Moderate", "status-moderate", "🟡"
        else:
            status, status_class, emoji = "Poor", "status-poor", "🔴"

        st.markdown(
            f"### {emoji} Status: <span class='{status_class}'>{status}</span>",
            unsafe_allow_html=True,
        )

        if hasattr(model, "feature_importances_") and feature_names:
            st.markdown("---")
            st.markdown("#### Feature importance")
            importance_df = pd.DataFrame(
                {"Feature": feature_names, "Importance": model.feature_importances_}
            ).sort_values("Importance", ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            top_features = importance_df.head(10)
            ax.barh(top_features["Feature"], top_features["Importance"])
            ax.set_xlabel("Importance")
            ax.set_title("Top 10 — feature importance")
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("#### Values vs importance (this input)")
            fv = X_pred.iloc[0] if isinstance(X_pred, pd.DataFrame) else pd.Series(
                X_pred[0], index=feature_names
            )
            contrib_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Value": fv.values,
                    "Importance": model.feature_importances_,
                }
            ).sort_values("Importance", ascending=False)
            st.dataframe(contrib_df.head(10), use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)

disclaimer_research_only()
