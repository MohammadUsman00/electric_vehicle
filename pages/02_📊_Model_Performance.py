"""Model performance diagnostics."""

import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split

from ev_app.sidebar import render_sidebar
from ev_app.state import init_session_state, load_data, load_model_data
from ev_app.ui import disclaimer_research_only, inject_custom_css
from utils import preprocess_data

st.set_page_config(
    page_title="Model Performance — EV Battery",
    page_icon="📊",
    layout="wide",
)

inject_custom_css()
init_session_state()
render_sidebar()

st.markdown(
    '<div class="main-header">📊 Model Performance</div>', unsafe_allow_html=True
)

model_data = load_model_data()
if model_data is None:
    st.warning("No model found. Run `python notebooks/train_model.py` first.")
    st.stop()

metrics = model_data.get("metrics", {})
st.info(f"**Model:** {metrics.get('model_type', 'Unknown')}")

train_m = metrics.get("train", {})
test_m = metrics.get("test", {})

st.markdown("---")
st.subheader("Metrics")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Test R²", f"{test_m.get('r2', 0):.4f}")
with c2:
    st.metric("Test MAE", f"{test_m.get('mae', 0):.4f}")
with c3:
    st.metric("Test RMSE", f"{test_m.get('rmse', 0):.4f}")
with c4:
    st.metric("Train R²", f"{train_m.get('r2', 0):.4f}")

st.markdown("---")
st.subheader("Visualizations")

df = load_data()
if df is None:
    st.info("Load a dataset from the sidebar to plot test-set diagnostics.")
    disclaimer_research_only()
    st.stop()

try:
    X, y, _feat = preprocess_data(df, target_col="SOH")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = model_data.get("scaler")
    if scaler:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test.values

    clf = model_data["model"]
    y_pred = clf.predict(X_test_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].scatter(y_test, y_pred, alpha=0.5)
    axes[0].plot(
        [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2
    )
    axes[0].set_xlabel("Actual SOH (%)")
    axes[0].set_ylabel("Predicted SOH (%)")
    axes[0].set_title("Predicted vs actual")
    axes[0].grid(True, alpha=0.3)

    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5)
    axes[1].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[1].set_xlabel("Predicted SOH (%)")
    axes[1].set_ylabel("Residuals (%)")
    axes[1].set_title("Residuals")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("#### Error distribution")
    fig2, ax = plt.subplots(figsize=(10, 5))
    ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Residual (actual − predicted)")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution")
    ax.axvline(x=0, color="r", linestyle="--", lw=2)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig2)

except Exception as e:
    st.error(f"Could not build plots: {e}")
    st.exception(e)

disclaimer_research_only()
