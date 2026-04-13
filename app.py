"""
EV Battery Health & Range Prediction — main entry (Home).
Additional views live under `pages/`.
"""

import streamlit as st

from ev_app import __version__
from ev_app.sidebar import render_sidebar
from ev_app.state import init_session_state, load_data
from ev_app.ui import disclaimer_research_only, inject_custom_css

st.set_page_config(
    page_title="EV Battery Health & Range",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()
init_session_state()
render_sidebar()

st.markdown(
    '<div class="main-header">🔋 EV Battery Health & Range Prediction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
### Welcome

**Version {__version__}** · A Streamlit + scikit-learn / XGBoost demo for predicting **State of Health (SOH)**
and a simple **range estimate** from operational-style inputs.

**Features**
- **Make Prediction** — voltage, temperature, current, charge cycles → SOH and range
- **Model Performance** — train/test metrics and diagnostic plots
- **Chatbot** — optional Gemini-powered Q&A about the app and data (requires API key)

**Quick start**
1. Train a model: `python notebooks/train_model.py`
2. Run: `streamlit run app.py`
3. Place a battery CSV in `data/` (see README)
"""
)

st.markdown("---")
st.subheader("Dataset preview")

df = load_data()

if df is not None:
    st.success(
        f"Dataset ready: **{len(df):,}** rows × **{len(df.columns)}** columns"
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total rows", f"{len(df):,}")
    with c2:
        st.metric("Columns", len(df.columns))
    with c3:
        if "SOH" in df.columns:
            st.metric("Avg SOH", f"{df['SOH'].mean():.1f}%")
    st.dataframe(df.head(), use_container_width=True)
    if st.checkbox("Show describe()", value=False):
        st.dataframe(df.describe(), use_container_width=True)
else:
    st.info("Use **Load dataset** in the sidebar after adding a CSV to `data/`.")

st.markdown("---")
st.markdown("### Links")
st.markdown(
    """
- **Docs:** `README.md`
- **Train model:** `python notebooks/train_model.py`
- **Tests:** `pytest tests/ -q`
"""
)

disclaimer_research_only()
