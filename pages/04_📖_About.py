"""About / methodology."""

import json
from pathlib import Path

import streamlit as st

from ev_app import __version__
from ev_app.sidebar import render_sidebar
from ev_app.state import init_session_state
from ev_app.ui import disclaimer_research_only, inject_custom_css

st.set_page_config(
    page_title="About — EV Battery",
    page_icon="📖",
    layout="wide",
)

inject_custom_css()
init_session_state()
render_sidebar()

st.markdown('<div class="main-header">📖 About</div>', unsafe_allow_html=True)

st.markdown(
    f"""
## EV Battery Health & Range

**App version:** `{__version__}`

This project demonstrates a classical ML pipeline (feature engineering → scaling →
RandomForest or XGBoost) exposed through Streamlit, with optional Gemini Q&A.

### Goals

1. Predict **SOH** from operational-style inputs  
2. Provide a **simple range heuristic** (not a vehicle-certified range model)  
3. Surface **metrics and plots** for sanity checking the regressor  

### Method notes

- **Train/test split** defaults to random 80/20; for temporal degradation studies, consider time-based splits.  
- **Metrics** (MAE, RMSE, R²) are saved with the model bundle.  
- See `notebooks/train_model.py` for training options (`--xgboost`, `--test-size`, etc.).

### License

See the `LICENSE` file in the repository root (MIT unless you change it).

### Disclaimer

This software is not intended for safety-critical or regulatory use.
"""
)

metrics_path = Path("model") / "model_metrics.json"
if metrics_path.exists():
    try:
        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)
        with st.expander("Saved metrics snapshot (`model/model_metrics.json`)"):
            st.json(m)
    except OSError:
        pass

st.markdown("---")
st.markdown(
    """
**Resources**

- `README.md` — setup and screenshots  
- `example_queries.txt` — sample chatbot prompts  
- `.env.example` — environment template  
"""
)

disclaimer_research_only()
