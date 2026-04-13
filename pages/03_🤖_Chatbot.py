"""Gemini-powered chatbot."""

import streamlit as st

from ev_app.sidebar import render_sidebar
from ev_app.state import init_session_state, load_data
from ev_app.ui import disclaimer_research_only, inject_custom_css
from utils import call_gemini, get_dataset_statistics, get_gemini_api_key

st.set_page_config(
    page_title="Chatbot — EV Battery",
    page_icon="🤖",
    layout="wide",
)

inject_custom_css()
init_session_state()
render_sidebar()

st.markdown(
    '<div class="main-header">🤖 AI assistant</div>', unsafe_allow_html=True
)

st.markdown(
    """
Ask about battery health, the dataset, model metrics, or how to use this app.
Requires a **Gemini API key** (see README).
"""
)

api_key = get_gemini_api_key()
if not api_key:
    st.warning(
        """
**Gemini API key not found.**

Set `GEMINI_API_KEY` locally, or add it under Streamlit **Secrets** for deployment.

```powershell
# Windows PowerShell
$env:GEMINI_API_KEY = "your_key_here"
```
"""
    )
    st.stop()

st.success("Gemini API key found.")

df = load_data()
context = ""
if df is not None:
    stats = get_dataset_statistics(df)
    context = f"""
Dataset context:
- Rows: {stats['total_rows']}
- Columns: {', '.join(stats['columns'][:12])}
"""
    if "soh_mean" in stats:
        context += f"""
- Avg SOH: {stats['soh_mean']:.2f}%
- SOH range: {stats['soh_min']:.2f}% – {stats['soh_max']:.2f}%
"""
    if "voltage_mean" in stats:
        context += f"\n- Avg voltage: {stats['voltage_mean']:.2f} V\n"
    if "temperature_mean" in stats:
        context += f"- Avg temperature: {stats['temperature_mean']:.2f} °C\n"
    if stats.get("feature_correlations"):
        top_feature = max(stats["feature_correlations"].items(), key=lambda x: x[1])
        context += f"- Top |corr| with SOH: {top_feature[0]} ({top_feature[1]:.3f})\n"

st.markdown("---")

if st.session_state.chat_history:
    st.markdown("### Conversation")
    for role, message in st.session_state.chat_history:
        prefix = "**You:** " if role == "user" else "**Assistant:** "
        st.markdown(f"{prefix}{message}")
    st.markdown("---")

user_query = st.text_input("Your question:", key="chat_input")

c1, c2 = st.columns([1, 4])
with c1:
    send = st.button("Send", type="primary", use_container_width=True)
with c2:
    if st.button("Clear history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

if send and user_query:
    st.session_state.chat_history.append(("user", user_query))

    system_context = """You are an expert assistant for an EV Battery Health & Range Prediction project.
Be concise. Use dataset context when relevant. Do not invent metrics not supported by the context."""

    full_context = system_context + "\n\n" + context

    with st.spinner("Thinking…"):
        try:
            response = call_gemini(user_query, full_context)
            st.session_state.chat_history.append(("assistant", response))
            st.markdown(f"**Assistant:** {response}")
        except Exception as e:
            err = str(e)
            st.error(err)
            st.session_state.chat_history.append(("assistant", err))

disclaimer_research_only()
