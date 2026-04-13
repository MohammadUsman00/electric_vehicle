"""Shared sidebar for multipage Streamlit app."""

from __future__ import annotations

import streamlit as st

from ev_app.state import load_data


def render_sidebar() -> None:
    """Navigation is automatic via multipage; show dataset controls here."""
    st.sidebar.title("EV Battery Health")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dataset")
    if st.sidebar.button("Load dataset"):
        df = load_data()
        if df is not None:
            st.sidebar.success(
                f"Loaded: {len(df):,} rows, {len(df.columns)} columns"
            )
    st.sidebar.markdown("---")
    st.sidebar.caption("Use the sidebar pages above to navigate the app.")
