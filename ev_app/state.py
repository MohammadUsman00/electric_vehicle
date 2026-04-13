"""Session state helpers shared across Streamlit pages."""

from __future__ import annotations

import streamlit as st

from utils import compute_battery_features, load_dataset, load_model


def init_session_state() -> None:
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "model_data" not in st.session_state:
        st.session_state.model_data = None
    if "chat_history" not in st.session_state:
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
            st.warning(
                "No saved model found. Please train a model first using "
                "`python notebooks/train_model.py`"
            )
    return st.session_state.model_data
