"""Shared Streamlit styling."""

import streamlit as st


def inject_custom_css() -> None:
    st.markdown(
        """
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
    /* Disclaimer footnote */
    .disclaimer {
        font-size: 0.85rem;
        color: #666;
        margin-top: 2rem;
        padding: 0.75rem 1rem;
        background: #f8f9fa;
        border-radius: 0.35rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""",
        unsafe_allow_html=True,
    )


def disclaimer_research_only() -> None:
    st.markdown(
        """
<div class="disclaimer">
<strong>Disclaimer:</strong> This application is for research and demonstration purposes only.
Predictions are not validated for vehicle safety or warranty decisions. Always follow manufacturer
guidance for battery servicing and diagnostics.
</div>
""",
        unsafe_allow_html=True,
    )
