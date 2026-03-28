from __future__ import annotations

import pandas as pd
import streamlit as st

from core.data import dataset_cache_key, load_airbnb_bundle
from core.i18n import get_app_title
from core.styles import inject_styles, set_page_config
from pages.chatbot import render_page as render_chatbot_page
from pages.conclusion import render_page as render_conclusion_page
from pages.data_raw import render_page as render_data_raw_page
from pages.eda import render_page as render_eda_page
from pages.login import render_page as render_login_page
from pages.overview import render_page as render_overview_page
from pages.preprocessing import run_processing_pipeline, store_processed_outputs
from pages.register import render_page as render_register_page
from pages.sidebar import render_sidebar
from users import initialize_session_state


def main() -> None:
    initialize_session_state()
    set_page_config(get_app_title())
    inject_styles()

    if not st.session_state["authenticated"]:
        auth_page = st.session_state.get("auth_page", "login")
        if auth_page == "register":
            render_register_page()
        else:
            render_login_page()
        return

    raw_data, cleaned_data, source_label, report = load_airbnb_bundle(dataset_cache_key())

    if "processed_df" not in st.session_state or not isinstance(st.session_state["processed_df"], pd.DataFrame):
        before_frame, df_cleaned, df_scaled, df_ml_ready, processing_report = run_processing_pipeline(raw_data)
        store_processed_outputs(
            before_frame,
            df_cleaned,
            df_scaled,
            df_ml_ready,
            processing_report,
            persist=False,
        )

    # If user has uploaded and processed a file, use that data instead of the default.
    if isinstance(st.session_state.get("processed_df"), pd.DataFrame):
        cleaned_data = st.session_state["processed_df"]
        session_raw_df = st.session_state.get("raw_df")
        if isinstance(session_raw_df, pd.DataFrame):
            raw_data = session_raw_df
        source_label = st.session_state.get("raw_df_name") or source_label
        report = st.session_state.get("processing_report", report)

    page = render_sidebar(source_label, cleaned_data)

    if page == "overview":
        render_overview_page(cleaned_data, source_label)
    elif page == "data_raw":
        render_data_raw_page(raw_data, cleaned_data)
    elif page == "preprocessing":
        render_eda_page(cleaned_data, page_mode="preprocessing")
    elif page == "eda":
        render_eda_page(cleaned_data)
    elif page == "conclusion":
        render_conclusion_page(cleaned_data)
    elif page == "chatbot":
        render_chatbot_page(cleaned_data, source_label)


if __name__ == "__main__":
    main()
