from __future__ import annotations

import pandas as pd
import streamlit as st

from core.i18n import get_app_title, nav_label, t
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
from users import get_navigation_pages_for_role, initialize_session_state, user_can_access_page


def _render_upload_required_page(page: str) -> None:
    st.title(nav_label(page))
    st.info(t("common.upload_required"))
    st.caption(t("common.upload_required_detail"))
    if st.button(t("common.go_to_input_data"), type="primary"):
        st.session_state["current_page"] = "data_raw"
        st.rerun()


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

    has_uploaded_data = bool(st.session_state.get("raw_upload_token"))
    raw_data = pd.DataFrame()
    cleaned_data = pd.DataFrame()
    source_label = ""

    if has_uploaded_data:
        session_raw_df = st.session_state.get("raw_df")
        if isinstance(session_raw_df, pd.DataFrame):
            raw_data = session_raw_df.copy()
            source_label = st.session_state.get("raw_df_name") or ""

            if not isinstance(st.session_state.get("processed_df"), pd.DataFrame):
                before_frame, df_cleaned, df_scaled, df_ml_ready, processing_report = run_processing_pipeline(raw_data)
                store_processed_outputs(
                    before_frame,
                    df_cleaned,
                    df_scaled,
                    df_ml_ready,
                    processing_report,
                    persist=False,
                )

        if isinstance(st.session_state.get("processed_df"), pd.DataFrame):
            cleaned_data = st.session_state["processed_df"].copy()

    page = render_sidebar(source_label, cleaned_data, has_uploaded_data=has_uploaded_data)
    if not user_can_access_page(page) and page != "data_raw":
        fallback_pages = get_navigation_pages_for_role()
        page = fallback_pages[0]
        st.session_state["current_page"] = page

    if page == "overview":
        render_overview_page(cleaned_data, source_label)
    elif page == "data_raw":
        render_data_raw_page(raw_data, cleaned_data)
    elif page == "preprocessing":
        if has_uploaded_data:
            render_eda_page(cleaned_data, page_mode="preprocessing")
        else:
            _render_upload_required_page(page)
    elif page == "eda":
        if has_uploaded_data:
            render_eda_page(cleaned_data)
        else:
            _render_upload_required_page(page)
    elif page == "conclusion":
        if has_uploaded_data:
            render_conclusion_page(cleaned_data)
        else:
            _render_upload_required_page(page)
    elif page == "chatbot":
        if has_uploaded_data:
            render_chatbot_page(cleaned_data, source_label)
        else:
            _render_upload_required_page(page)


if __name__ == "__main__":
    main()
