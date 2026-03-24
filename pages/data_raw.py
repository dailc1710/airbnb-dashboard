from __future__ import annotations

import hashlib
import io
import pandas as pd
import plotly.express as px
import streamlit as st

from core.data import (
    build_missing_table,
)
from core.i18n import localize_dataframe_for_display, t, translate_room_type
from pages.preprocessing import run_processing_pipeline, store_processed_outputs


def _clear_preprocessing_session_state() -> None:
    for key in (
        "preprocessing_before_df",
        "processing_report",
        "processed_df",
        "processed_scaled_df",
        "processed_ml_df",
        "cleaned_data",
    ):
        st.session_state.pop(key, None)


def filter_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    filtered = frame.copy()
    filter_cols = st.columns(4)

    if "neighbourhood_group" in frame.columns:
        area_options = sorted(frame["neighbourhood_group"].dropna().astype(str).unique().tolist())
        selected_areas = filter_cols[0].multiselect(t("raw.filter.neighborhood_group"), area_options)
        if selected_areas:
            filtered = filtered[filtered["neighbourhood_group"].isin(selected_areas)]

    if "room_type" in frame.columns:
        room_options = sorted(frame["room_type"].dropna().astype(str).unique().tolist())
        selected_rooms = filter_cols[1].multiselect(
            t("raw.filter.room_type"),
            room_options,
            format_func=translate_room_type,
        )
        if selected_rooms:
            filtered = filtered[filtered["room_type"].isin(selected_rooms)]

    if "price" in frame.columns and not filtered.empty and not filtered["price"].dropna().empty:
        min_price = float(filtered["price"].min())
        max_price = float(filtered["price"].max())
        if min_price != max_price:
            price_range = filter_cols[2].slider(
                t("raw.filter.price_range"),
                min_price,
                max_price,
                (min_price, max_price),
            )
            filtered = filtered[filtered["price"].between(price_range[0], price_range[1])]

    if "number_of_reviews" in frame.columns and not filtered.empty and not filtered["number_of_reviews"].dropna().empty:
        review_cap = int(filtered["number_of_reviews"].max())
        review_threshold = filter_cols[3].slider(t("raw.filter.minimum_reviews"), 0, review_cap, 0)
        filtered = filtered[filtered["number_of_reviews"] >= review_threshold]

    return filtered


def render_page(raw_frame: pd.DataFrame, cleaned_frame: pd.DataFrame) -> None:
    _ = cleaned_frame
    st.title(t("raw.title"))
    st.caption(t("raw.caption"))

    session_raw_frame = st.session_state.get("raw_df")
    session_raw_name = st.session_state.get("raw_df_name")
    audit_frame = session_raw_frame.copy() if isinstance(session_raw_frame, pd.DataFrame) else raw_frame.copy()
    uploaded_file = st.file_uploader(
        t("raw.upload.label"),
        type=["csv"],
        help=t("raw.upload.help"),
    )
    if uploaded_file is not None:
        try:
            uploaded_bytes = uploaded_file.getvalue()
            upload_token = hashlib.md5(uploaded_bytes).hexdigest()
            if st.session_state.get("raw_upload_token") != upload_token:
                audit_frame = pd.read_csv(io.BytesIO(uploaded_bytes))
                st.session_state["raw_df"] = audit_frame.copy()
                st.session_state["raw_df_name"] = uploaded_file.name
                st.session_state["raw_upload_token"] = upload_token
                _clear_preprocessing_session_state()
                before_frame, df_cleaned, df_scaled, df_ml_ready, processing_report = run_processing_pipeline(audit_frame)
                store_processed_outputs(before_frame, df_cleaned, df_scaled, df_ml_ready, processing_report)
                st.rerun()
            else:
                audit_frame = session_raw_frame.copy() if isinstance(session_raw_frame, pd.DataFrame) else pd.read_csv(io.BytesIO(uploaded_bytes))
            st.caption(t("raw.source.uploaded", file_name=uploaded_file.name))
        except Exception as exc:
            audit_frame = raw_frame
            st.error(t("raw.upload.error", error=str(exc)))
    elif isinstance(session_raw_frame, pd.DataFrame):
        st.caption(t("raw.source.uploaded", file_name=session_raw_name or "uploaded CSV"))
    else:
        st.session_state["raw_df"] = raw_frame.copy()
        st.session_state["raw_df_name"] = None
        st.caption(t("raw.source.default"))

    tab_overview, tab_sort_data = st.tabs(
        [
            "Overview",
            t("raw.tab.cleaned"),
        ]
    )

    with tab_overview:
        overview_metrics = st.columns(3)
        overview_metrics[0].metric("Total Rows", f"{len(audit_frame):,}")
        overview_metrics[1].metric("Total Columns", f"{audit_frame.shape[1]:,}")
        overview_metrics[2].metric("Total Missing Values", f"{int(audit_frame.isna().sum().sum()):,}")

        st.subheader(t("raw.preview.title"))
        st.dataframe(localize_dataframe_for_display(audit_frame.head(50)), use_container_width=True, height=360)

        st.subheader(t("raw.missing.title"))
        health_table = build_missing_table(audit_frame)
        st.dataframe(
            localize_dataframe_for_display(health_table),
            use_container_width=True,
            hide_index=True,
            height=620,
        )

        null_chart = px.bar(
            health_table.sort_values(["missing_values", "column"], ascending=[False, True]),
            x="missing_values",
            y="column",
            orientation="h",
            color="missing_pct",
            title="Null Values by Column",
            color_continuous_scale=["#f3dcc0", "#c95c36", "#7e3120"],
        )
        null_chart.update_layout(
            coloraxis_colorbar_title_text="Null %",
            margin=dict(l=10, r=10, t=50, b=10),
            yaxis=dict(categoryorder="total ascending"),
        )
        st.plotly_chart(null_chart, use_container_width=True)

        st.markdown("---")
        st.header("Raw Data Audit")

        missing_data = (
            health_table.rename(columns={"missing_pct": "missing_percent"})
            .loc[lambda frame: frame["missing_percent"] > 0]
            .sort_values(["missing_percent", "column"], ascending=[False, True])
        )

        fig1 = px.bar(
            missing_data,
            x="column",
            y="missing_percent",
            title="Phần Trăm Giá Trị Thiếu Theo Từng Cột (Missing Values)",
            labels={"column": "Column Name", "missing_percent": "Phần trăm (%)"},
            text=missing_data["missing_percent"].apply(lambda x: f"{x:.2f}%"),
            color_discrete_sequence=["#440154"]
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Ma Trận Giá Trị Thiếu (Missing Data)")
        sample_frame = audit_frame.sample(n=min(len(audit_frame), 1000), random_state=42).sort_index()
        null_matrix = sample_frame.isnull().astype(int)
        fig2 = px.imshow(
            null_matrix,
            labels=dict(x="Column Names", y="Số dòng dữ liệu", color="Missing"),
            color_continuous_scale=["#440154", "#fde725"]
        )
        fig2.update_traces(hovertemplate='Column: %{x}<br>Row: %{y}<br>Missing: %{z}')
        fig2.update_layout(
            title_text="Ma Trận Giá Trị Thiếu (Missing Data)",
        )
        st.plotly_chart(fig2, use_container_width=True)

        numerical_cols = [
            "price", "minimum_nights", "number_of_reviews", "service_fee",
            "reviews_per_month", "calculated_host_listings_count", "availability_365"
        ]
        available_numerical_cols = [col for col in numerical_cols if col in audit_frame.columns and pd.api.types.is_numeric_dtype(audit_frame[col])]

        if available_numerical_cols:
            melted_frame = audit_frame[available_numerical_cols].melt()
            fig3 = px.box(
                melted_frame,
                x="value",
                y="variable",
                title="Box Plot - Nhận Diện Outliers (Giá trị ngoại lai)",
                labels={"variable": "Column", "value": "Giá trị"},
                orientation="h",
                color="variable",
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No numerical columns found for outlier detection.")
    
    with tab_sort_data:
        st.subheader(t("raw.cleaned.title"))
        processed_frame = st.session_state.get("processed_df")
        if not isinstance(processed_frame, pd.DataFrame):
            st.info("Upload CSV để chạy preprocessing tự động. Phần pipeline chi tiết nằm ở tab Preprocessing bên sidebar.")
            return

        filtered = filter_dataframe(processed_frame)
        st.metric(t("raw.metric.filtered_rows"), f"{len(filtered):,}")
        st.dataframe(localize_dataframe_for_display(filtered), use_container_width=True, height=420)
