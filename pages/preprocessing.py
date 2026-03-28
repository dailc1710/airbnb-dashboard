from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from core.config import PREPROCESSING_PIPELINE_STEPS
from core.data import build_missing_table
from core.i18n import localize_dataframe_for_display
from preprocessing import run_preprocessing_pipeline as execute_preprocessing_pipeline

CLEANED_OUTPUT_PATH = Path("data/Airbnb_Data_cleaned.csv")
SCALED_OUTPUT_PATH = Path("data/Airbnb_Data_scaled.csv")
ML_OUTPUT_PATH = Path("data/Airbnb_Data_encoded.csv")
BOXPLOT_COLORS = [
    "#378ADD",
    "#1D9E75",
    "#D85A30",
    "#BA7517",
    "#7F77DD",
    "#888780",
    "#D4537E",
    "#639922",
    "#EF9F27",
    "#5DCAA5",
]


def save_processed_outputs(df_cleaned: pd.DataFrame, df_scaled: pd.DataFrame, df_ml_ready: pd.DataFrame) -> None:
    CLEANED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(CLEANED_OUTPUT_PATH, index=False)
    df_scaled.to_csv(SCALED_OUTPUT_PATH, index=False)
    df_ml_ready.to_csv(ML_OUTPUT_PATH, index=False)


def store_processed_outputs(
    before_frame: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    df_scaled: pd.DataFrame,
    df_ml_ready: pd.DataFrame,
    processing_report: dict[str, object],
    *,
    persist: bool = True,
) -> None:
    if persist:
        save_processed_outputs(df_cleaned, df_scaled, df_ml_ready)
    st.session_state["preprocessing_before_df"] = before_frame.copy()
    st.session_state["processing_report"] = processing_report
    st.session_state["processed_df"] = df_cleaned.copy()
    st.session_state["processed_scaled_df"] = df_scaled.copy()
    st.session_state["processed_ml_df"] = df_ml_ready.copy()
    st.session_state["cleaned_data"] = df_cleaned.copy()


@st.cache_data(show_spinner=False)
def run_processing_pipeline(
    raw_frame: pd.DataFrame,
    fallback_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    _ = fallback_frame
    return execute_preprocessing_pipeline(raw_frame)


def _render_download_outputs(df_cleaned: pd.DataFrame, df_ml_ready: pd.DataFrame) -> None:
    cleaned_csv = df_cleaned.to_csv(index=False).encode("utf-8")
    ml_csv = df_ml_ready.to_csv(index=False).encode("utf-8")

    st.subheader("Download preprocessing outputs")
    st.caption(
        "File clean là đầu ra chính sau toàn bộ bước preprocessing và feature engineering. "
        "Bản encoded giữ để đưa vào học máy."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download cleaned data",
            data=cleaned_csv,
            file_name="Airbnb_Data_cleaned.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary",
        )
    with col2:
        st.download_button(
            label="Download encoded file",
            data=ml_csv,
            file_name="Airbnb_Data_encoded.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.caption(
        "File scaled vẫn được tạo nội bộ cho visualization bằng MinMaxScaler trên các cột numeric đã chọn. "
        "Các feature mới như booking_demand, availability_efficiency và revenue_per_available_night được giữ raw."
    )


def render_processing_panel(raw_frame: pd.DataFrame) -> None:
    _ = raw_frame
    df_cleaned = st.session_state.get("processed_df")
    df_scaled = st.session_state.get("processed_scaled_df")
    df_ml_ready = st.session_state.get("processed_ml_df")
    before_frame = st.session_state.get("preprocessing_before_df")
    processing_report = st.session_state.get("processing_report") or {}

    if (
        not isinstance(df_cleaned, pd.DataFrame)
        or not isinstance(df_scaled, pd.DataFrame)
        or not isinstance(df_ml_ready, pd.DataFrame)
    ):
        st.info("Upload a CSV file in Input Data first. Preprocessing runs automatically, then the cleaned result appears in this tab.")
        return

    def _build_clean_schema_table(frame: pd.DataFrame) -> pd.DataFrame:
        return (
            pd.DataFrame(
                {
                    "column": frame.columns,
                    "dtype_after_clean": frame.dtypes.astype(str).tolist(),
                    "non_null_rows": frame.notna().sum().tolist(),
                    "missing_after_clean": frame.isna().sum().tolist(),
                }
            )
            .sort_values(["missing_after_clean", "column"], ascending=[False, True])
            .reset_index(drop=True)
        )

    def _build_missing_fill_audit_table(before_df: pd.DataFrame, after_df: pd.DataFrame) -> pd.DataFrame:
        before_missing = build_missing_table(before_df).rename(
            columns={"missing_values": "missing_before", "missing_pct": "missing_pct_before"}
        )
        after_missing = build_missing_table(after_df).rename(
            columns={"missing_values": "missing_after", "missing_pct": "missing_pct_after"}
        )
        merged = before_missing.merge(after_missing, on="column", how="outer").fillna(0)
        merged["filled_count"] = (merged["missing_before"] - merged["missing_after"]).clip(lower=0).astype(int)
        merged["resolution"] = merged.apply(
            lambda row: (
                "Dropped column"
                if row["column"] not in after_df.columns
                else "Rows dropped"
                if row["column"] == "last_review" and int(row["missing_before"]) > 0 and int(row["missing_after"]) == 0
                else "Filled/Resolved"
                if int(row["missing_after"]) == 0
                else "Needs review"
            ),
            axis=1,
        )
        merged = merged.loc[(merged["missing_before"] > 0) | (merged["missing_after"] > 0)].copy()
        merged[["missing_before", "missing_after"]] = merged[["missing_before", "missing_after"]].astype(int)
        return merged.sort_values(["missing_before", "column"], ascending=[False, True]).reset_index(drop=True)

    rows_after = int(processing_report.get("rows_after", len(df_cleaned)))
    columns_after = int(processing_report.get("columns_after", df_cleaned.shape[1]))
    duplicates_removed = int(processing_report.get("duplicates_removed", 0))
    rows_dropped_last_review = int(processing_report.get("rows_dropped_missing_last_review", 0))
    remaining_missing = int(df_cleaned.isna().sum().sum())

    st.success("Preprocessing completed successfully.")
    _render_download_outputs(df_cleaned, df_ml_ready)

    metric_cols = st.columns(5)
    metric_cols[0].metric("Rows after clean", f"{rows_after:,}")
    metric_cols[1].metric("Columns after clean", f"{columns_after:,}")
    metric_cols[2].metric("Duplicates removed", f"{duplicates_removed:,}")
    metric_cols[3].metric("Rows dropped by last_review", f"{rows_dropped_last_review:,}")
    metric_cols[4].metric("Missing after clean", f"{remaining_missing:,}")

    st.subheader("Dữ liệu sau preprocessing")
    st.caption("Bảng dưới đây là dữ liệu sau toàn bộ bước clean, fill missing, outlier handling, feature engineering và chưa scale.")
    st.dataframe(localize_dataframe_for_display(df_cleaned.head(100)), use_container_width=True, height=420)

    schema_col, missing_col = st.columns(2)
    with schema_col:
        st.subheader("Bảng 1. Schema sau khi clean")
        st.dataframe(_build_clean_schema_table(df_cleaned), use_container_width=True, hide_index=True, height=420)

    with missing_col:
        st.subheader("Bảng 2. Missing value trước và sau clean")
        if isinstance(before_frame, pd.DataFrame):
            st.dataframe(
                _build_missing_fill_audit_table(before_frame, df_cleaned),
                use_container_width=True,
                hide_index=True,
                height=420,
            )
        else:
            st.info("Chưa có dữ liệu trước xử lý để so sánh missing values.")

    scaled_columns = processing_report.get("scaled_columns", df_scaled.select_dtypes(include="number").columns.tolist())
    if not scaled_columns:
        return

    melted = df_scaled[scaled_columns].melt(var_name="column", value_name="value")
    fig = px.box(
        melted,
        x="value",
        y="column",
        orientation="h",
        title="Phân phối sau MinMaxScaler cho các cột số",
        color="column",
        color_discrete_sequence=BOXPLOT_COLORS,
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
        height=max(420, len(scaled_columns) * 50),
    )
    st.subheader("Phân phối dữ liệu sau chuẩn hóa")
    st.plotly_chart(fig, use_container_width=True)


def render_processing_steps_panel() -> None:
    st.subheader("Processing Steps")
    st.caption(
        "Các bước dưới đây bám theo nội dung preprocessing mới. "
        "Mỗi mục gồm mô tả nghiệp vụ và đoạn code đại diện cho phần logic đang chạy trong pipeline."
    )
    for step in PREPROCESSING_PIPELINE_STEPS:
        with st.expander(step["title"]):
            summary = step.get("summary")
            if summary:
                st.markdown(summary)
            st.code(step["code"], language="python")
