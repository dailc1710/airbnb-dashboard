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

    st.subheader("Tải đầu ra tiền xử lý")
    st.caption(
        "Tệp đã làm sạch là đầu ra chính sau toàn bộ bước tiền xử lý và tạo đặc trưng. "
        "Bản đã mã hóa được giữ lại để đưa vào học máy."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Tải dữ liệu đã làm sạch",
            data=cleaned_csv,
            file_name="Airbnb_Data_cleaned.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary",
        )
    with col2:
        st.download_button(
            label="Tải tệp đã mã hóa",
            data=ml_csv,
            file_name="Airbnb_Data_encoded.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.caption(
        "Tệp đã chuẩn hóa vẫn được tạo nội bộ cho trực quan hóa bằng MinMaxScaler trên các cột số đã chọn. "
        "Các feature mới như booking_demand, availability_efficiency và revenue_per_available_night được giữ nguyên."
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
        st.info("Hãy tải CSV ở trang Dữ liệu đầu vào trước. Tiền xử lý sẽ chạy tự động và kết quả đã làm sạch sẽ xuất hiện tại tab này.")
        return

    def _build_clean_schema_table(frame: pd.DataFrame) -> pd.DataFrame:
        return (
            pd.DataFrame(
                {
                    "Cột": frame.columns,
                    "Kiểu dữ liệu sau xử lý": frame.dtypes.astype(str).tolist(),
                    "Số dòng không rỗng": frame.notna().sum().tolist(),
                    "Giá trị thiếu sau xử lý": frame.isna().sum().tolist(),
                }
            )
            .sort_values(["Giá trị thiếu sau xử lý", "Cột"], ascending=[False, True])
            .reset_index(drop=True)
        )

    def _build_missing_fill_audit_table(before_df: pd.DataFrame, after_df: pd.DataFrame) -> pd.DataFrame:
        before_missing = build_missing_table(before_df).rename(
            columns={"column": "Cột", "missing_values": "Thiếu trước xử lý", "missing_pct": "Tỷ lệ thiếu trước xử lý"}
        )
        after_missing = build_missing_table(after_df).rename(
            columns={"column": "Cột", "missing_values": "Thiếu sau xử lý", "missing_pct": "Tỷ lệ thiếu sau xử lý"}
        )
        merged = before_missing.merge(after_missing, on="Cột", how="outer").fillna(0)
        merged["Số giá trị đã xử lý"] = (merged["Thiếu trước xử lý"] - merged["Thiếu sau xử lý"]).clip(lower=0).astype(int)
        merged["Trạng thái"] = merged.apply(
            lambda row: (
                "Đã xóa cột"
                if row["Cột"] not in after_df.columns
                else "Đã loại dòng"
                if row["Cột"] == "last_review" and int(row["Thiếu trước xử lý"]) > 0 and int(row["Thiếu sau xử lý"]) == 0
                else "Đã điền/xử lý"
                if int(row["Thiếu sau xử lý"]) == 0
                else "Cần xem xét"
            ),
            axis=1,
        )
        merged = merged.loc[(merged["Thiếu trước xử lý"] > 0) | (merged["Thiếu sau xử lý"] > 0)].copy()
        merged[["Thiếu trước xử lý", "Thiếu sau xử lý"]] = merged[["Thiếu trước xử lý", "Thiếu sau xử lý"]].astype(int)
        return merged.sort_values(["Thiếu trước xử lý", "Cột"], ascending=[False, True]).reset_index(drop=True)

    rows_after = int(processing_report.get("rows_after", len(df_cleaned)))
    columns_after = int(processing_report.get("columns_after", df_cleaned.shape[1]))
    duplicates_removed = int(processing_report.get("duplicates_removed", 0))
    rows_dropped_last_review = int(processing_report.get("rows_dropped_missing_last_review", 0))
    remaining_missing = int(df_cleaned.isna().sum().sum())

    st.success("Tiền xử lý đã hoàn tất.")
    _render_download_outputs(df_cleaned, df_ml_ready)

    metric_cols = st.columns(5)
    metric_cols[0].metric("Dòng sau làm sạch", f"{rows_after:,}")
    metric_cols[1].metric("Cột sau làm sạch", f"{columns_after:,}")
    metric_cols[2].metric("Bản ghi trùng đã xóa", f"{duplicates_removed:,}")
    metric_cols[3].metric("Dòng bị loại do `last_review`", f"{rows_dropped_last_review:,}")
    metric_cols[4].metric("Giá trị thiếu sau làm sạch", f"{remaining_missing:,}")

    st.subheader("Dữ liệu sau tiền xử lý")
    st.caption("Bảng dưới đây là dữ liệu sau toàn bộ bước làm sạch, xử lý thiếu, xử lý ngoại lệ, tạo đặc trưng và chưa được scale.")
    st.dataframe(localize_dataframe_for_display(df_cleaned.head(100)), use_container_width=True, height=420)

    schema_col, missing_col = st.columns(2)
    with schema_col:
        st.subheader("Bảng 1. Schema sau làm sạch")
        st.dataframe(_build_clean_schema_table(df_cleaned), use_container_width=True, hide_index=True, height=420)

    with missing_col:
        st.subheader("Bảng 2. Giá trị thiếu trước và sau làm sạch")
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
    st.subheader("Các bước tiền xử lý")
    st.caption(
        "Các bước dưới đây bám theo nội dung tiền xử lý mới. "
        "Mỗi mục gồm mô tả nghiệp vụ và đoạn code đại diện cho phần logic đang chạy trong pipeline."
    )
    for step in PREPROCESSING_PIPELINE_STEPS:
        with st.expander(step["title"]):
            summary = step.get("summary")
            if summary:
                st.markdown(summary)
            st.code(step["code"], language="python")
