from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from core.config import PREPROCESSING_PIPELINE_STEPS
from core.data import build_missing_table
from core.i18n import get_language, localize_dataframe_for_display, t
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

PREP_TEXT = {
    "download_outputs_title": {"en": "Download preprocessing outputs", "vi": "Tải đầu ra tiền xử lý"},
    "download_outputs_caption": {
        "en": "The cleaned file is the primary output after preprocessing and feature engineering. The encoded file is kept for machine-learning workflows.",
        "vi": "Tệp đã làm sạch là đầu ra chính sau toàn bộ bước tiền xử lý và tạo đặc trưng. Bản đã mã hóa được giữ lại để đưa vào học máy.",
    },
    "download_cleaned": {"en": "Download cleaned dataset", "vi": "Tải dữ liệu đã làm sạch"},
    "download_encoded": {"en": "Download encoded dataset", "vi": "Tải tệp đã mã hóa"},
    "download_scaled_caption": {
        "en": "The scaled file is still created internally for visualization with MinMaxScaler on selected numeric columns. Engineered features such as booking_demand, availability_efficiency, and revenue_per_available_night remain unscaled.",
        "vi": "Tệp đã chuẩn hóa vẫn được tạo nội bộ cho trực quan hóa bằng MinMaxScaler trên các cột số đã chọn. Các feature mới như booking_demand, availability_efficiency và revenue_per_available_night được giữ nguyên.",
    },
    "upload_first_notice": {
        "en": "Upload a CSV in the Input Data page first. Preprocessing runs automatically and the cleaned result will appear in this tab.",
        "vi": "Hãy tải CSV ở trang Dữ liệu đầu vào trước. Tiền xử lý sẽ chạy tự động và kết quả đã làm sạch sẽ xuất hiện tại tab này.",
    },
    "schema_column": {"en": "Column", "vi": "Cột"},
    "schema_dtype": {"en": "Data type after processing", "vi": "Kiểu dữ liệu sau xử lý"},
    "schema_non_null": {"en": "Non-null rows", "vi": "Số dòng không rỗng"},
    "schema_missing_after": {"en": "Missing values after processing", "vi": "Giá trị thiếu sau xử lý"},
    "missing_before": {"en": "Missing before processing", "vi": "Thiếu trước xử lý"},
    "missing_before_pct": {"en": "Missing rate before processing", "vi": "Tỷ lệ thiếu trước xử lý"},
    "missing_after": {"en": "Missing after processing", "vi": "Thiếu sau xử lý"},
    "missing_after_pct": {"en": "Missing rate after processing", "vi": "Tỷ lệ thiếu sau xử lý"},
    "handled_value_count": {"en": "Handled values", "vi": "Số giá trị đã xử lý"},
    "status": {"en": "Status", "vi": "Trạng thái"},
    "status.column_removed": {"en": "Column removed", "vi": "Đã xóa cột"},
    "status.row_removed": {"en": "Rows removed", "vi": "Đã loại dòng"},
    "status.filled": {"en": "Filled/processed", "vi": "Đã điền/xử lý"},
    "status.review": {"en": "Needs review", "vi": "Cần xem xét"},
    "success": {"en": "Preprocessing completed.", "vi": "Tiền xử lý đã hoàn tất."},
    "metric.rows_after": {"en": "Rows after cleaning", "vi": "Dòng sau làm sạch"},
    "metric.columns_after": {"en": "Columns after cleaning", "vi": "Cột sau làm sạch"},
    "metric.duplicates_removed": {"en": "Duplicate records removed", "vi": "Bản ghi trùng đã xóa"},
    "metric.rows_removed_last_review": {"en": "Rows removed by `last_review`", "vi": "Dòng bị loại do `last_review`"},
    "metric.remaining_missing": {"en": "Missing values after cleaning", "vi": "Giá trị thiếu sau làm sạch"},
    "processed_data_title": {"en": "Processed dataset", "vi": "Dữ liệu sau tiền xử lý"},
    "processed_data_caption": {
        "en": "The table below shows the dataset after cleaning, missing-value handling, outlier handling, and feature engineering, before scaling.",
        "vi": "Bảng dưới đây là dữ liệu sau toàn bộ bước làm sạch, xử lý thiếu, xử lý ngoại lệ, tạo đặc trưng và chưa được scale.",
    },
    "schema_table_title": {"en": "Table 1. Cleaned schema", "vi": "Bảng 1. Schema sau làm sạch"},
    "missing_compare_title": {"en": "Table 2. Missing values before and after cleaning", "vi": "Bảng 2. Giá trị thiếu trước và sau làm sạch"},
    "missing_compare_unavailable": {"en": "No pre-processing snapshot is available to compare missing values.", "vi": "Chưa có dữ liệu trước xử lý để so sánh missing values."},
    "scaled_distribution_title": {"en": "Distribution after MinMax scaling", "vi": "Phân phối dữ liệu sau chuẩn hóa"},
    "scaled_distribution_chart": {"en": "Post-MinMaxScaler distribution for numeric columns", "vi": "Phân phối sau MinMaxScaler cho các cột số"},
    "steps_title": {"en": "Preprocessing steps", "vi": "Các bước tiền xử lý"},
    "steps_caption": {
        "en": "The steps below follow the current preprocessing specification. Each item includes a business summary and a representative code snippet from the active pipeline.",
        "vi": "Các bước dưới đây bám theo nội dung tiền xử lý mới. Mỗi mục gồm mô tả nghiệp vụ và đoạn code đại diện cho phần logic đang chạy trong pipeline.",
    },
}

PIPELINE_STEP_TRANSLATIONS = {
    "A.I. Làm sạch dữ liệu": {
        "title": {"en": "A.I. Data cleaning", "vi": "A.I. Làm sạch dữ liệu"},
        "summary": {
            "en": "- Normalize column names: lowercase, trim extra spaces, replace special characters with `_`.\n- Trim extra spaces in every string column.\n- Remove emoji, unusual symbols, HTML tags, and HTML entities.\n- Remove `$` and `,` from `price` and `service_fee`.",
            "vi": PREPROCESSING_PIPELINE_STEPS[0]["summary"],
        },
    },
    "A.II. Xử lý dữ liệu trùng lặp": {
        "title": {"en": "A.II. Duplicate handling", "vi": "A.II. Xử lý dữ liệu trùng lặp"},
        "summary": {
            "en": "- Remove duplicated rows based on `id`.\n- Keep the first record for each `id`.",
            "vi": PREPROCESSING_PIPELINE_STEPS[1]["summary"],
        },
    },
    "A.III. Chọn đặc trưng": {
        "title": {"en": "A.III. Feature selection", "vi": "A.III. Chọn đặc trưng"},
        "summary": {
            "en": "- Drop columns not used in the analysis: `id`, `name`, `host_name`, `lat`, `long`, `country`, `country_code`, `service_fee`, `reviews_per_month`, `house_rules`, `license`, `instant_bookable`, `cancellation_policy`.\n- The goal is to reduce data volume and the number of columns processed in the project.",
            "vi": PREPROCESSING_PIPELINE_STEPS[2]["summary"],
        },
    },
    "A.IV. Chuyển đổi kiểu dữ liệu": {
        "title": {"en": "A.IV. Data type conversion", "vi": "A.IV. Chuyển đổi kiểu dữ liệu"},
        "summary": {
            "en": "- Convert key numeric columns to `float64`.\n- Convert `construction_year` and `last_review` to `datetime`.\n- Convert string columns to lowercase.\n- Normalize common typos such as `brookln -> brooklyn` and `manhatan -> manhattan`.",
            "vi": PREPROCESSING_PIPELINE_STEPS[3]["summary"],
        },
    },
    "A.V.1. Xử lý giá trị thiếu - Dữ liệu phân loại": {
        "title": {"en": "A.V.1. Missing values - categorical data", "vi": "A.V.1. Xử lý giá trị thiếu - Dữ liệu phân loại"},
        "summary": {
            "en": "- Fill `host_identity_verified` with `unconfirmed`.\n- Map `neighbourhood_group` from `neighbourhood`.\n- Fill `neighbourhood` with the mode within `neighbourhood_group + room_type`.",
            "vi": PREPROCESSING_PIPELINE_STEPS[4]["summary"],
        },
    },
    "A.V.2. Xử lý giá trị thiếu - Ngày giờ": {
        "title": {"en": "A.V.2. Missing values - datetime", "vi": "A.V.2. Xử lý giá trị thiếu - Ngày giờ"},
        "summary": {
            "en": "- Fill `construction_year` with the mode within `neighbourhood + room_type`.\n- Treat `last_review` values later than `2022-12-31` as invalid and convert them to missing.\n- Drop rows with missing `last_review` because this field is sensitive and not suitable for interpolation.",
            "vi": PREPROCESSING_PIPELINE_STEPS[5]["summary"],
        },
    },
    "A.V.3. Xử lý giá trị thiếu - Dữ liệu số": {
        "title": {"en": "A.V.3. Missing values - numeric data", "vi": "A.V.3. Xử lý giá trị thiếu - Dữ liệu số"},
        "summary": {
            "en": "- Fill `price` with the mean within `neighbourhood + room_type`.\n- Make `minimum_nights` absolute, convert values `> 365` to missing, then fill with the group median.\n- Fill `number_of_reviews` with the median within `neighbourhood + room_type`.\n- Fill `review_rate_number` with the mode within `neighbourhood + room_type`.\n- Fill `calculated_host_listings_count` from host frequency, with fallback `1` when `host_id` is missing.\n- Make `availability_365` absolute, convert values `> 365` to missing, then fill with the group median.",
            "vi": PREPROCESSING_PIPELINE_STEPS[6]["summary"],
        },
    },
    "A.VI. Xử lý ngoại lệ": {
        "title": {"en": "A.VI. Outlier handling", "vi": "A.VI. Xử lý ngoại lệ"},
        "summary": {
            "en": "- Apply percentile capping at 1% and 99% for `price`.\n- Convert `minimum_nights` from `0 -> 1`, then apply IQR capping.\n- Apply IQR capping to `number_of_reviews` and `calculated_host_listings_count`.\n- Apply `clip(0, 5)` to `review_rate_number` and `clip(0, 365)` to `availability_365`.\n- Compute skewness beforehand to explain why each column uses a different strategy.\n- Round count/day-style columns after outlier handling for easier reading.",
            "vi": PREPROCESSING_PIPELINE_STEPS[7]["summary"],
        },
    },
    "B. Tải file preprocessing": {
        "title": {"en": "B. Export preprocessing files", "vi": "B. Tải file preprocessing"},
        "summary": {
            "en": "- Export `Airbnb_Data_cleaned.csv` after the full preprocessing pipeline.\n- The dashboard also keeps `scaled` output for visualization and `encoded` output for machine learning.",
            "vi": PREPROCESSING_PIPELINE_STEPS[8]["summary"],
        },
    },
    "C. Tạo đặc trưng": {
        "title": {"en": "C. Feature engineering", "vi": "C. Tạo đặc trưng"},
        "summary": {
            "en": "- `Booking Demand`: estimate demand from unavailable nights using `365 - availability_365`.\n- `Availability Category`: split listings into Low / Medium / High Availability groups.\n- `Availability Efficiency`: measure monetized utilization from price and booked nights.\n- `Revenue per Available Night`: normalize estimated revenue to an average per available night.",
            "vi": PREPROCESSING_PIPELINE_STEPS[9]["summary"],
        },
    },
    "D. Chuẩn hóa cho trực quan hóa": {
        "title": {"en": "D. Scaling for visualization", "vi": "D. Chuẩn hóa cho trực quan hóa"},
        "summary": {
            "en": "- Use `MinMaxScaler` for the exported `scaled` file used by the dashboard.\n- Scale only numeric columns that need direct visual comparison.\n- Keep engineered metrics such as `booking_demand`, `availability_efficiency`, and `revenue_per_available_night` unscaled.\n- Keep `availability_category` unscaled because it is ordinal and encoded separately in the ML-ready export.\n- Alternative scalers are kept only as modeling references.",
            "vi": PREPROCESSING_PIPELINE_STEPS[10]["summary"],
        },
    },
    "E. Trực quan hóa": {
        "title": {"en": "E. Visualization", "vi": "E. Trực quan hóa"},
        "summary": {
            "en": "- Pie chart for `availability_category` to read market supply state.\n- Boxplot of `booking_demand` by `neighbourhood_group` to compare demand across areas.\n- Scatter plot of `price` vs `booking_demand` with a trendline to check price sensitivity.\n- Heatmap of average `availability_efficiency` by `room_type` and `neighbourhood_group` to find business sweet spots.\n- Read the four visuals in the order Supply -> Demand -> Price -> Efficiency.",
            "vi": PREPROCESSING_PIPELINE_STEPS[11]["summary"],
        },
    },
    "F. Mã hóa cho học máy": {
        "title": {"en": "F. Encoding for machine learning", "vi": "F. Mã hóa cho học máy"},
        "summary": {
            "en": "- Keep `host_id` as an identifier rather than label-encoding it.\n- Encode `host_identity_verified` as binary: `unconfirmed -> 0`, `verified -> 1`.\n- Use label encoding for `neighbourhood_group` and `neighbourhood` because they have many levels.\n- Use one-hot encoding for `room_type` because it is nominal with few levels.\n- Keep core numeric variables as numeric.\n- Convert `construction_year` to numeric year and `last_review` to `days_since_last_review`.\n- Keep engineered features in the ML-ready file, while ordinal-encoding `availability_category` with `Low=0`, `Medium=1`, `High=2`.",
            "vi": PREPROCESSING_PIPELINE_STEPS[12]["summary"],
        },
    },
}


def _prep_text(key: str, **kwargs: object) -> str:
    lang = get_language()
    template = PREP_TEXT.get(key, {}).get(lang) or PREP_TEXT.get(key, {}).get("en") or key
    return template.format(**kwargs)


def _localize_pipeline_step(step: dict[str, str]) -> dict[str, str]:
    lang = get_language()
    translated = PIPELINE_STEP_TRANSLATIONS.get(step["title"], {})
    return {
        "title": translated.get("title", {}).get(lang) or step["title"],
        "summary": translated.get("summary", {}).get(lang) or step.get("summary", ""),
        "code": step["code"],
    }


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

    st.subheader(_prep_text("download_outputs_title"))
    st.caption(_prep_text("download_outputs_caption"))

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label=_prep_text("download_cleaned"),
            data=cleaned_csv,
            file_name="Airbnb_Data_cleaned.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary",
        )
    with col2:
        st.download_button(
            label=_prep_text("download_encoded"),
            data=ml_csv,
            file_name="Airbnb_Data_encoded.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.caption(_prep_text("download_scaled_caption"))


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
        st.info(_prep_text("upload_first_notice"))
        return

    def _build_clean_schema_table(frame: pd.DataFrame) -> pd.DataFrame:
        return (
            pd.DataFrame(
                {
                    _prep_text("schema_column"): frame.columns,
                    _prep_text("schema_dtype"): frame.dtypes.astype(str).tolist(),
                    _prep_text("schema_non_null"): frame.notna().sum().tolist(),
                    _prep_text("schema_missing_after"): frame.isna().sum().tolist(),
                }
            )
            .sort_values([_prep_text("schema_missing_after"), _prep_text("schema_column")], ascending=[False, True])
            .reset_index(drop=True)
        )

    def _build_missing_fill_audit_table(before_df: pd.DataFrame, after_df: pd.DataFrame) -> pd.DataFrame:
        before_missing = build_missing_table(before_df).rename(
            columns={"column": _prep_text("schema_column"), "missing_values": _prep_text("missing_before"), "missing_pct": _prep_text("missing_before_pct")}
        )
        after_missing = build_missing_table(after_df).rename(
            columns={"column": _prep_text("schema_column"), "missing_values": _prep_text("missing_after"), "missing_pct": _prep_text("missing_after_pct")}
        )
        merged = before_missing.merge(after_missing, on=_prep_text("schema_column"), how="outer").fillna(0)
        merged[_prep_text("handled_value_count")] = (
            merged[_prep_text("missing_before")] - merged[_prep_text("missing_after")]
        ).clip(lower=0).astype(int)
        merged[_prep_text("status")] = merged.apply(
            lambda row: (
                _prep_text("status.column_removed")
                if row[_prep_text("schema_column")] not in after_df.columns
                else _prep_text("status.row_removed")
                if row[_prep_text("schema_column")] == "last_review"
                and int(row[_prep_text("missing_before")]) > 0
                and int(row[_prep_text("missing_after")]) == 0
                else _prep_text("status.filled")
                if int(row[_prep_text("missing_after")]) == 0
                else _prep_text("status.review")
            ),
            axis=1,
        )
        merged = merged.loc[
            (merged[_prep_text("missing_before")] > 0) | (merged[_prep_text("missing_after")] > 0)
        ].copy()
        merged[[_prep_text("missing_before"), _prep_text("missing_after")]] = merged[
            [_prep_text("missing_before"), _prep_text("missing_after")]
        ].astype(int)
        return merged.sort_values([_prep_text("missing_before"), _prep_text("schema_column")], ascending=[False, True]).reset_index(drop=True)

    rows_after = int(processing_report.get("rows_after", len(df_cleaned)))
    columns_after = int(processing_report.get("columns_after", df_cleaned.shape[1]))
    duplicates_removed = int(processing_report.get("duplicates_removed", 0))
    rows_dropped_last_review = int(processing_report.get("rows_dropped_missing_last_review", 0))
    remaining_missing = int(df_cleaned.isna().sum().sum())

    st.success(_prep_text("success"))
    _render_download_outputs(df_cleaned, df_ml_ready)

    metric_cols = st.columns(5)
    metric_cols[0].metric(_prep_text("metric.rows_after"), f"{rows_after:,}")
    metric_cols[1].metric(_prep_text("metric.columns_after"), f"{columns_after:,}")
    metric_cols[2].metric(_prep_text("metric.duplicates_removed"), f"{duplicates_removed:,}")
    metric_cols[3].metric(_prep_text("metric.rows_removed_last_review"), f"{rows_dropped_last_review:,}")
    metric_cols[4].metric(_prep_text("metric.remaining_missing"), f"{remaining_missing:,}")

    st.subheader(_prep_text("processed_data_title"))
    st.caption(_prep_text("processed_data_caption"))
    st.dataframe(localize_dataframe_for_display(df_cleaned.head(100)), use_container_width=True, height=420)

    schema_col, missing_col = st.columns(2)
    with schema_col:
        st.subheader(_prep_text("schema_table_title"))
        st.dataframe(_build_clean_schema_table(df_cleaned), use_container_width=True, hide_index=True, height=420)

    with missing_col:
        st.subheader(_prep_text("missing_compare_title"))
        if isinstance(before_frame, pd.DataFrame):
            st.dataframe(
                _build_missing_fill_audit_table(before_frame, df_cleaned),
                use_container_width=True,
                hide_index=True,
                height=420,
            )
        else:
            st.info(_prep_text("missing_compare_unavailable"))

    scaled_columns = processing_report.get("scaled_columns", df_scaled.select_dtypes(include="number").columns.tolist())
    if not scaled_columns:
        return

    melted = df_scaled[scaled_columns].melt(var_name="column", value_name="value")
    fig = px.box(
        melted,
        x="value",
        y="column",
        orientation="h",
        title=_prep_text("scaled_distribution_chart"),
        color="column",
        color_discrete_sequence=BOXPLOT_COLORS,
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
        height=max(420, len(scaled_columns) * 50),
    )
    st.subheader(_prep_text("scaled_distribution_title"))
    st.plotly_chart(fig, use_container_width=True)


def render_processing_steps_panel() -> None:
    st.subheader(_prep_text("steps_title"))
    st.caption(_prep_text("steps_caption"))
    for step in PREPROCESSING_PIPELINE_STEPS:
        localized_step = _localize_pipeline_step(step)
        with st.expander(localized_step["title"]):
            summary = localized_step.get("summary")
            if summary:
                st.markdown(summary)
            st.code(localized_step["code"], language="python")
