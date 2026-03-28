from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core.config import CHART_COLORS
from core.data import (
    _prepare_boxplot_comparison,
    _prepare_null_comparison,
    build_missing_table,
    coerce_currency,
    normalize_columns,
)
from core.i18n import localize_dataframe_for_display, t, translate_room_type
from pages.preprocessing import render_processing_panel, render_processing_steps_panel
from users import logout_user

AREA_SEQUENCE = ["Brooklyn", "Manhattan", "Queens", "Bronx", "Staten Island"]
ROOM_SEQUENCE = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
ROOM_TYPE_COLOR_MAP = {
    "Entire home/apt": "#1f3c5b",
    "Private room": "#c95c36",
    "Shared room": "#d8a65d",
    "Hotel room": "#6d8f71",
}
ROOM_TYPE_CANONICAL_MAP = {
    "entire home/apt": "Entire home/apt",
    "private room": "Private room",
    "shared room": "Shared room",
    "hotel room": "Hotel room",
}
AVAILABILITY_CATEGORY_ORDER = [
    "Low Availability",
    "Medium Availability",
    "High Availability",
]
AVAILABILITY_CATEGORY_COLOR_MAP = {
    "Low Availability": "#1f3c5b",
    "Medium Availability": "#c95c36",
    "High Availability": "#d8a65d",
}
AVAILABILITY_CATEGORY_LABEL_MAP = {
    "Low Availability": "Sẵn có thấp",
    "Medium Availability": "Sẵn có trung bình",
    "High Availability": "Sẵn có cao",
}
BOROUGH_DISPLAY_ORDER = ["brooklyn", "manhattan", "queens", "bronx", "staten island"]
BOROUGH_LABEL_MAP = {
    "brooklyn": "Brooklyn",
    "manhattan": "Manhattan",
    "queens": "Queens",
    "bronx": "Bronx",
    "staten island": "Staten Island",
}
BOROUGH_COLOR_MAP = {
    "brooklyn": "#1f3c5b",
    "manhattan": "#c95c36",
    "queens": "#d8a65d",
    "bronx": "#6d8f71",
    "staten island": "#7b8795",
}
NEIGHBOURHOOD_LOOKUP = {
    "Brooklyn": ["Williamsburg", "Bushwick", "Park Slope", "DUMBO"],
    "Manhattan": ["Midtown", "Chelsea", "Harlem", "SoHo"],
    "Queens": ["Astoria", "Flushing", "Long Island City", "Sunnyside"],
    "Bronx": ["Mott Haven", "Fordham", "Riverdale", "Belmont"],
    "Staten Island": ["St. George", "Great Kills", "Tottenville", "New Dorp"],
}
MISSING_VALUE_DISPLAY_ORDER = [
    "license",
    "house_rules",
    "last_review",
    "reviews_per_month",
    "country",
    "availability_365",
    "minimum_nights",
    "host_name",
    "review_rate_number",
    "calculated_host_listings_count",
    "host_identity_verified",
    "service_fee",
    "name",
    "price",
    "construction_year",
    "number_of_reviews",
    "country_code",
    "neighbourhood_group",
    "neighbourhood",
    "long",
    "lat",
]
OUTLIER_DISPLAY_ORDER = [
    "price",
    "minimum_nights",
    "number_of_reviews",
    "review_rate_number",
    "calculated_host_listings_count",
    "availability_365",
]


def _coerce_numeric(series: pd.Series, fill_value: float | None = None) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors="coerce")
    if fill_value is None:
        median_value = numeric_series.median(skipna=True)
        fill_value = 0.0 if pd.isna(median_value) else float(median_value)
    return numeric_series.fillna(fill_value)


def _prepare_processed_eda_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = normalize_columns(frame).copy().reset_index(drop=True)
    if prepared.empty:
        return prepared

    for column in ("neighbourhood_group", "neighbourhood", "room_type", "availability_category"):
        if column in prepared.columns:
            prepared[column] = prepared[column].astype("string").str.strip()

    for column in (
        "price",
        "minimum_nights",
        "number_of_reviews",
        "review_rate_number",
        "calculated_host_listings_count",
        "availability_365",
        "booking_demand",
        "availability_efficiency",
        "revenue_per_available_night",
    ):
        if column not in prepared.columns:
            continue
        if column == "price":
            prepared[column] = _coerce_numeric(coerce_currency(prepared[column]))
            continue
        prepared[column] = _coerce_numeric(prepared[column])

    if "availability_365" in prepared.columns:
        prepared["availability_365"] = prepared["availability_365"].clip(lower=0, upper=365)

    if "booking_demand" not in prepared.columns and "availability_365" in prepared.columns:
        prepared["booking_demand"] = (365 - prepared["availability_365"]).clip(lower=0).astype("float64")

    if "availability_category" not in prepared.columns and "availability_365" in prepared.columns:
        prepared["availability_category"] = pd.cut(
            prepared["availability_365"],
            bins=[-1, 150, 300, 365],
            labels=AVAILABILITY_CATEGORY_ORDER,
            include_lowest=True,
            right=True,
        ).astype("string")
    elif "availability_category" in prepared.columns:
        prepared["availability_category"] = (
            prepared["availability_category"]
            .astype("string")
            .str.strip()
            .replace(
                {
                    "low availability": "Low Availability",
                    "medium availability": "Medium Availability",
                    "high availability": "High Availability",
                }
            )
        )

    if "availability_efficiency" not in prepared.columns and {"price", "booking_demand"}.issubset(prepared.columns):
        prepared["availability_efficiency"] = (prepared["price"] * prepared["booking_demand"]).astype("float64")

    if (
        "revenue_per_available_night" not in prepared.columns
        and "availability_efficiency" in prepared.columns
    ):
        prepared["revenue_per_available_night"] = (
            prepared["availability_efficiency"] / 365.0
        ).astype("float64")

    return prepared

def _render_chart(title: str, fig: px.scatter, _conclusion: str = "") -> None:
    st.subheader(title)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _format_column_list(columns: list[str], empty_label: str = "Không có") -> str:
    if not columns:
        return empty_label
    return ", ".join(columns)


def _inject_audit_table_styles() -> None:
    st.markdown(
        """
        <style>
            .audit-card {
                background: linear-gradient(180deg, #fffaf3 0%, #f4eee4 100%);
                border: 1px solid rgba(201, 92, 54, 0.14);
                border-radius: 22px;
                padding: 1rem 1rem 0.85rem;
                margin: 0.35rem 0 0.9rem;
                box-shadow: 0 16px 34px rgba(125, 93, 62, 0.10);
                color: #2b3649;
            }
            .audit-card__header {
                display: flex;
                justify-content: space-between;
                align-items: flex-end;
                gap: 1rem;
                padding-bottom: 0.8rem;
                border-bottom: 1px solid rgba(201, 92, 54, 0.10);
                margin-bottom: 0.85rem;
            }
            .audit-card__title {
                color: #243041;
                font-size: 0.95rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            .audit-card__meta {
                color: #7b8795;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                white-space: nowrap;
            }
            .audit-table-wrap {
                overflow-x: auto;
            }
            .audit-table {
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                min-width: 760px;
            }
            .audit-table thead th {
                padding: 0.62rem 0.5rem;
                color: #7a8594;
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                text-align: left;
                border-bottom: 1px solid rgba(201, 92, 54, 0.10);
            }
            .audit-table tbody td {
                padding: 0.7rem 0.5rem;
                border-bottom: 1px solid rgba(39, 50, 70, 0.06);
                vertical-align: middle;
                color: #2f3a4c;
                font-size: 0.9rem;
            }
            .audit-table tbody tr:last-child td {
                border-bottom: none;
            }
            .audit-section-row td {
                padding: 0.82rem 0.5rem 0.58rem;
                background: rgba(201, 92, 54, 0.05);
                border-bottom: 1px solid rgba(201, 92, 54, 0.10);
            }
            .audit-section-row--divider td {
                border-top: 1px solid rgba(201, 92, 54, 0.18);
            }
            .audit-section-label {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                color: #2a3446;
                font-size: 0.8rem;
                font-weight: 800;
                letter-spacing: 0.06em;
                text-transform: uppercase;
            }
            .audit-section-label::before {
                content: "";
                width: 0.5rem;
                height: 0.5rem;
                border-radius: 999px;
                background: #9ba5b2;
            }
            .audit-section-label--categorical::before {
                background: #6f92ff;
            }
            .audit-section-label--numeric::before {
                background: #2db98a;
            }
            .audit-col-name {
                font-weight: 700;
                color: #243041;
            }
            .audit-pill,
            .audit-chip {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 0.24rem 0.56rem;
                border-radius: 999px;
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.02em;
                white-space: nowrap;
            }
            .audit-pill--categorical {
                background: rgba(94, 129, 244, 0.10);
                color: #4965d1;
                border: 1px solid rgba(94, 129, 244, 0.18);
            }
            .audit-pill--numeric {
                background: rgba(52, 211, 153, 0.10);
                color: #187e61;
                border: 1px solid rgba(52, 211, 153, 0.18);
            }
            .audit-pill--geoid {
                background: rgba(148, 163, 184, 0.12);
                color: #5f6d7c;
                border: 1px solid rgba(148, 163, 184, 0.18);
            }
            .audit-progress-cell {
                display: flex;
                align-items: center;
                gap: 0.55rem;
                min-width: 12rem;
            }
            .audit-progress {
                flex: 1 1 auto;
                height: 0.38rem;
                border-radius: 999px;
                background: rgba(39, 50, 70, 0.10);
                overflow: hidden;
            }
            .audit-progress > span {
                display: block;
                height: 100%;
                border-radius: inherit;
                min-width: 0.18rem;
            }
            .audit-progress--success > span {
                background: linear-gradient(90deg, #22c55e 0%, #86efac 100%);
            }
            .audit-progress--warning > span {
                background: linear-gradient(90deg, #f59e0b 0%, #fcd34d 100%);
            }
            .audit-progress--danger > span {
                background: linear-gradient(90deg, #f97316 0%, #ef4444 100%);
            }
            .audit-progress-cell strong {
                color: #5a6776;
                font-size: 0.76rem;
                min-width: 3.1rem;
                text-align: right;
            }
            .audit-chip--neutral {
                background: rgba(148, 163, 184, 0.12);
                color: #607081;
                border: 1px solid rgba(148, 163, 184, 0.16);
            }
            .audit-chip--success {
                background: rgba(52, 211, 153, 0.10);
                color: #187e61;
                border: 1px solid rgba(52, 211, 153, 0.18);
            }
            .audit-chip--warning {
                background: rgba(245, 158, 11, 0.12);
                color: #9c6408;
                border: 1px solid rgba(245, 158, 11, 0.18);
            }
            .audit-chip--danger {
                background: rgba(248, 113, 113, 0.11);
                color: #b74e4e;
                border: 1px solid rgba(248, 113, 113, 0.18);
            }
            .audit-chip--info {
                background: rgba(96, 165, 250, 0.10);
                color: #426da9;
                border: 1px solid rgba(96, 165, 250, 0.18);
            }
            .audit-card__footer {
                margin-top: 0.75rem;
                color: #7b8795;
                font-size: 0.78rem;
                line-height: 1.5;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_metric_number(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "--"
    numeric_value = float(value)
    if abs(numeric_value) >= 1000:
        return f"{numeric_value:,.0f}" if numeric_value.is_integer() else f"{numeric_value:,.2f}"
    if numeric_value.is_integer():
        return f"{numeric_value:.0f}"
    if abs(numeric_value) >= 100:
        return f"{numeric_value:.1f}"
    if abs(numeric_value) >= 10:
        return f"{numeric_value:.2f}"
    return f"{numeric_value:.3f}"


def _progress_tone(percent: float) -> str:
    if percent >= 20:
        return "danger"
    if percent >= 5:
        return "warning"
    return "success"


def _chip_tone(label: str) -> str:
    lowered = label.lower()
    if "drop" in lowered or "xóa" in lowered or "loại bỏ" in lowered:
        return "danger"
    if "median" in lowered or "mode" in lowered or "trung vị" in lowered:
        return "success"
    if (
        "condition" in lowered
        or "nat" in lowered
        or "geo" in lowered
        or "derive" in lowered
        or "điều kiện" in lowered
        or "ánh xạ" in lowered
        or "sinh" in lowered
    ):
        return "info"
    if (
        "fill" in lowered
        or "false" in lowered
        or "other" in lowered
        or "unknown" in lowered
        or "unconfirmed" in lowered
        or "điền" in lowered
        or "không xác định" in lowered
        or "chưa xác minh" in lowered
    ):
        return "warning"
    return "neutral"


def _skew_chip_tone(skewness: float | None) -> str:
    if skewness is None or pd.isna(skewness):
        return "neutral"
    abs_skew = abs(float(skewness))
    if abs_skew >= 1.5:
        return "danger"
    if abs_skew >= 0.5:
        return "warning"
    return "success"


def _missing_type_label(column: str, series: pd.Series) -> tuple[str, str]:
    if column in {"lat", "long"}:
        return "Tọa độ", "geoid"
    if column in set(OUTLIER_DISPLAY_ORDER):
        return "Số", "numeric"
    if pd.api.types.is_numeric_dtype(series):
        return "Số", "numeric"
    return "Phân loại", "categorical"


def _prepare_profile_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype="float64")
    series = frame[column]
    if column in {"price", "service_fee"}:
        return coerce_currency(series)
    if column == "construction_year":
        if pd.api.types.is_datetime64_any_dtype(series):
            return series.dt.year.astype("float64")
        return pd.to_numeric(series.astype("string").str.extract(r"(\d{4})", expand=False), errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def _safe_skew(series: pd.Series) -> float | None:
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric_series) < 3 or numeric_series.nunique() <= 1:
        return None
    skewness = float(numeric_series.skew())
    if pd.isna(skewness):
        return None
    return skewness


def _choose_outlier_method(column: str, skewness: float | None) -> str:
    if column == "price":
        return "Percentile Capping (1%, 99%)"
    if column in {"minimum_nights", "number_of_reviews", "calculated_host_listings_count"}:
        return "IQR Capping"
    if column == "review_rate_number":
        return "Clip [0, 5]"
    if column == "availability_365":
        return "Clip [0, 365]"
    if skewness is None:
        return "Review"
    return "IQR Capping" if abs(float(skewness)) >= 0.5 else "Percentile Review"


def _display_outlier_method(method: str) -> str:
    mapping = {
        "Percentile Capping (1%, 99%)": "Chặn theo percentile (1%, 99%)",
        "IQR Capping": "Chặn theo IQR",
        "Clip [0, 5]": "Giới hạn [0, 5]",
        "Clip [0, 365]": "Giới hạn [0, 365]",
        "Review": "Xem xét",
        "Percentile Review": "Xem xét theo percentile",
    }
    return mapping.get(method, method)


def _count_outliers(series: pd.Series, method: str) -> int:
    numeric_series = pd.to_numeric(series, errors="coerce").dropna().astype("float64")
    if numeric_series.empty or numeric_series.nunique() <= 1:
        return 0
    if method == "Percentile Capping (1%, 99%)":
        lower = float(numeric_series.quantile(0.01))
        upper = float(numeric_series.quantile(0.99))
        return int(((numeric_series < lower) | (numeric_series > upper)).sum())
    if method == "IQR Capping":
        q1 = float(numeric_series.quantile(0.25))
        q3 = float(numeric_series.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            return 0
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return int(((numeric_series < lower) | (numeric_series > upper)).sum())
    if method == "Clip [0, 5]":
        return int(((numeric_series < 0) | (numeric_series > 5)).sum())
    if method == "Clip [0, 365]":
        return int(((numeric_series < 0) | (numeric_series > 365)).sum())
    return 0


def _build_missing_strategy_map(
    missing_value_handling: dict[str, object],
    dropped_columns: list[str],
) -> dict[str, str]:
    concise_labels = {
        "host_identity_verified": 'Điền "unconfirmed"',
        "neighbourhood": "Mode theo neighbourhood_group + room_type",
        "neighbourhood_group": "Ánh xạ theo địa lý",
        "construction_year": "Mode theo neighbourhood + room_type",
        "price": "Mean theo neighbourhood + room_type",
        "service_fee": "Xóa cột",
        "minimum_nights": "Lấy trị tuyệt đối, >365 -> trung vị theo nhóm",
        "number_of_reviews": "Trung vị theo neighbourhood + room_type",
        "last_review": "Loại dòng bị thiếu",
        "reviews_per_month": "Xóa cột",
        "review_rate_number": "Mode theo neighbourhood + room_type",
        "calculated_host_listings_count": "Tần suất host",
        "availability_365": "Lấy trị tuyệt đối, >365 -> trung vị theo nhóm",
        "host_id": "Sinh giá trị dự phòng duy nhất sau khi điền theo tần suất host",
        "host_name": "Xóa cột",
        "name": "Xóa cột",
        "country": "Xóa cột",
        "country_code": "Xóa cột",
        "house_rules": "Xóa cột",
        "license": "Xóa cột",
        "lat": "Xóa cột",
        "long": "Xóa cột",
        "id": "Loại trùng rồi xóa",
    }
    for column in dropped_columns:
        concise_labels.setdefault(column, "Xóa cột")
    return concise_labels


def _build_missing_value_rows(
    before_frame: pd.DataFrame,
    missing_value_handling: dict[str, object],
    dropped_columns: list[str],
) -> list[dict[str, object]]:
    strategy_map = _build_missing_strategy_map(missing_value_handling, dropped_columns)
    missing_table = build_missing_table(before_frame)
    missing_lookup = missing_table.set_index("column")
    rows: list[dict[str, object]] = []

    available_columns = [column for column in MISSING_VALUE_DISPLAY_ORDER if column in before_frame.columns]
    remaining_columns = [
        column
        for column in missing_lookup.index.tolist()
        if column not in MISSING_VALUE_DISPLAY_ORDER and int(missing_lookup.loc[column, "missing_values"]) > 0
    ]

    for column in available_columns + remaining_columns:
        missing_count = int(missing_lookup.loc[column, "missing_values"]) if column in missing_lookup.index else 0
        if missing_count <= 0:
            continue
        missing_pct = float(missing_lookup.loc[column, "missing_pct"])
        series = before_frame[column]
        dtype_label, dtype_tone = _missing_type_label(column, series)
        profile_series = _prepare_profile_series(before_frame, column)
        skewness = _safe_skew(profile_series) if dtype_tone == "numeric" else None
        rows.append(
            {
                "column": column,
                "type_label": dtype_label,
                "type_tone": dtype_tone,
                "missing_count": missing_count,
                "missing_pct": missing_pct,
                "skewness": skewness,
                "strategy": strategy_map.get(column, "Cần xem xét thủ công"),
            }
        )
    return rows


def _build_missing_strategy_table(
    before_frame: pd.DataFrame,
    after_frame: pd.DataFrame,
    missing_value_handling: dict[str, object],
    dropped_columns: list[str],
    section: str,
) -> pd.DataFrame:
    rows = _build_missing_value_rows(before_frame, missing_value_handling, dropped_columns)
    after_lookup = build_missing_table(after_frame).set_index("column") if not after_frame.empty else pd.DataFrame()
    selected_rows = [row for row in rows if row["type_tone"] == "numeric"] if section == "numeric" else [
        row for row in rows if row["type_tone"] != "numeric"
    ]
    table_rows: list[dict[str, object]] = []
    for row in selected_rows:
        column = str(row["column"])
        after_missing = int(after_lookup.loc[column, "missing_values"]) if column in after_lookup.index else 0
        status = (
            "Đã xóa cột"
            if column in dropped_columns or column not in after_frame.columns
            else "Đã loại dòng"
            if column == "last_review" and int(row["missing_count"]) > 0 and after_missing == 0
            else "Đã điền/xử lý"
            if after_missing == 0
            else "Cần xem xét"
        )
        row_data: dict[str, object] = {
            "Cột": column,
            "Thiếu trước xử lý": int(row["missing_count"]),
            "Thiếu sau xử lý": after_missing,
            "Cách xử lý": str(row["strategy"]),
            "Trạng thái": status,
        }
        if section == "numeric":
            row_data["Độ lệch"] = None if row["skewness"] is None else round(float(row["skewness"]), 3)
        table_rows.append(row_data)

    ordered_columns = ["Cột", "Thiếu trước xử lý", "Thiếu sau xử lý", "Cách xử lý", "Trạng thái"]
    if section == "numeric":
        ordered_columns = ["Cột", "Thiếu trước xử lý", "Thiếu sau xử lý", "Độ lệch", "Cách xử lý", "Trạng thái"]
    if not table_rows:
        return pd.DataFrame(columns=ordered_columns)
    return pd.DataFrame(table_rows)[ordered_columns]


def _build_outlier_rows(before_frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column in OUTLIER_DISPLAY_ORDER:
        if column not in before_frame.columns:
            continue
        numeric_series = _prepare_profile_series(before_frame, column).dropna().astype("float64")
        if numeric_series.empty:
            continue
        skewness = _safe_skew(numeric_series)
        method = _choose_outlier_method(column, skewness)
        outlier_count = _count_outliers(numeric_series, method)
        rows.append(
            {
                "column": column,
                "min_value": float(numeric_series.min()),
                "max_value": float(numeric_series.max()),
                "skewness": skewness,
                "method": method,
                "outlier_count": outlier_count,
                "outlier_pct": (outlier_count / len(numeric_series) * 100.0) if len(numeric_series) else 0.0,
            }
        )
    return rows


def _build_outlier_strategy_table(
    before_frame: pd.DataFrame,
    outlier_handling: dict[str, object],
) -> pd.DataFrame:
    table_rows: list[dict[str, object]] = []
    adjusted_value_counts = outlier_handling.get("adjusted_value_counts", {})
    applied_methods = outlier_handling.get("applied_methods", {})
    for column in OUTLIER_DISPLAY_ORDER:
        if column not in before_frame.columns:
            continue
        numeric_series = _prepare_profile_series(before_frame, column).dropna().astype("float64")
        if numeric_series.empty:
            continue
        method = applied_methods.get(column, _choose_outlier_method(column, _safe_skew(numeric_series)))
        if column == "price":
            lower = float(numeric_series.quantile(0.01))
            upper = float(numeric_series.quantile(0.99))
            threshold_text = f"< {round(lower, 3)} or > {round(upper, 3)}"
        elif column in {"minimum_nights", "number_of_reviews", "calculated_host_listings_count"}:
            q1 = float(numeric_series.quantile(0.25))
            q3 = float(numeric_series.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr if iqr != 0 else q1
            upper = q3 + 1.5 * iqr if iqr != 0 else q3
            threshold_text = f"< {round(lower, 3)} or > {round(upper, 3)}"
        elif column == "review_rate_number":
            lower, upper = 0.0, 5.0
            threshold_text = "< 0 or > 5"
        else:
            lower, upper = 0.0, 365.0
            threshold_text = "< 0 or > 365"
        table_rows.append(
            {
                "Cột": column,
                "Ngưỡng dưới": round(lower, 3),
                "Ngưỡng trên": round(upper, 3),
                "Điều kiện ngoại lệ": threshold_text,
                "Số giá trị đã điều chỉnh": int(adjusted_value_counts.get(column, 0)),
                "Cách xử lý áp dụng": _display_outlier_method(str(method)),
            }
        )
    return pd.DataFrame(table_rows)


def _render_progress_cell(percent: float) -> str:
    tone = _progress_tone(percent)
    width = min(max(percent, 0.0), 100.0)
    return (
        f'<div class="audit-progress-cell">'
        f'<div class="audit-progress audit-progress--{tone}"><span style="width:{width:.2f}%"></span></div>'
        f"<strong>{percent:.1f}%</strong>"
        f"</div>"
    )


def _render_missing_values_card(
    before_frame: pd.DataFrame,
    missing_value_handling: dict[str, object],
    dropped_columns: list[str],
    section: str,
) -> None:
    rows = _build_missing_value_rows(before_frame, missing_value_handling, dropped_columns)
    section_rows = [row for row in rows if row["type_tone"] == "numeric"] if section == "numeric" else [
        row for row in rows if row["type_tone"] != "numeric"
    ]
    if not section_rows:
        return

    def _render_row(row: dict[str, object]) -> str:
        skew_text = "--" if row["skewness"] is None else f"{row['skewness']:.3f}"
        skew_tone = _skew_chip_tone(row["skewness"])
        strategy_tone = _chip_tone(str(row["strategy"]))
        return (
            "<tr>"
            f'<td class="audit-col-name">{escape(str(row["column"]))}</td>'
            f'<td><span class="audit-pill audit-pill--{escape(str(row["type_tone"]))}">{escape(str(row["type_label"]))}</span></td>'
            f'<td>{row["missing_count"]:,}</td>'
            f"<td>{_render_progress_cell(float(row['missing_pct']))}</td>"
            f'<td><span class="audit-chip audit-chip--{skew_tone}">{escape(skew_text)}</span></td>'
            f'<td><span class="audit-chip audit-chip--{strategy_tone}">{escape(str(row["strategy"]))}</span></td>'
            "</tr>"
        )

    title = "Bảng 2 - Giá trị thiếu (dữ liệu số)" if section == "numeric" else "Bảng 1 - Giá trị thiếu (dữ liệu phân loại)"
    meta = (
        f"Chỉ gồm cột số - {len(section_rows):,} cột"
        if section == "numeric"
        else f"Chỉ gồm cột phân loại - {len(section_rows):,} cột"
    )
    body_rows = [_render_row(row) for row in section_rows]

    st.markdown(
        f"""
        <div class="audit-card">
            <div class="audit-card__header">
                <div class="audit-card__title">{title}</div>
                <div class="audit-card__meta">{meta}</div>
            </div>
            <div class="audit-table-wrap">
                <table class="audit-table">
                    <thead>
                        <tr>
                            <th>Cột</th>
                            <th>Loại dữ liệu</th>
                            <th>Số giá trị thiếu</th>
                            <th>Tỷ lệ thiếu</th>
                            <th>Độ lệch</th>
                            <th>Cách xử lý</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(body_rows)}
                    </tbody>
                </table>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_outlier_card(before_frame: pd.DataFrame) -> None:
    rows = _build_outlier_rows(before_frame)
    if not rows:
        st.info("Không có cột số phù hợp để kiểm tra ngoại lệ.")
        return

    body_rows = []
    for row in rows:
        skew_text = "--" if row["skewness"] is None else f"{row['skewness']:.3f}"
        skew_tone = _skew_chip_tone(row["skewness"])
        method_tone = _chip_tone(str(row["method"]))
        body_rows.append(
            "<tr>"
            f'<td class="audit-col-name">{escape(str(row["column"]))}</td>'
            f"<td>{escape(_format_metric_number(row['min_value']))}</td>"
            f"<td>{escape(_format_metric_number(row['max_value']))}</td>"
            f'<td><span class="audit-chip audit-chip--{skew_tone}">{escape(skew_text)}</span></td>'
            f'<td><span class="audit-chip audit-chip--{method_tone}">{escape(_display_outlier_method(str(row["method"])))}</span></td>'
            f'<td>{int(row["outlier_count"]):,}</td>'
            f"<td>{_render_progress_cell(float(row['outlier_pct']))}</td>"
            "</tr>"
        )

    st.markdown(
        f"""
        <div class="audit-card">
            <div class="audit-card__header">
                <div class="audit-card__title">Bảng 3 - Phát hiện ngoại lệ</div>
                <div class="audit-card__meta">Chỉ gồm cột số - {len(rows):,} cột</div>
            </div>
            <div class="audit-table-wrap">
                <table class="audit-table">
                    <thead>
                        <tr>
                            <th>Cột</th>
                            <th>Nhỏ nhất</th>
                            <th>Lớn nhất</th>
                            <th>Độ lệch</th>
                            <th>Cách xử lý tự động</th>
                            <th>Số ngoại lệ</th>
                            <th>Tỷ lệ ngoại lệ</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(body_rows)}
                    </tbody>
                </table>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_pipeline_summary(
    processing_report: dict[str, object],
    before_frame: pd.DataFrame,
) -> None:
    after_frame = st.session_state.get("processed_df")
    step_metrics = processing_report.get("step_metrics", {})
    rows_before = int(processing_report.get("rows_before", 0))
    rows_after = int(processing_report.get("rows_after", 0))
    columns_after = int(processing_report.get("columns_after", 0))
    duplicates_removed = int(processing_report.get("duplicates_removed", 0))
    rows_dropped_last_review = int(processing_report.get("rows_dropped_missing_last_review", 0))
    dropped_columns = processing_report.get("dropped_columns", [])
    scaled_columns = processing_report.get("scaled_columns", [])
    remaining_issues = processing_report.get("remaining_issues", [])
    _inject_audit_table_styles()

    st.subheader("Tóm tắt lần chạy tiền xử lý")
    st.caption("Tab này cho biết lần chạy tiền xử lý gần nhất đã thực hiện những gì trên bộ dữ liệu được tải lên.")

    metric_cols = st.columns(5)
    metric_cols[0].metric("Dòng trước xử lý", f"{rows_before:,}")
    metric_cols[1].metric("Dòng sau xử lý", f"{rows_after:,}")
    metric_cols[2].metric("Cột sau xử lý", f"{columns_after:,}")
    metric_cols[3].metric("Bản ghi trùng đã xóa", f"{duplicates_removed:,}")
    metric_cols[4].metric("Dòng bị loại do `last_review`", f"{rows_dropped_last_review:,}")

    data_cleaning = step_metrics.get("data_cleaning", {})
    with st.expander("1. Làm sạch dữ liệu", expanded=True):
        st.write("Trạng thái: Hoàn tất")
        st.write(
            "Chuẩn hóa tên cột: "
            + str(data_cleaning.get("column_name_normalization", "lowercase + trim + ký tự đặc biệt -> _"))
        )
        st.write(f"Cột chuỗi đã trim: {_format_column_list(data_cleaning.get('string_columns_stripped', []))}")
        st.write(f"Cột văn bản đã làm sạch: {_format_column_list(data_cleaning.get('text_cleaned_columns', []))}")
        st.write(f"Cột tiền tệ đã chuẩn hóa: {_format_column_list(data_cleaning.get('currency_cleaned_columns', []))}")

    duplicate_handling = step_metrics.get("duplicate_handling", {})
    with st.expander("2. Xử lý dữ liệu trùng lặp", expanded=True):
        st.write("Trạng thái: Hoàn tất")
        st.write(f"Cột mục tiêu: {duplicate_handling.get('target_column') or 'Không tìm thấy cột id'}")
        st.write(f"Số bản ghi trùng đã xóa: {int(duplicate_handling.get('duplicates_removed', 0)):,}")
        st.write(f"Số dòng sau khi loại trùng: {int(duplicate_handling.get('rows_after_dedup', rows_after)):,}")

    feature_selection = step_metrics.get("feature_selection", {})
    with st.expander("3. Chọn đặc trưng", expanded=True):
        st.write("Trạng thái: Hoàn tất")
        st.write(f"Cột đã loại bỏ ({len(dropped_columns)}): {_format_column_list(feature_selection.get('dropped_columns', []))}")
        st.write(f"Số cột còn lại: {len(feature_selection.get('remaining_columns', []))}")

    type_conversion = step_metrics.get("data_type_conversion", {})
    with st.expander("4. Chuyển đổi kiểu dữ liệu", expanded=True):
        st.write("Trạng thái: Hoàn tất")
        st.write(f"Cột `float64`: {_format_column_list(type_conversion.get('float64_columns', []))}")
        st.write(f"Cột datetime: {_format_column_list(type_conversion.get('datetime_columns', []))}")
        st.write(f"Cột chuỗi đã chuyển lowercase: {_format_column_list(type_conversion.get('lowercase_string_columns', []))}")

    missing_value_handling = step_metrics.get("missing_value_handling", {})
    with st.expander("5. Xử lý giá trị thiếu", expanded=True):
        st.write("Trạng thái: Hoàn tất")
        st.write(f"Listing thiếu `host_id` trước khi điền theo tần suất: {int(missing_value_handling.get('host_id_missing_count', 0)):,}")
        st.write(f"Số dòng bị loại vì thiếu `last_review`: {int(missing_value_handling.get('rows_dropped_missing_last_review', 0)):,}")
        st.write(
            "Số giá trị null còn lại sau tiền xử lý: "
            f"{int(missing_value_handling.get('remaining_null_count', 0)):,}"
        )
        st.write("Quy tắc cho dữ liệu phân loại:")
        for rule in missing_value_handling.get("categorical_fill_rules", []):
            st.write(f"- {rule}")
        st.write("Quy tắc cho dữ liệu ngày giờ:")
        for rule in missing_value_handling.get("datetime_fill_rules", []):
            st.write(f"- {rule}")
        if isinstance(after_frame, pd.DataFrame):
            categorical_strategy_table = _build_missing_strategy_table(
                before_frame,
                after_frame,
                missing_value_handling,
                dropped_columns,
                section="categorical",
            )
            if not categorical_strategy_table.empty:
                st.dataframe(categorical_strategy_table, use_container_width=True, hide_index=True)
        _render_missing_values_card(before_frame, missing_value_handling, dropped_columns, section="categorical")
        st.write("Quy tắc cho dữ liệu số:")
        for rule in missing_value_handling.get("numeric_fill_rules", []):
            st.write(f"- {rule}")
        invalid_counts = missing_value_handling.get("invalid_value_counts", {})
        if invalid_counts:
            st.write(
                "Giá trị không hợp lệ đã được chuyển thành missing trước khi điền: "
                + ", ".join(f"{column}={int(count):,}" for column, count in invalid_counts.items())
            )
        if isinstance(after_frame, pd.DataFrame):
            numeric_strategy_table = _build_missing_strategy_table(
                before_frame,
                after_frame,
                missing_value_handling,
                dropped_columns,
                section="numeric",
            )
            if not numeric_strategy_table.empty:
                st.dataframe(numeric_strategy_table, use_container_width=True, hide_index=True)
        _render_missing_values_card(before_frame, missing_value_handling, dropped_columns, section="numeric")
        st.caption("Các bảng thẻ bên dưới làm rõ mẫu thiếu dữ liệu ban đầu và quyết định điền, ánh xạ hoặc loại dòng mà pipeline đã áp dụng.")

    outlier_handling = step_metrics.get("outlier_handling", {})
    with st.expander("6. Xử lý ngoại lệ", expanded=True):
        st.write("Trạng thái: Hoàn tất")
        st.write(f"Cột đã chặn ngưỡng: {_format_column_list(outlier_handling.get('clipped_columns', []))}")
        st.write(f"Cột đã làm tròn: {_format_column_list(outlier_handling.get('rounded_columns', []))}")
        applied_methods = outlier_handling.get("applied_methods", {})
        if applied_methods:
            st.write(
                "Phương pháp đã áp dụng: "
                + ", ".join(f"{column} -> {_display_outlier_method(str(method))}" for column, method in applied_methods.items())
            )
        outlier_strategy_table = _build_outlier_strategy_table(before_frame, outlier_handling)
        if not outlier_strategy_table.empty:
            st.dataframe(outlier_strategy_table, use_container_width=True, hide_index=True)
        adjusted_value_counts = outlier_handling.get("adjusted_value_counts", {})
        adjusted_table = pd.DataFrame(
            [
                {"column": column, "adjusted_value_count": int(count)}
                for column, count in adjusted_value_counts.items()
            ]
        )
        if not adjusted_table.empty:
            st.dataframe(adjusted_table, use_container_width=True, hide_index=True)
        if adjusted_value_counts:
            st.caption(
                "Số giá trị đã điều chỉnh sau khi phát hiện ngoại lệ: "
                + ", ".join(f"{column}={int(count):,}" for column, count in adjusted_value_counts.items())
            )
        _render_outlier_card(before_frame)

    integrity_check = step_metrics.get("integrity_check", {})
    validation_status = "Đạt" if integrity_check.get("passed", False) else "Cần lưu ý"
    st.caption(
        "Kiểm tra ràng buộc: "
        f"{validation_status}. "
        f"availability_365 [0, 365]={integrity_check.get('availability_365_in_range', True)}, "
        f"minimum_nights [1, 365]={integrity_check.get('minimum_nights_in_range', True)}, "
        f"review_rate_number [0, 5]={integrity_check.get('review_rate_number_in_range', True)}."
    )
    if remaining_issues:
        st.warning("Vấn đề còn lại: " + "; ".join(str(issue) for issue in remaining_issues))

    download_export = step_metrics.get("download_export", {})
    with st.expander("7. Tải file tiền xử lý", expanded=True):
        st.write("Trạng thái: Hoàn tất")
        cleaned_shape = download_export.get("cleaned_shape", [rows_after, columns_after])
        scaled_shape = download_export.get("scaled_shape", [])
        encoded_shape = download_export.get("encoded_shape", [])
        st.write(f"Tệp đã làm sạch: {download_export.get('cleaned_file', 'data/Airbnb_Data_cleaned.csv')}")
        if isinstance(cleaned_shape, list) and len(cleaned_shape) == 2:
            st.write(f"Kích thước dataframe đã làm sạch: {cleaned_shape[0]:,} dòng x {cleaned_shape[1]:,} cột")
        st.write(f"Tệp đã scale: {download_export.get('scaled_file', 'data/Airbnb_Data_scaled.csv')}")
        if isinstance(scaled_shape, list) and len(scaled_shape) == 2:
            st.write(f"Kích thước dataframe đã scale: {scaled_shape[0]:,} dòng x {scaled_shape[1]:,} cột")
        st.write(f"Tệp đã mã hóa: {download_export.get('encoded_file', 'data/Airbnb_Data_encoded.csv')}")
        if isinstance(encoded_shape, list) and len(encoded_shape) == 2:
            st.write(f"Kích thước dataframe đã mã hóa: {encoded_shape[0]:,} dòng x {encoded_shape[1]:,} cột")

    feature_engineering = step_metrics.get("feature_engineering", {})
    with st.expander("8. Tạo đặc trưng", expanded=True):
        st.write("Trạng thái: Hoàn tất")
        st.write(f"Cột mới được tạo: {_format_column_list(feature_engineering.get('engineered_columns', []))}")
        feature_details = feature_engineering.get("details", [])
        if feature_details:
            for item in feature_details:
                st.write(f"• {item.get('name', '')}")
                st.write(f"  Logic: {item.get('logic', '')}")
                st.write(f"  Cách thực hiện: {item.get('formula', '')}")
        else:
            for definition in feature_engineering.get("definitions", []):
                st.write(f"- {definition}")

    scaling = step_metrics.get("scaling", {})
    with st.expander("9. Chuẩn hóa dữ liệu", expanded=True):
        st.write("Trạng thái: Hoàn tất")
        st.write(f"Bộ scaler đang dùng: {scaling.get('active_scaler', 'MinMaxScaler')}")
        st.write(f"Cột được scale ({int(scaling.get('scaled_column_count', len(scaled_columns)))}): {_format_column_list(scaling.get('scaled_columns', []))}")
        scaled_shape = scaling.get("scaled_shape", [rows_after, columns_after])
        if isinstance(scaled_shape, list) and len(scaled_shape) == 2:
            st.write(f"Kích thước dataframe scaled: {scaled_shape[0]:,} dòng x {scaled_shape[1]:,} cột")
        st.write(
            f"Cột được giữ nguyên không scale: {_format_column_list(scaling.get('passthrough_columns', []))}"
        )
        st.write(f"Các scaler tham chiếu khác: {_format_column_list(scaling.get('alternative_scalers', []))}")
        for note in scaling.get("notes", []):
            st.write(f"- {note}")
        recommended_by_column = scaling.get("recommended_by_column", {})
        if recommended_by_column:
            recommended_table = pd.DataFrame(
                [
                    {"column": column, "recommended_scaling": recommendation}
                    for column, recommendation in recommended_by_column.items()
                ]
            )
            st.dataframe(recommended_table, use_container_width=True, hide_index=True)

    ml_ready_export = step_metrics.get("ml_ready_export", {})
    with st.expander("D. Encoding các cột trong DataFrame (xuất file để đưa vào học máy)", expanded=True):
        st.write("Trạng thái: Hoàn tất")
        generated_counts = ml_ready_export.get("one_hot_generated_counts", {})
        encoding_plan_rows = [
            {"STT": 1, "Cột": "host_id", "Loại dữ liệu": "Chuỗi / định danh", "Encoding": "Giữ nguyên để theo dõi host; không xem đây là biến phân loại cần label encoding.", "Đầu ra": "1 cột host_id"},
            {"STT": 2, "Cột": "host_identity_verified", "Loại dữ liệu": "Nhị phân", "Encoding": 'Mã hóa nhị phân: "unconfirmed" -> 0, "verified" -> 1.', "Đầu ra": "1 cột số"},
            {"STT": 3, "Cột": "neighbourhood_group", "Loại dữ liệu": "Phân loại nhiều mức", "Encoding": "Label Encoding để gom mỗi nhóm khu vực thành một mã số duy nhất, giúp dữ liệu gọn hơn cho mô hình dạng bảng.", "Đầu ra": "1 cột số"},
            {"STT": 4, "Cột": "neighbourhood", "Loại dữ liệu": "Phân loại nhiều mức", "Encoding": "Label Encoding để biến tên khu vực thành mã số, tránh làm tăng quá nhiều số cột trong file ML-ready.", "Đầu ra": "1 cột số"},
            {"STT": 5, "Cột": "room_type", "Loại dữ liệu": "Phân loại danh nghĩa", "Encoding": "One-Hot Encoding vì đây là biến ít mức và không có thứ bậc, nên không phù hợp để gán mã số thứ tự.", "Đầu ra": f"{int(generated_counts.get('room_type', 0)):,} cột nhị phân"},
            {"STT": 6, "Cột": "construction_year", "Loại dữ liệu": "Năm / dữ liệu số", "Encoding": "Không label encode; chỉ chuyển về năm số để mô hình đọc trực tiếp.", "Đầu ra": "1 cột số"},
            {"STT": 7, "Cột": "price", "Loại dữ liệu": "Dữ liệu số liên tục", "Encoding": "Giữ nguyên vì đây là biến số liên tục mang ý nghĩa trực tiếp.", "Đầu ra": "1 cột số"},
            {"STT": 8, "Cột": "minimum_nights", "Loại dữ liệu": "Dữ liệu số", "Encoding": "Giữ nguyên sau khi làm sạch; không cần label encoding.", "Đầu ra": "1 cột số"},
            {"STT": 9, "Cột": "number_of_reviews", "Loại dữ liệu": "Dữ liệu số", "Encoding": "Giữ nguyên để bảo toàn thông tin về mức độ quan tâm của khách hàng.", "Đầu ra": "1 cột số"},
            {"STT": 10, "Cột": "last_review", "Loại dữ liệu": "Datetime", "Encoding": "Chuyển thành `days_since_last_review` rồi loại bỏ cột ngày gốc để mô hình xử lý dễ hơn.", "Đầu ra": "1 cột days_since_last_review"},
            {"STT": 11, "Cột": "review_rate_number", "Loại dữ liệu": "Dữ liệu số", "Encoding": "Giữ nguyên vì giá trị đã ở dạng số có ý nghĩa thứ bậc tự nhiên.", "Đầu ra": "1 cột số"},
            {"STT": 12, "Cột": "calculated_host_listings_count", "Loại dữ liệu": "Dữ liệu số", "Encoding": "Giữ nguyên để phản ánh quy mô listing của host.", "Đầu ra": "1 cột số"},
            {"STT": 13, "Cột": "availability_365", "Loại dữ liệu": "Dữ liệu số", "Encoding": "Giữ nguyên vì đây là biến số quan trọng cho cung và cầu.", "Đầu ra": "1 cột số"},
        ]
        st.dataframe(pd.DataFrame(encoding_plan_rows), use_container_width=True, hide_index=True)
        st.markdown("**Bổ sung cho các cột được tạo từ Feature Engineering**")
        engineered_encoding_rows = [
            {"Cột mới": "booking_demand", "Cách xử lý": "Giữ nguyên numeric để đọc trực tiếp số đêm đã được đặt."},
            {"Cột mới": "availability_category", "Cách xử lý": "Ordinal Encoding vì đây là biến có thứ bậc: Low Availability = 0, Medium Availability = 1, High Availability = 2."},
            {"Cột mới": "availability_efficiency", "Cách xử lý": "Giữ nguyên numeric để so sánh hiệu quả khai thác giữa các nhóm listing."},
            {"Cột mới": "revenue_per_available_night", "Cách xử lý": "Giữ nguyên numeric vì đây là chỉ số doanh thu trung bình trên mỗi đêm khả dụng."},
        ]
        st.dataframe(pd.DataFrame(engineered_encoding_rows), use_container_width=True, hide_index=True)

        ml_shape = ml_ready_export.get("ml_shape", [])
        if isinstance(ml_shape, list) and len(ml_shape) == 2:
            st.write(f"Kích thước dataframe encoded: {ml_shape[0]:,} dòng x {ml_shape[1]:,} cột")
        dropped_identifier_columns = ml_ready_export.get("dropped_identifier_columns", [])
        if dropped_identifier_columns:
            st.write(
                f"Cột định danh bị loại ở bước encoding: {_format_column_list(dropped_identifier_columns)}"
            )
        st.write(
            f"Cột không phải số được giữ lại có chủ đích: {_format_column_list(ml_ready_export.get('non_numeric_columns', []))}"
        )
        st.write(
            "Tóm lại, label encoding chỉ áp dụng cho các cột phân loại nhiều mức như `neighbourhood_group` và `neighbourhood`. "
            "`room_type` dùng one-hot encoding để tránh tạo ra thứ tự giả, còn `availability_category` dùng ordinal encoding vì bản thân biến này có mức độ thấp, trung bình và cao."
        )

        with st.expander("Giải thích các cột One-Hot", expanded=False):
            st.markdown(
                "- `drop_first=True` nghĩa là file one-hot sẽ bỏ đi một nhóm chuẩn, nên số cột sinh ra bằng `số nhóm ban đầu - 1`."
            )
            generated_one_hot_columns = ml_ready_export.get("one_hot_generated_columns", [])
            room_type_columns = [
                column for column in generated_one_hot_columns if column.startswith("room_type_")
            ]
            if generated_counts.get("room_type", 0):
                st.markdown(
                    f"- `room_type` có 4 nhóm gốc, nên sau khi one-hot với `drop_first=True` sẽ còn **{int(generated_counts.get('room_type', 0))} cột**."
                )
                st.write("Các cột được tạo từ `room_type`:")
                st.write(_format_column_list(room_type_columns))
        with st.expander("Ví dụ mã hóa dữ liệu", expanded=False):
            st.code(
                """
encoded_df = df.copy()
encoded_df["host_identity_verified"] = encoded_df["host_identity_verified"].map(
    {"unconfirmed": 0, "verified": 1}
)

def label_encode_text(series, fill_value="unknown"):
    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .fillna(fill_value)
    )
    categories = sorted(normalized.unique().tolist())
    mapping = {value: index for index, value in enumerate(categories)}
    return normalized.map(mapping).astype("int64")

encoded_df["neighbourhood_group"] = label_encode_text(encoded_df["neighbourhood_group"])
encoded_df["neighbourhood"] = label_encode_text(encoded_df["neighbourhood"])

encoded_df["construction_year"] = pd.to_datetime(
    encoded_df["construction_year"],
    errors="coerce",
).dt.year

encoded_df["days_since_last_review"] = (
    pd.to_datetime("today").normalize() - pd.to_datetime(encoded_df["last_review"])
).dt.days
encoded_df = encoded_df.drop(columns=["last_review"])
encoded_df["availability_category"] = encoded_df["availability_category"].map({
    "Low Availability": 0,
    "Medium Availability": 1,
    "High Availability": 2,
})

encoded_df = pd.get_dummies(
    encoded_df,
    columns=["room_type"],
    drop_first=True,
    prefix_sep="__",
    dtype="int64",
)

# Keep numeric columns as-is:
# price, minimum_nights, number_of_reviews, review_rate_number,
# calculated_host_listings_count, availability_365
                """.strip(),
                language="python",
            )

    st.markdown("---")
    st.write(f"Tóm tắt cột đã loại: {_format_column_list(dropped_columns)}")
    st.write(f"Tóm tắt cột đã scale: {_format_column_list(scaled_columns)}")


def render_page(_frame: pd.DataFrame, page_mode: str = "eda") -> None:
    processing_report = st.session_state.get("processing_report")
    before_frame = st.session_state.get("preprocessing_before_df")

    if page_mode == "preprocessing":
        st.title(t("prep.title"))
        st.caption(t("prep.caption"))
        tab_data, tab_steps = st.tabs(["Dữ liệu", "Các bước tiền xử lý"])
        with tab_data:
            render_processing_panel(_frame)
        with tab_steps:
            if processing_report is None or before_frame is None:
                st.info("Chưa có tóm tắt runtime trong session hiện tại. Mô tả từng bước vẫn được hiển thị bên dưới.")
                render_processing_steps_panel()
            else:
                _render_pipeline_summary(processing_report, before_frame)
                st.markdown("---")
                render_processing_steps_panel()
        return

    eda_frame = _prepare_processed_eda_frame(_frame)
    if not isinstance(eda_frame, pd.DataFrame) or eda_frame.empty:
        st.info("Hãy tải CSV ở trang Dữ liệu đầu vào trước để xem các biểu đồ EDA.")
        return

    with st.container():
        viz_frame = eda_frame.copy()
        if "neighbourhood_group" in viz_frame.columns:
            viz_frame["borough_key"] = (
                viz_frame["neighbourhood_group"].astype("string").str.strip().str.lower()
            )
            viz_frame = viz_frame.loc[viz_frame["borough_key"].isin(BOROUGH_DISPLAY_ORDER)].copy()
            viz_frame["borough_label"] = viz_frame["borough_key"].map(BOROUGH_LABEL_MAP)
        if "room_type" in viz_frame.columns:
            room_type_series = viz_frame["room_type"].astype("string").str.strip()
            viz_frame["room_type"] = room_type_series.str.lower().map(ROOM_TYPE_CANONICAL_MAP).fillna(room_type_series)
        if "availability_category" in viz_frame.columns:
            viz_frame["availability_category"] = (
                viz_frame["availability_category"]
                .astype("string")
                .str.strip()
                .replace(
                    {
                        "low availability": "Low Availability",
                        "medium availability": "Medium Availability",
                        "high availability": "High Availability",
                    }
                )
            )

        availability_mix = pd.DataFrame()
        if "availability_category" in viz_frame.columns:
            availability_mix = (
                viz_frame["availability_category"]
                .dropna()
                .loc[lambda s: s.isin(AVAILABILITY_CATEGORY_ORDER)]
                .value_counts(normalize=True)
                .reindex(AVAILABILITY_CATEGORY_ORDER, fill_value=0.0)
                .rename_axis("availability_category")
                .reset_index(name="share")
            )
            availability_mix["availability_category_label"] = availability_mix["availability_category"].map(
                AVAILABILITY_CATEGORY_LABEL_MAP
            )
            availability_mix["share_pct"] = availability_mix["share"] * 100
            pie_chart = px.pie(
                availability_mix,
                names="availability_category_label",
                values="share_pct",
                hole=0.55,
                color="availability_category_label",
                category_orders={
                    "availability_category_label": [
                        AVAILABILITY_CATEGORY_LABEL_MAP[item] for item in AVAILABILITY_CATEGORY_ORDER
                    ]
                },
                color_discrete_map={
                    AVAILABILITY_CATEGORY_LABEL_MAP[item]: color
                    for item, color in AVAILABILITY_CATEGORY_COLOR_MAP.items()
                },
            )
            dominant_category = availability_mix.sort_values("share_pct", ascending=False).iloc[0]
            high_availability_pct = float(
                availability_mix.loc[
                    availability_mix["availability_category"] == "High Availability",
                    "share_pct",
                ].iloc[0]
            )
            _render_chart(
                "1. Phân bố mức độ sẵn có (Cung)",
                pie_chart,
                (
                    f'Insight: {dominant_category["share_pct"]:.1f}% listings thuộc nhóm '
                    f'"{dominant_category["availability_category_label"]}". Điều này cho thấy thị trường đang vận hành khá năng động, '
                    f"với phần lớn căn hộ không còn trống nhiều ngày trong năm. Chỉ khoảng {high_availability_pct:.1f}% "
                    "thuộc nhóm Sẵn có cao, gợi ý rằng nhóm này có thể đang gặp khó khăn hơn trong việc thu hút khách."
                ),
            )

        area_demand = pd.DataFrame()
        if {"borough_label", "booking_demand"}.issubset(viz_frame.columns):
            area_demand = viz_frame.dropna(subset=["borough_label", "booking_demand"]).copy()
            if not area_demand.empty:
                box_chart = px.box(
                    area_demand,
                    x="borough_label",
                    y="booking_demand",
                    color="borough_label",
                    category_orders={"borough_label": AREA_SEQUENCE},
                    color_discrete_map={
                        BOROUGH_LABEL_MAP[key]: color for key, color in BOROUGH_COLOR_MAP.items()
                    },
                )
                booking_demand_rank = (
                    area_demand.groupby("borough_label", as_index=False)["booking_demand"]
                    .median()
                    .sort_values("booking_demand", ascending=False)
                )
                demand_spread = (
                    area_demand.groupby("borough_label")["booking_demand"]
                    .agg(
                        q1=lambda s: s.quantile(0.25),
                        q3=lambda s: s.quantile(0.75),
                    )
                )
                demand_spread["iqr"] = demand_spread["q3"] - demand_spread["q1"]
                lead_one = booking_demand_rank.iloc[0]
                lead_two = booking_demand_rank.iloc[1] if len(booking_demand_rank) > 1 else None
                consistency_note = ""
                if {"Brooklyn", "Manhattan"}.issubset(demand_spread.index):
                    brooklyn_iqr = float(demand_spread.loc["Brooklyn", "iqr"])
                    manhattan_iqr = float(demand_spread.loc["Manhattan", "iqr"])
                    if brooklyn_iqr < manhattan_iqr:
                        consistency_note = " Brooklyn có độ phân tán hẹp hơn Manhattan, nên nhu cầu cũng đồng đều hơn."
                second_clause = (
                    f' và {lead_two["borough_label"]} ({lead_two["booking_demand"]:.0f} đêm)'
                    if lead_two is not None
                    else ""
                )
                _render_chart(
                    "2. Nhu cầu đặt phòng theo khu vực (Cầu)",
                    box_chart,
                    (
                        f'Insight: {lead_one["borough_label"]} dẫn đầu với trung vị khoảng {lead_one["booking_demand"]:.0f} đêm{second_clause}. '
                        "Brooklyn và Manhattan vẫn là hai khu vực có nhu cầu đặt phòng cao và ổn định nhất."
                        f"{consistency_note}"
                    ),
                )

        scatter_df = pd.DataFrame()
        price_demand_corr = None
        if {"price", "booking_demand", "borough_label"}.issubset(viz_frame.columns):
            scatter_columns = ["price", "booking_demand", "borough_label"]
            if "room_type" in viz_frame.columns:
                scatter_columns.append("room_type")
            scatter_df = viz_frame.loc[viz_frame["price"].between(0, 1200), scatter_columns].dropna()
            if not scatter_df.empty:
                scatter_chart = px.scatter(
                    scatter_df,
                    x="price",
                    y="booking_demand",
                    color="borough_label",
                    hover_data=[column for column in ["room_type"] if column in scatter_df.columns],
                    opacity=0.45,
                    category_orders={"borough_label": AREA_SEQUENCE},
                    color_discrete_map={
                        BOROUGH_LABEL_MAP[key]: color for key, color in BOROUGH_COLOR_MAP.items()
                    },
                )
                if len(scatter_df) > 1:
                    price_demand_corr = float(scatter_df[["price", "booking_demand"]].corr().iloc[0, 1])
                    slope, intercept = np.polyfit(
                        scatter_df["price"].astype(float),
                        scatter_df["booking_demand"].astype(float),
                        1,
                    )
                    trend_x = np.linspace(float(scatter_df["price"].min()), float(scatter_df["price"].max()), 100)
                    scatter_chart.add_scatter(
                        x=trend_x,
                        y=slope * trend_x + intercept,
                        mode="lines",
                        name="Đường xu hướng",
                        line=dict(color="#223247", width=3),
                    )
                _render_chart(
                    "3. Giá theo nhu cầu đặt phòng (Giá và cầu)",
                    scatter_chart,
                    (
                        "Insight: Đường xu hướng gần như nằm ngang trong vùng giá từ 0-1200 USD. "
                        f"Hệ số tương quan hiện tại là {price_demand_corr:.3f} cho thấy giá gần như không tỷ lệ nghịch với nhu cầu đặt phòng. "
                        "Nhu cầu cao vẫn xuất hiện ở nhiều mức giá khác nhau, nên vị trí và trải nghiệm có vẻ quan trọng hơn giảm giá đơn thuần."
                        if price_demand_corr is not None
                        else "Insight: Dữ liệu cho thấy nhu cầu vẫn xuất hiện ở nhiều mức giá khác nhau."
                    ),
                )

        efficiency_matrix = pd.DataFrame()
        best_combo_text = "n/a"
        if {"room_type", "borough_label", "availability_efficiency"}.issubset(viz_frame.columns):
            efficiency_matrix = (
                viz_frame.pivot_table(
                    index="room_type",
                    columns="borough_label",
                    values="availability_efficiency",
                    aggfunc="mean",
                )
                .reindex(index=ROOM_SEQUENCE, columns=AREA_SEQUENCE)
            )
            efficiency_values = efficiency_matrix.stack().dropna()
            if not efficiency_values.empty:
                top_pair = efficiency_values.idxmax()
                top_two_pairs = efficiency_values.sort_values(ascending=False).head(2)
                bottom_pair = efficiency_values.idxmin()
                best_combo_text = f"{top_pair[1]} + {top_pair[0]}"
                room_specific_notes: list[str] = []
                for focus_room in ("Private room", "Entire home/apt"):
                    if focus_room in efficiency_matrix.index:
                        room_row = efficiency_matrix.loc[focus_room].dropna()
                        if not room_row.empty:
                            room_specific_notes.append(
                                f"{focus_room} mạnh nhất tại {room_row.idxmax()} ({room_row.max():,.0f})"
                            )
                queens_hotel_note = ""
                if "Hotel room" in efficiency_matrix.index and "Queens" in efficiency_matrix.columns:
                    queens_hotel_value = efficiency_matrix.loc["Hotel room", "Queens"]
                    if pd.notna(queens_hotel_value):
                        queens_hotel_note = f" Hotel room tại Queens đang thấp nhất ở mức {queens_hotel_value:,.0f}."
                heatmap = px.imshow(
                    efficiency_matrix,
                    text_auto=".0f",
                    aspect="auto",
                    color_continuous_scale=["#f4efe8", "#d8a65d", "#c95c36", "#5d2014"],
                )
                _render_chart(
                    "4. Heatmap hiệu quả khai thác theo mức sẵn có (Hiệu quả)",
                    heatmap,
                    (
                        f"Insight: {top_two_pairs.index[0][1]} + {top_two_pairs.index[0][0]} ({top_two_pairs.iloc[0]:,.0f}) "
                        f"và {top_two_pairs.index[1][1]} + {top_two_pairs.index[1][0]} ({top_two_pairs.iloc[1]:,.0f}) "
                        "là hai điểm hiệu quả nổi bật nhất. "
                        + (" ".join(room_specific_notes) + "." if room_specific_notes else "")
                        + queens_hotel_note
                        + f" Cặp thấp nhất hiện tại là {bottom_pair[1]} + {bottom_pair[0]} ({efficiency_values.loc[bottom_pair]:,.0f})."
                    ),
                )
