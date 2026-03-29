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
from core.i18n import (
    get_language,
    localize_dataframe_for_display,
    t,
    translate_availability_category,
    translate_room_type,
)
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


def _lang_text(vi: str, en: str) -> str:
    return en if get_language() == "en" else vi


def _availability_category_label_map() -> dict[str, str]:
    return {
        category: translate_availability_category(category)
        for category in AVAILABILITY_CATEGORY_ORDER
    }


FEATURE_DETAIL_TRANSLATIONS = {
    "booking_demand": {
        "name": {"en": "Booking Demand", "vi": "Booking Demand (Nhu cầu đặt phòng)"},
        "logic": {
            "en": "Estimate booking demand from the number of unavailable nights.",
            "vi": "Tính toán nhu cầu đặt phòng dựa trên số đêm không sẵn có.",
        },
        "formula": "booking_demand = 365 - availability_365",
    },
    "availability_category": {
        "name": {"en": "Availability Category", "vi": "Availability Category (Phân loại mức độ sẵn có)"},
        "logic": {
            "en": "Split listings into Low Availability, Medium Availability, and High Availability groups based on available nights.",
            "vi": "Phân chia các căn hộ thành 3 nhóm Low Availability, Medium Availability và High Availability dựa trên số đêm có sẵn.",
        },
        "formula": "availability_category = pd.cut(availability_365, bins=[-1, 150, 300, 365], labels=['Low Availability', 'Medium Availability', 'High Availability'])",
    },
    "availability_efficiency": {
        "name": {"en": "Availability Efficiency", "vi": "Availability Efficiency (Hiệu quả sẵn có)"},
        "logic": {
            "en": "Evaluate how effectively available nights are used based on price and availability_365.",
            "vi": "Đánh giá hiệu quả sử dụng các đêm có sẵn dựa trên price và availability_365.",
        },
        "formula": "availability_efficiency = price * (365 - availability_365)",
    },
    "revenue_per_available_night": {
        "name": {"en": "Revenue per Available Night", "vi": "Revenue per Available Night (Doanh thu mỗi đêm có sẵn)"},
        "logic": {
            "en": "Evaluate revenue generated per available night when price and availability are linked.",
            "vi": "Đánh giá doanh thu mỗi đêm có sẵn khi giá và mức độ sẵn có liên quan đến nhau.",
        },
        "formula": "revenue_per_available_night = price * (365 - availability_365) / 365",
    },
}


def _feature_detail_key(item: dict[str, object]) -> str | None:
    explicit_key = str(item.get("key") or "").strip()
    if explicit_key in FEATURE_DETAIL_TRANSLATIONS:
        return explicit_key

    formula = str(item.get("formula") or "").strip()
    for key, data in FEATURE_DETAIL_TRANSLATIONS.items():
        if formula == data["formula"]:
            return key

    name = str(item.get("name") or "").lower()
    if "booking demand" in name:
        return "booking_demand"
    if "availability category" in name:
        return "availability_category"
    if "availability efficiency" in name:
        return "availability_efficiency"
    if "revenue per available night" in name:
        return "revenue_per_available_night"
    return None


def _localize_feature_detail(item: dict[str, object]) -> dict[str, str]:
    key = _feature_detail_key(item)
    lang = get_language()
    if key is None:
        return {
            "name": str(item.get("name", "")),
            "logic": str(item.get("logic", "")),
            "formula": str(item.get("formula", "")),
        }

    localized = FEATURE_DETAIL_TRANSLATIONS[key]
    return {
        "name": localized["name"][lang],
        "logic": localized["logic"][lang],
        "formula": localized["formula"],
    }


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


def _format_column_list(columns: list[str], empty_label: str | None = None) -> str:
    if empty_label is None:
        empty_label = _lang_text("Không có", "None")
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
        return _lang_text("Tọa độ", "Coordinate"), "geoid"
    if column in set(OUTLIER_DISPLAY_ORDER):
        return _lang_text("Số", "Numeric"), "numeric"
    if pd.api.types.is_numeric_dtype(series):
        return _lang_text("Số", "Numeric"), "numeric"
    return _lang_text("Phân loại", "Categorical"), "categorical"


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
    if get_language() == "en":
        return method
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
        "host_identity_verified": _lang_text('Điền "unconfirmed"', 'Fill with "unconfirmed"'),
        "neighbourhood": _lang_text("Mode theo neighbourhood_group + room_type", "Mode by neighbourhood_group + room_type"),
        "neighbourhood_group": _lang_text("Ánh xạ theo địa lý", "Geographic mapping"),
        "construction_year": _lang_text("Mode theo neighbourhood + room_type", "Mode by neighbourhood + room_type"),
        "price": _lang_text("Mean theo neighbourhood + room_type", "Mean by neighbourhood + room_type"),
        "service_fee": _lang_text("Xóa cột", "Drop column"),
        "minimum_nights": _lang_text("Lấy trị tuyệt đối, >365 -> trung vị theo nhóm", "Absolute value, >365 -> group median"),
        "number_of_reviews": _lang_text("Trung vị theo neighbourhood + room_type", "Median by neighbourhood + room_type"),
        "last_review": _lang_text("Loại dòng bị thiếu", "Drop missing rows"),
        "reviews_per_month": _lang_text("Xóa cột", "Drop column"),
        "review_rate_number": _lang_text("Mode theo neighbourhood + room_type", "Mode by neighbourhood + room_type"),
        "calculated_host_listings_count": _lang_text("Tần suất host", "Host frequency"),
        "availability_365": _lang_text("Lấy trị tuyệt đối, >365 -> trung vị theo nhóm", "Absolute value, >365 -> group median"),
        "host_id": _lang_text("Sinh giá trị dự phòng duy nhất sau khi điền theo tần suất host", "Generate unique fallback values after host-frequency fill"),
        "host_name": _lang_text("Xóa cột", "Drop column"),
        "name": _lang_text("Xóa cột", "Drop column"),
        "country": _lang_text("Xóa cột", "Drop column"),
        "country_code": _lang_text("Xóa cột", "Drop column"),
        "house_rules": _lang_text("Xóa cột", "Drop column"),
        "license": _lang_text("Xóa cột", "Drop column"),
        "lat": _lang_text("Xóa cột", "Drop column"),
        "long": _lang_text("Xóa cột", "Drop column"),
        "id": _lang_text("Loại trùng rồi xóa", "Deduplicate then drop"),
    }
    for column in dropped_columns:
        concise_labels.setdefault(column, _lang_text("Xóa cột", "Drop column"))
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
                "strategy": strategy_map.get(column, _lang_text("Cần xem xét thủ công", "Needs manual review")),
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
            _lang_text("Đã xóa cột", "Column removed")
            if column in dropped_columns or column not in after_frame.columns
            else _lang_text("Đã loại dòng", "Rows removed")
            if column == "last_review" and int(row["missing_count"]) > 0 and after_missing == 0
            else _lang_text("Đã điền/xử lý", "Filled/processed")
            if after_missing == 0
            else _lang_text("Cần xem xét", "Needs review")
        )
        row_data: dict[str, object] = {
            _lang_text("Cột", "Column"): column,
            _lang_text("Thiếu trước xử lý", "Missing before processing"): int(row["missing_count"]),
            _lang_text("Thiếu sau xử lý", "Missing after processing"): after_missing,
            _lang_text("Cách xử lý", "Handling"): str(row["strategy"]),
            _lang_text("Trạng thái", "Status"): status,
        }
        if section == "numeric":
            row_data[_lang_text("Độ lệch", "Skewness")] = None if row["skewness"] is None else round(float(row["skewness"]), 3)
        table_rows.append(row_data)

    ordered_columns = [
        _lang_text("Cột", "Column"),
        _lang_text("Thiếu trước xử lý", "Missing before processing"),
        _lang_text("Thiếu sau xử lý", "Missing after processing"),
        _lang_text("Cách xử lý", "Handling"),
        _lang_text("Trạng thái", "Status"),
    ]
    if section == "numeric":
        ordered_columns = [
            _lang_text("Cột", "Column"),
            _lang_text("Thiếu trước xử lý", "Missing before processing"),
            _lang_text("Thiếu sau xử lý", "Missing after processing"),
            _lang_text("Độ lệch", "Skewness"),
            _lang_text("Cách xử lý", "Handling"),
            _lang_text("Trạng thái", "Status"),
        ]
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
                _lang_text("Cột", "Column"): column,
                _lang_text("Ngưỡng dưới", "Lower bound"): round(lower, 3),
                _lang_text("Ngưỡng trên", "Upper bound"): round(upper, 3),
                _lang_text("Điều kiện ngoại lệ", "Outlier condition"): threshold_text,
                _lang_text("Số giá trị đã điều chỉnh", "Adjusted values"): int(adjusted_value_counts.get(column, 0)),
                _lang_text("Cách xử lý áp dụng", "Applied handling"): _display_outlier_method(str(method)),
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

    title = (
        _lang_text("Bảng 2 - Giá trị thiếu (dữ liệu số)", "Table 2 - Missing values (numeric data)")
        if section == "numeric"
        else _lang_text("Bảng 1 - Giá trị thiếu (dữ liệu phân loại)", "Table 1 - Missing values (categorical data)")
    )
    meta = (
        _lang_text(f"Chỉ gồm cột số - {len(section_rows):,} cột", f"Numeric columns only - {len(section_rows):,} columns")
        if section == "numeric"
        else _lang_text(f"Chỉ gồm cột phân loại - {len(section_rows):,} cột", f"Categorical columns only - {len(section_rows):,} columns")
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
                            <th>{_lang_text("Cột", "Column")}</th>
                            <th>{_lang_text("Loại dữ liệu", "Data type")}</th>
                            <th>{_lang_text("Số giá trị thiếu", "Missing values")}</th>
                            <th>{_lang_text("Tỷ lệ thiếu", "Missing rate")}</th>
                            <th>{_lang_text("Độ lệch", "Skewness")}</th>
                            <th>{_lang_text("Cách xử lý", "Handling")}</th>
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
        st.info(_lang_text("Không có cột số phù hợp để kiểm tra ngoại lệ.", "No numeric columns are available for outlier checks."))
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
                <div class="audit-card__title">{_lang_text("Bảng 3 - Phát hiện ngoại lệ", "Table 3 - Outlier detection")}</div>
                <div class="audit-card__meta">{_lang_text(f"Chỉ gồm cột số - {len(rows):,} cột", f"Numeric columns only - {len(rows):,} columns")}</div>
            </div>
            <div class="audit-table-wrap">
                <table class="audit-table">
                    <thead>
                        <tr>
                            <th>{_lang_text("Cột", "Column")}</th>
                            <th>{_lang_text("Nhỏ nhất", "Minimum")}</th>
                            <th>{_lang_text("Lớn nhất", "Maximum")}</th>
                            <th>{_lang_text("Độ lệch", "Skewness")}</th>
                            <th>{_lang_text("Cách xử lý tự động", "Automatic handling")}</th>
                            <th>{_lang_text("Số ngoại lệ", "Outlier count")}</th>
                            <th>{_lang_text("Tỷ lệ ngoại lệ", "Outlier rate")}</th>
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

    completed_label = _lang_text("Trạng thái: Hoàn tất", "Status: Completed")
    st.subheader(_lang_text("Tóm tắt lần chạy tiền xử lý", "Latest preprocessing run summary"))
    st.caption(
        _lang_text(
            "Tab này cho biết lần chạy tiền xử lý gần nhất đã thực hiện những gì trên bộ dữ liệu được tải lên.",
            "This tab explains what the latest preprocessing run did to the uploaded dataset.",
        )
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric(t("prep.metric.rows_before"), f"{rows_before:,}")
    metric_cols[1].metric(t("prep.metric.rows_after"), f"{rows_after:,}")
    metric_cols[2].metric(_lang_text("Cột sau xử lý", "Columns after processing"), f"{columns_after:,}")
    metric_cols[3].metric(t("prep.metric.duplicates_removed"), f"{duplicates_removed:,}")
    metric_cols[4].metric(_lang_text("Dòng bị loại do `last_review`", "Rows removed by `last_review`"), f"{rows_dropped_last_review:,}")

    data_cleaning = step_metrics.get("data_cleaning", {})
    with st.expander(_lang_text("1. Làm sạch dữ liệu", "1. Data cleaning"), expanded=True):
        st.write(completed_label)
        st.write(
            _lang_text("Chuẩn hóa tên cột: ", "Column normalization: ")
            + str(data_cleaning.get("column_name_normalization", "lowercase + trim + special chars -> _"))
        )
        st.write(f"{_lang_text('Cột chuỗi đã trim', 'Trimmed string columns')}: {_format_column_list(data_cleaning.get('string_columns_stripped', []))}")
        st.write(f"{_lang_text('Cột văn bản đã làm sạch', 'Sanitized text columns')}: {_format_column_list(data_cleaning.get('text_cleaned_columns', []))}")
        st.write(f"{_lang_text('Cột tiền tệ đã chuẩn hóa', 'Normalized currency columns')}: {_format_column_list(data_cleaning.get('currency_cleaned_columns', []))}")

    duplicate_handling = step_metrics.get("duplicate_handling", {})
    with st.expander(_lang_text("2. Xử lý dữ liệu trùng lặp", "2. Duplicate handling"), expanded=True):
        st.write(completed_label)
        st.write(
            f"{_lang_text('Cột mục tiêu', 'Target column')}: "
            f"{duplicate_handling.get('target_column') or _lang_text('Không tìm thấy cột id', 'No id column found')}"
        )
        st.write(f"{_lang_text('Số bản ghi trùng đã xóa', 'Duplicate records removed')}: {int(duplicate_handling.get('duplicates_removed', 0)):,}")
        st.write(f"{_lang_text('Số dòng sau khi loại trùng', 'Rows after deduplication')}: {int(duplicate_handling.get('rows_after_dedup', rows_after)):,}")

    feature_selection = step_metrics.get("feature_selection", {})
    with st.expander(_lang_text("3. Chọn đặc trưng", "3. Feature selection"), expanded=True):
        st.write(completed_label)
        st.write(f"{_lang_text('Cột đã loại bỏ', 'Dropped columns')} ({len(dropped_columns)}): {_format_column_list(feature_selection.get('dropped_columns', []))}")
        st.write(f"{_lang_text('Số cột còn lại', 'Remaining columns')}: {len(feature_selection.get('remaining_columns', []))}")

    type_conversion = step_metrics.get("data_type_conversion", {})
    with st.expander(_lang_text("4. Chuyển đổi kiểu dữ liệu", "4. Data type conversion"), expanded=True):
        st.write(completed_label)
        st.write(f"{_lang_text('Cột `float64`', '`float64` columns')}: {_format_column_list(type_conversion.get('float64_columns', []))}")
        st.write(f"{_lang_text('Cột datetime', 'Datetime columns')}: {_format_column_list(type_conversion.get('datetime_columns', []))}")
        st.write(f"{_lang_text('Cột chuỗi đã chuyển lowercase', 'Lowercased string columns')}: {_format_column_list(type_conversion.get('lowercase_string_columns', []))}")

    missing_value_handling = step_metrics.get("missing_value_handling", {})
    with st.expander(_lang_text("5. Xử lý giá trị thiếu", "5. Missing-value handling"), expanded=True):
        st.write(completed_label)
        st.write(f"{_lang_text('Listing thiếu `host_id` trước khi điền theo tần suất', 'Listings missing `host_id` before host-frequency fill')}: {int(missing_value_handling.get('host_id_missing_count', 0)):,}")
        st.write(f"{_lang_text('Số dòng bị loại vì thiếu `last_review`', 'Rows removed due to missing `last_review`')}: {int(missing_value_handling.get('rows_dropped_missing_last_review', 0)):,}")
        st.write(
            _lang_text("Số giá trị null còn lại sau tiền xử lý: ", "Remaining null values after preprocessing: ")
            +
            f"{int(missing_value_handling.get('remaining_null_count', 0)):,}"
        )
        st.write(_lang_text("Quy tắc cho dữ liệu phân loại:", "Rules for categorical data:"))
        for rule in missing_value_handling.get("categorical_fill_rules", []):
            st.write(f"- {rule}")
        st.write(_lang_text("Quy tắc cho dữ liệu ngày giờ:", "Rules for datetime data:"))
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
        st.write(_lang_text("Quy tắc cho dữ liệu số:", "Rules for numeric data:"))
        for rule in missing_value_handling.get("numeric_fill_rules", []):
            st.write(f"- {rule}")
        invalid_counts = missing_value_handling.get("invalid_value_counts", {})
        if invalid_counts:
            st.write(
                _lang_text(
                    "Giá trị không hợp lệ đã được chuyển thành missing trước khi điền: ",
                    "Invalid values converted to missing before imputation: ",
                )
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
        st.caption(
            _lang_text(
                "Các bảng thẻ bên dưới làm rõ mẫu thiếu dữ liệu ban đầu và quyết định điền, ánh xạ hoặc loại dòng mà pipeline đã áp dụng.",
                "The cards below make the original missing-data pattern and each fill, mapping, or row-removal decision easier to read.",
            )
        )

    outlier_handling = step_metrics.get("outlier_handling", {})
    with st.expander(_lang_text("6. Xử lý ngoại lệ", "6. Outlier handling"), expanded=True):
        st.write(completed_label)
        st.write(f"{_lang_text('Cột đã chặn ngưỡng', 'Clipped columns')}: {_format_column_list(outlier_handling.get('clipped_columns', []))}")
        st.write(f"{_lang_text('Cột đã làm tròn', 'Rounded columns')}: {_format_column_list(outlier_handling.get('rounded_columns', []))}")
        applied_methods = outlier_handling.get("applied_methods", {})
        if applied_methods:
            st.write(
                _lang_text("Phương pháp đã áp dụng: ", "Applied methods: ")
                + ", ".join(f"{column} -> {_display_outlier_method(str(method))}" for column, method in applied_methods.items())
            )
        outlier_strategy_table = _build_outlier_strategy_table(before_frame, outlier_handling)
        if not outlier_strategy_table.empty:
            st.dataframe(outlier_strategy_table, use_container_width=True, hide_index=True)
        adjusted_value_counts = outlier_handling.get("adjusted_value_counts", {})
        adjusted_table = pd.DataFrame(
            [
                {
                    _lang_text("Cột", "Column"): column,
                    _lang_text("Số giá trị đã điều chỉnh", "Adjusted values"): int(count),
                }
                for column, count in adjusted_value_counts.items()
            ]
        )
        if not adjusted_table.empty:
            st.dataframe(adjusted_table, use_container_width=True, hide_index=True)
        if adjusted_value_counts:
            st.caption(
                _lang_text(
                    "Số giá trị đã điều chỉnh sau khi phát hiện ngoại lệ: ",
                    "Adjusted values after outlier detection: ",
                )
                + ", ".join(f"{column}={int(count):,}" for column, count in adjusted_value_counts.items())
            )
        _render_outlier_card(before_frame)

    integrity_check = step_metrics.get("integrity_check", {})
    validation_status = _lang_text("Đạt", "Passed") if integrity_check.get("passed", False) else _lang_text("Cần lưu ý", "Attention needed")
    st.caption(
        _lang_text("Kiểm tra ràng buộc: ", "Constraint check: ")
        +
        f"{validation_status}. "
        f"availability_365 [0, 365]={integrity_check.get('availability_365_in_range', True)}, "
        f"minimum_nights [1, 365]={integrity_check.get('minimum_nights_in_range', True)}, "
        f"review_rate_number [0, 5]={integrity_check.get('review_rate_number_in_range', True)}."
    )
    if remaining_issues:
        st.warning(_lang_text("Vấn đề còn lại: ", "Remaining issues: ") + "; ".join(str(issue) for issue in remaining_issues))

    download_export = step_metrics.get("download_export", {})
    with st.expander(_lang_text("7. Tải file tiền xử lý", "7. Export preprocessing files"), expanded=True):
        st.write(completed_label)
        cleaned_shape = download_export.get("cleaned_shape", [rows_after, columns_after])
        scaled_shape = download_export.get("scaled_shape", [])
        encoded_shape = download_export.get("encoded_shape", [])
        st.write(f"{_lang_text('Tệp đã làm sạch', 'Cleaned file')}: {download_export.get('cleaned_file', 'data/Airbnb_Data_cleaned.csv')}")
        if isinstance(cleaned_shape, list) and len(cleaned_shape) == 2:
            st.write(f"{_lang_text('Kích thước dataframe đã làm sạch', 'Cleaned dataframe shape')}: {cleaned_shape[0]:,} {_lang_text('dòng', 'rows')} x {cleaned_shape[1]:,} {_lang_text('cột', 'columns')}")
        st.write(f"{_lang_text('Tệp đã scale', 'Scaled file')}: {download_export.get('scaled_file', 'data/Airbnb_Data_scaled.csv')}")
        if isinstance(scaled_shape, list) and len(scaled_shape) == 2:
            st.write(f"{_lang_text('Kích thước dataframe đã scale', 'Scaled dataframe shape')}: {scaled_shape[0]:,} {_lang_text('dòng', 'rows')} x {scaled_shape[1]:,} {_lang_text('cột', 'columns')}")
        st.write(f"{_lang_text('Tệp đã mã hóa', 'Encoded file')}: {download_export.get('encoded_file', 'data/Airbnb_Data_encoded.csv')}")
        if isinstance(encoded_shape, list) and len(encoded_shape) == 2:
            st.write(f"{_lang_text('Kích thước dataframe đã mã hóa', 'Encoded dataframe shape')}: {encoded_shape[0]:,} {_lang_text('dòng', 'rows')} x {encoded_shape[1]:,} {_lang_text('cột', 'columns')}")

    feature_engineering = step_metrics.get("feature_engineering", {})
    with st.expander(_lang_text("8. Tạo đặc trưng", "8. Feature engineering"), expanded=True):
        st.write(completed_label)
        st.write(f"{_lang_text('Cột mới được tạo', 'New columns created')}: {_format_column_list(feature_engineering.get('engineered_columns', []))}")
        feature_details = feature_engineering.get("details", [])
        if feature_details:
            for item in feature_details:
                localized_item = _localize_feature_detail(item)
                st.write(f"• {localized_item['name']}")
                st.write(f"  {_lang_text('Logic', 'Logic')}: {localized_item['logic']}")
                st.write(f"  {_lang_text('Cách thực hiện', 'Implementation')}: {localized_item['formula']}")
        else:
            for definition in feature_engineering.get("definitions", []):
                st.write(f"- {definition}")

    scaling = step_metrics.get("scaling", {})
    with st.expander(_lang_text("9. Chuẩn hóa dữ liệu", "9. Data scaling"), expanded=True):
        st.write(completed_label)
        st.write(f"{_lang_text('Bộ scaler đang dùng', 'Active scaler')}: {scaling.get('active_scaler', 'MinMaxScaler')}")
        st.write(f"{_lang_text('Cột được scale', 'Scaled columns')} ({int(scaling.get('scaled_column_count', len(scaled_columns)))}): {_format_column_list(scaling.get('scaled_columns', []))}")
        scaled_shape = scaling.get("scaled_shape", [rows_after, columns_after])
        if isinstance(scaled_shape, list) and len(scaled_shape) == 2:
            st.write(f"{_lang_text('Kích thước dataframe scaled', 'Scaled dataframe shape')}: {scaled_shape[0]:,} {_lang_text('dòng', 'rows')} x {scaled_shape[1]:,} {_lang_text('cột', 'columns')}")
        st.write(
            f"{_lang_text('Cột được giữ nguyên không scale', 'Columns kept unscaled')}: {_format_column_list(scaling.get('passthrough_columns', []))}"
        )
        st.write(f"{_lang_text('Các scaler tham chiếu khác', 'Reference scalers')}: {_format_column_list(scaling.get('alternative_scalers', []))}")
        for note in scaling.get("notes", []):
            st.write(f"- {note}")
        recommended_by_column = scaling.get("recommended_by_column", {})
        if recommended_by_column:
            recommended_table = pd.DataFrame(
                [
                    {
                        _lang_text("Cột", "Column"): column,
                        _lang_text("Khuyến nghị scale", "Scaling recommendation"): recommendation,
                    }
                    for column, recommendation in recommended_by_column.items()
                ]
            )
            st.dataframe(recommended_table, use_container_width=True, hide_index=True)

    ml_ready_export = step_metrics.get("ml_ready_export", {})
    with st.expander(_lang_text("D. Encoding các cột trong DataFrame (xuất file để đưa vào học máy)", "D. Column encoding for the ML-ready export"), expanded=True):
        st.write(completed_label)
        generated_counts = ml_ready_export.get("one_hot_generated_counts", {})
        encoding_plan_rows = [
            {
                _lang_text("STT", "No."): 1,
                _lang_text("Cột", "Column"): "host_id",
                _lang_text("Loại dữ liệu", "Data type"): _lang_text("Chuỗi / định danh", "String / identifier"),
                _lang_text("Encoding", "Encoding"): _lang_text("Giữ nguyên để theo dõi host; không xem đây là biến phân loại cần label encoding.", "Keep as-is to track hosts; do not treat it as a categorical feature for label encoding."),
                _lang_text("Đầu ra", "Output"): _lang_text("1 cột host_id", "1 host_id column"),
            },
            {
                _lang_text("STT", "No."): 2,
                _lang_text("Cột", "Column"): "host_identity_verified",
                _lang_text("Loại dữ liệu", "Data type"): _lang_text("Nhị phân", "Binary"),
                _lang_text("Encoding", "Encoding"): _lang_text('Mã hóa nhị phân: "unconfirmed" -> 0, "verified" -> 1.', 'Binary encoding: "unconfirmed" -> 0, "verified" -> 1.'),
                _lang_text("Đầu ra", "Output"): _lang_text("1 cột số", "1 numeric column"),
            },
            {
                _lang_text("STT", "No."): 3,
                _lang_text("Cột", "Column"): "neighbourhood_group",
                _lang_text("Loại dữ liệu", "Data type"): _lang_text("Phân loại nhiều mức", "Multi-class categorical"),
                _lang_text("Encoding", "Encoding"): _lang_text("Label Encoding để gom mỗi nhóm khu vực thành một mã số duy nhất, giúp dữ liệu gọn hơn cho mô hình dạng bảng.", "Use label encoding so each neighborhood group becomes a single numeric code and the table model stays compact."),
                _lang_text("Đầu ra", "Output"): _lang_text("1 cột số", "1 numeric column"),
            },
            {
                _lang_text("STT", "No."): 4,
                _lang_text("Cột", "Column"): "neighbourhood",
                _lang_text("Loại dữ liệu", "Data type"): _lang_text("Phân loại nhiều mức", "Multi-class categorical"),
                _lang_text("Encoding", "Encoding"): _lang_text("Label Encoding để biến tên khu vực thành mã số, tránh làm tăng quá nhiều số cột trong file ML-ready.", "Use label encoding to convert area names into numeric codes without exploding the number of columns in the ML-ready file."),
                _lang_text("Đầu ra", "Output"): _lang_text("1 cột số", "1 numeric column"),
            },
            {
                _lang_text("STT", "No."): 5,
                _lang_text("Cột", "Column"): "room_type",
                _lang_text("Loại dữ liệu", "Data type"): _lang_text("Phân loại danh nghĩa", "Nominal categorical"),
                _lang_text("Encoding", "Encoding"): _lang_text("One-Hot Encoding vì đây là biến ít mức và không có thứ bậc, nên không phù hợp để gán mã số thứ tự.", "Use one-hot encoding because this feature has few unordered levels and should not be assigned an artificial rank."),
                _lang_text("Đầu ra", "Output"): _lang_text(f"{int(generated_counts.get('room_type', 0)):,} cột nhị phân", f"{int(generated_counts.get('room_type', 0)):,} binary columns"),
            },
            {
                _lang_text("STT", "No."): 6,
                _lang_text("Cột", "Column"): "construction_year",
                _lang_text("Loại dữ liệu", "Data type"): _lang_text("Năm / dữ liệu số", "Year / numeric"),
                _lang_text("Encoding", "Encoding"): _lang_text("Không label encode; chỉ chuyển về năm số để mô hình đọc trực tiếp.", "Do not label-encode it; convert it to a numeric year so the model can read it directly."),
                _lang_text("Đầu ra", "Output"): _lang_text("1 cột số", "1 numeric column"),
            },
            {
                _lang_text("STT", "No."): 7,
                _lang_text("Cột", "Column"): "price",
                _lang_text("Loại dữ liệu", "Data type"): _lang_text("Dữ liệu số liên tục", "Continuous numeric"),
                _lang_text("Encoding", "Encoding"): _lang_text("Giữ nguyên vì đây là biến số liên tục mang ý nghĩa trực tiếp.", "Keep as-is because it is a continuous numeric feature with direct meaning."),
                _lang_text("Đầu ra", "Output"): _lang_text("1 cột số", "1 numeric column"),
            },
            {
                _lang_text("STT", "No."): 8,
                _lang_text("Cột", "Column"): "minimum_nights",
                _lang_text("Loại dữ liệu", "Data type"): _lang_text("Dữ liệu số", "Numeric"),
                _lang_text("Encoding", "Encoding"): _lang_text("Giữ nguyên sau khi làm sạch; không cần label encoding.", "Keep as-is after cleaning; label encoding is not needed."),
                _lang_text("Đầu ra", "Output"): _lang_text("1 cột số", "1 numeric column"),
            },
            {
                _lang_text("STT", "No."): 9,
                _lang_text("Cột", "Column"): "number_of_reviews",
                _lang_text("Loại dữ liệu", "Data type"): _lang_text("Dữ liệu số", "Numeric"),
                _lang_text("Encoding", "Encoding"): _lang_text("Giữ nguyên để bảo toàn thông tin về mức độ quan tâm của khách hàng.", "Keep as-is to preserve information about customer attention and demand."),
                _lang_text("Đầu ra", "Output"): _lang_text("1 cột số", "1 numeric column"),
            },
            {
                _lang_text("STT", "No."): 10,
                _lang_text("Cột", "Column"): "last_review",
                _lang_text("Loại dữ liệu", "Data type"): "Datetime",
                _lang_text("Encoding", "Encoding"): _lang_text("Chuyển thành `days_since_last_review` rồi loại bỏ cột ngày gốc để mô hình xử lý dễ hơn.", "Convert it to `days_since_last_review` and drop the original date column so the model can handle it more easily."),
                _lang_text("Đầu ra", "Output"): _lang_text("1 cột days_since_last_review", "1 days_since_last_review column"),
            },
            {
                _lang_text("STT", "No."): 11,
                _lang_text("Cột", "Column"): "review_rate_number",
                _lang_text("Loại dữ liệu", "Data type"): _lang_text("Dữ liệu số", "Numeric"),
                _lang_text("Encoding", "Encoding"): _lang_text("Giữ nguyên vì giá trị đã ở dạng số có ý nghĩa thứ bậc tự nhiên.", "Keep as-is because the value is already numeric and has a natural ordering."),
                _lang_text("Đầu ra", "Output"): _lang_text("1 cột số", "1 numeric column"),
            },
            {
                _lang_text("STT", "No."): 12,
                _lang_text("Cột", "Column"): "calculated_host_listings_count",
                _lang_text("Loại dữ liệu", "Data type"): _lang_text("Dữ liệu số", "Numeric"),
                _lang_text("Encoding", "Encoding"): _lang_text("Giữ nguyên để phản ánh quy mô listing của host.", "Keep as-is to reflect the host's listing portfolio size."),
                _lang_text("Đầu ra", "Output"): _lang_text("1 cột số", "1 numeric column"),
            },
            {
                _lang_text("STT", "No."): 13,
                _lang_text("Cột", "Column"): "availability_365",
                _lang_text("Loại dữ liệu", "Data type"): _lang_text("Dữ liệu số", "Numeric"),
                _lang_text("Encoding", "Encoding"): _lang_text("Giữ nguyên vì đây là biến số quan trọng cho cung và cầu.", "Keep as-is because it is a key numeric signal for supply and demand."),
                _lang_text("Đầu ra", "Output"): _lang_text("1 cột số", "1 numeric column"),
            },
        ]
        st.dataframe(pd.DataFrame(encoding_plan_rows), use_container_width=True, hide_index=True)
        st.markdown(f"**{_lang_text('Bổ sung cho các cột được tạo từ Feature Engineering', 'Additions for engineered columns')}**")
        engineered_encoding_rows = [
            {_lang_text("Cột mới", "New column"): "booking_demand", _lang_text("Cách xử lý", "Handling"): _lang_text("Giữ nguyên numeric để đọc trực tiếp số đêm đã được đặt.", "Keep numeric so booked nights remain directly readable.")},
            {_lang_text("Cột mới", "New column"): "availability_category", _lang_text("Cách xử lý", "Handling"): _lang_text("Ordinal Encoding vì đây là biến có thứ bậc: Low Availability = 0, Medium Availability = 1, High Availability = 2.", "Use ordinal encoding because this feature is ordered: Low Availability = 0, Medium Availability = 1, High Availability = 2.")},
            {_lang_text("Cột mới", "New column"): "availability_efficiency", _lang_text("Cách xử lý", "Handling"): _lang_text("Giữ nguyên numeric để so sánh hiệu quả khai thác giữa các nhóm listing.", "Keep numeric to compare operating efficiency across listing groups.")},
            {_lang_text("Cột mới", "New column"): "revenue_per_available_night", _lang_text("Cách xử lý", "Handling"): _lang_text("Giữ nguyên numeric vì đây là chỉ số doanh thu trung bình trên mỗi đêm khả dụng.", "Keep numeric because it already represents average revenue per available night.")},
        ]
        st.dataframe(pd.DataFrame(engineered_encoding_rows), use_container_width=True, hide_index=True)

        ml_shape = ml_ready_export.get("ml_shape", [])
        if isinstance(ml_shape, list) and len(ml_shape) == 2:
            st.write(f"{_lang_text('Kích thước dataframe encoded', 'Encoded dataframe shape')}: {ml_shape[0]:,} {_lang_text('dòng', 'rows')} x {ml_shape[1]:,} {_lang_text('cột', 'columns')}")
        dropped_identifier_columns = ml_ready_export.get("dropped_identifier_columns", [])
        if dropped_identifier_columns:
            st.write(
                f"{_lang_text('Cột định danh bị loại ở bước encoding', 'Identifier columns dropped during encoding')}: {_format_column_list(dropped_identifier_columns)}"
            )
        st.write(
            f"{_lang_text('Cột không phải số được giữ lại có chủ đích', 'Non-numeric columns intentionally kept')}: {_format_column_list(ml_ready_export.get('non_numeric_columns', []))}"
        )
        st.write(
            _lang_text(
                "Tóm lại, label encoding chỉ áp dụng cho các cột phân loại nhiều mức như `neighbourhood_group` và `neighbourhood`. `room_type` dùng one-hot encoding để tránh tạo ra thứ tự giả, còn `availability_category` dùng ordinal encoding vì bản thân biến này có mức độ thấp, trung bình và cao.",
                "In short, label encoding is used only for high-cardinality categorical columns such as `neighbourhood_group` and `neighbourhood`. `room_type` uses one-hot encoding to avoid a fake ordering, while `availability_category` uses ordinal encoding because it is inherently low, medium, and high.",
            )
        )

        with st.expander(_lang_text("Giải thích các cột One-Hot", "One-hot column notes"), expanded=False):
            st.markdown(
                _lang_text(
                    "- `drop_first=True` nghĩa là file one-hot sẽ bỏ đi một nhóm chuẩn, nên số cột sinh ra bằng `số nhóm ban đầu - 1`.",
                    "- `drop_first=True` removes one reference group, so the number of generated columns equals `original groups - 1`.",
                )
            )
            generated_one_hot_columns = ml_ready_export.get("one_hot_generated_columns", [])
            room_type_columns = [
                column for column in generated_one_hot_columns if column.startswith("room_type_")
            ]
            if generated_counts.get("room_type", 0):
                st.markdown(
                    _lang_text(
                        f"- `room_type` có 4 nhóm gốc, nên sau khi one-hot với `drop_first=True` sẽ còn **{int(generated_counts.get('room_type', 0))} cột**.",
                        f"- `room_type` starts with 4 original groups, so after one-hot encoding with `drop_first=True` there are **{int(generated_counts.get('room_type', 0))} columns** left.",
                    )
                )
                st.write(_lang_text("Các cột được tạo từ `room_type`:", "Columns generated from `room_type`:"))
                st.write(_format_column_list(room_type_columns))
        with st.expander(_lang_text("Ví dụ mã hóa dữ liệu", "Encoding example"), expanded=False):
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
    st.write(f"{_lang_text('Tóm tắt cột đã loại', 'Dropped-column summary')}: {_format_column_list(dropped_columns)}")
    st.write(f"{_lang_text('Tóm tắt cột đã scale', 'Scaled-column summary')}: {_format_column_list(scaled_columns)}")


def render_page(_frame: pd.DataFrame, page_mode: str = "eda") -> None:
    processing_report = st.session_state.get("processing_report")
    before_frame = st.session_state.get("preprocessing_before_df")

    if page_mode == "preprocessing":
        st.title(t("prep.title"))
        st.caption(t("prep.caption"))
        tab_data, tab_steps = st.tabs([
            _lang_text("Dữ liệu", "Data"),
            _lang_text("Các bước tiền xử lý", "Preprocessing steps"),
        ])
        with tab_data:
            render_processing_panel(_frame)
        with tab_steps:
            if processing_report is None or before_frame is None:
                st.info(
                    _lang_text(
                        "Chưa có tóm tắt runtime trong session hiện tại. Mô tả từng bước vẫn được hiển thị bên dưới.",
                        "No runtime summary is available in the current session yet. The step-by-step description is still shown below.",
                    )
                )
                render_processing_steps_panel()
            else:
                _render_pipeline_summary(processing_report, before_frame)
                st.markdown("---")
                render_processing_steps_panel()
        return

    eda_frame = _prepare_processed_eda_frame(_frame)
    st.title(t("eda.title"))
    st.caption(t("eda.caption"))
    if not isinstance(eda_frame, pd.DataFrame) or eda_frame.empty:
        st.info(_lang_text("Hãy tải CSV ở trang Dữ liệu đầu vào trước để xem các biểu đồ EDA.", "Upload a CSV in the Input Data page first to view the EDA charts."))
        return

    with st.container():
        viz_frame = eda_frame.copy()
        availability_label_map = _availability_category_label_map()
        room_sequence_display = [translate_room_type(item) for item in ROOM_SEQUENCE]
        if "neighbourhood_group" in viz_frame.columns:
            viz_frame["borough_key"] = (
                viz_frame["neighbourhood_group"].astype("string").str.strip().str.lower()
            )
            viz_frame = viz_frame.loc[viz_frame["borough_key"].isin(BOROUGH_DISPLAY_ORDER)].copy()
            viz_frame["borough_label"] = viz_frame["borough_key"].map(BOROUGH_LABEL_MAP)
        if "room_type" in viz_frame.columns:
            room_type_series = viz_frame["room_type"].astype("string").str.strip()
            viz_frame["room_type"] = room_type_series.str.lower().map(ROOM_TYPE_CANONICAL_MAP).fillna(room_type_series)
            viz_frame["room_type_label"] = viz_frame["room_type"].map(translate_room_type)
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
                availability_label_map
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
                        availability_label_map[item] for item in AVAILABILITY_CATEGORY_ORDER
                    ]
                },
                color_discrete_map={
                    availability_label_map[item]: color
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
                _lang_text("1. Phân bố mức độ sẵn có (Cung)", "1. Availability distribution (Supply)"),
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
                    _lang_text("2. Nhu cầu đặt phòng theo khu vực (Cầu)", "2. Booking demand by area (Demand)"),
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
            if "room_type_label" in viz_frame.columns:
                scatter_columns.append("room_type_label")
            scatter_df = viz_frame.loc[viz_frame["price"].between(0, 1200), scatter_columns].dropna()
            if not scatter_df.empty:
                scatter_chart = px.scatter(
                    scatter_df,
                    x="price",
                    y="booking_demand",
                    color="borough_label",
                    hover_data=[column for column in ["room_type_label"] if column in scatter_df.columns],
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
                        name=_lang_text("Đường xu hướng", "Trend line"),
                        line=dict(color="#223247", width=3),
                    )
                _render_chart(
                    _lang_text("3. Giá theo nhu cầu đặt phòng (Giá và cầu)", "3. Price versus booking demand (Price and demand)"),
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
                    index="room_type_label" if "room_type_label" in viz_frame.columns else "room_type",
                    columns="borough_label",
                    values="availability_efficiency",
                    aggfunc="mean",
                )
                .reindex(index=room_sequence_display, columns=AREA_SEQUENCE)
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
                    _lang_text("4. Heatmap hiệu quả khai thác theo mức sẵn có (Hiệu quả)", "4. Availability-efficiency heatmap (Efficiency)"),
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
