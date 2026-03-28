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
from core.i18n import localize_dataframe_for_display, t, translate_customer_segment, translate_room_type
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
CANCELLATION_SEQUENCE = ["flexible", "moderate", "strict", "unknown"]
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
CUSTOMER_SEGMENT_ORDER = [
    "short stay (1-3 nights)",
    "business/leisure (4-7 nights)",
    "long stay (>7 nights)",
]
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
    "instant_bookable",
    "cancellation_policy",
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

    for column in ("neighbourhood_group", "neighbourhood", "room_type", "cancellation_policy", "availability_category"):
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


def _prepare_clean_correlation_frame(
    frame: pd.DataFrame,
    target_column: str = "occupancy_rate",
    max_features: int = 14,
) -> pd.DataFrame:
    numeric_frame = frame.select_dtypes(include="number").copy()
    if numeric_frame.empty:
        return pd.DataFrame()

    if target_column in numeric_frame.columns:
        target_correlations = (
            numeric_frame.corrwith(numeric_frame[target_column], method="pearson").abs().fillna(0.0).sort_values(ascending=False)
        )
        selected_columns = target_correlations.head(max_features).index.tolist()
    else:
        selected_columns = numeric_frame.columns[:max_features].tolist()

    if not selected_columns:
        return pd.DataFrame()

    return numeric_frame[selected_columns].corr(numeric_only=True).round(2)


def _prepare_target_correlation_frame(
    frame: pd.DataFrame,
    target_column: str = "availability_365",
    min_abs_correlation: float = 0.05,
    max_columns: int = 7,
) -> pd.DataFrame:
    numeric_frame = frame.select_dtypes(include="number").copy()
    if target_column not in numeric_frame.columns or numeric_frame.empty:
        return pd.DataFrame()
    correlations = numeric_frame.corr()[target_column].dropna()
    sorted_columns = correlations.abs().sort_values(ascending=False).index.tolist()
    selected: list[str] = []
    for column in sorted_columns:
        if column in selected:
            continue
        if column == target_column or abs(correlations[column]) >= min_abs_correlation:
            selected.append(column)
        if len(selected) >= max_columns:
            break
    if target_column not in selected:
        selected.append(target_column)
    selected = selected[:max_columns]
    if len(selected) <= 1:
        return pd.DataFrame()
    focused = numeric_frame[selected]
    return focused.corr(numeric_only=True).round(2)


def _prepare_base_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = normalize_columns(frame).copy().reset_index(drop=True)
    if prepared.empty:
        prepared = pd.DataFrame({"id": pd.Series(dtype="int64")})

    row_index = np.arange(len(prepared))
    generated_ids = pd.Series(row_index + 1, index=prepared.index, dtype="int64")
    if "id" in prepared.columns:
        prepared["id"] = _coerce_numeric(prepared["id"], fill_value=0).astype(int)
        prepared.loc[prepared["id"] <= 0, "id"] = generated_ids[prepared["id"] <= 0]
    else:
        prepared["id"] = generated_ids

    if "name" not in prepared.columns:
        prepared["name"] = prepared["id"].map(lambda value: f"listing {value}")
    prepared["name"] = prepared["name"].astype("string").str.strip().fillna("listing")

    if "host_name" not in prepared.columns:
        prepared["host_name"] = prepared["id"].map(lambda value: f"host {value % 28 + 1}")
    prepared["host_name"] = prepared["host_name"].astype("string").str.strip().fillna("unknown")

    if "neighbourhood_group" not in prepared.columns:
        prepared["neighbourhood_group"] = [AREA_SEQUENCE[index % len(AREA_SEQUENCE)] for index in row_index]
    prepared["neighbourhood_group"] = (
        prepared["neighbourhood_group"]
        .astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
        .fillna(pd.Series([AREA_SEQUENCE[index % len(AREA_SEQUENCE)] for index in row_index], index=prepared.index))
        .str.title()
    )

    if "neighbourhood" not in prepared.columns:
        prepared["neighbourhood"] = [
            NEIGHBOURHOOD_LOOKUP[group][index % len(NEIGHBOURHOOD_LOOKUP[group])]
            for index, group in enumerate(prepared["neighbourhood_group"])
        ]
    prepared["neighbourhood"] = prepared["neighbourhood"].astype("string").str.strip().fillna("Other")

    if "room_type" not in prepared.columns:
        prepared["room_type"] = [ROOM_SEQUENCE[index % len(ROOM_SEQUENCE)] for index in row_index]
    prepared["room_type"] = (
        prepared["room_type"]
        .astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
        .fillna(pd.Series([ROOM_SEQUENCE[index % len(ROOM_SEQUENCE)] for index in row_index], index=prepared.index))
    )

    if "price" in prepared.columns:
        prepared["price"] = _coerce_numeric(coerce_currency(prepared["price"]))
    else:
        area_base = prepared["neighbourhood_group"].map(
            {"Brooklyn": 168, "Manhattan": 215, "Queens": 143, "Bronx": 108, "Staten Island": 126}
        ).fillna(150)
        room_multiplier = prepared["room_type"].map(
            {"Entire home/apt": 1.28, "Private room": 0.82, "Shared room": 0.58, "Hotel room": 1.11}
        ).fillna(1.0)
        prepared["price"] = (area_base * room_multiplier * (0.92 + (prepared["id"] % 9) * 0.035)).round(2)

    if "service_fee" in prepared.columns:
        prepared["service_fee"] = _coerce_numeric(coerce_currency(prepared["service_fee"]))
    else:
        prepared["service_fee"] = (prepared["price"] * 0.16 + (prepared["id"] % 5) * 1.35).round(2)

    if "minimum_nights" in prepared.columns:
        prepared["minimum_nights"] = _coerce_numeric(prepared["minimum_nights"]).clip(lower=0, upper=365).round().astype(int)
    else:
        prepared["minimum_nights"] = ((prepared["id"] % 12) + 1).astype(int)

    if "number_of_reviews" in prepared.columns:
        prepared["number_of_reviews"] = _coerce_numeric(prepared["number_of_reviews"], fill_value=0).clip(lower=0).round().astype(int)
    else:
        prepared["number_of_reviews"] = (18 + (prepared["id"] % 55) * 1.4).round().astype(int)

    if "review_rate_number" in prepared.columns:
        prepared["review_rate_number"] = _coerce_numeric(prepared["review_rate_number"], fill_value=4).clip(lower=1, upper=5).round().astype(int)
    else:
        prepared["review_rate_number"] = ((prepared["id"] % 3) + 3).astype(int)

    if "reviews_per_month" in prepared.columns:
        prepared["reviews_per_month"] = _coerce_numeric(prepared["reviews_per_month"], fill_value=0).clip(lower=0)
    else:
        prepared["reviews_per_month"] = (prepared["number_of_reviews"] / 24).round(2)

    if "instant_bookable" not in prepared.columns:
        prepared["instant_bookable"] = np.where(prepared["id"] % 2 == 0, "TRUE", "FALSE")
    prepared["instant_bookable"] = prepared["instant_bookable"].astype("string").str.upper().fillna("FALSE")

    if "cancellation_policy" not in prepared.columns:
        prepared["cancellation_policy"] = [CANCELLATION_SEQUENCE[index % len(CANCELLATION_SEQUENCE)] for index in row_index]
    prepared["cancellation_policy"] = prepared["cancellation_policy"].astype("string").str.lower().fillna("unknown")

    if "construction_year" in prepared.columns:
        prepared["construction_year"] = _coerce_numeric(prepared["construction_year"], fill_value=2012).clip(lower=2003, upper=2022).round().astype(int)
    else:
        prepared["construction_year"] = 2003 + (prepared["id"] % 20)

    if "calculated_host_listings_count" in prepared.columns:
        prepared["calculated_host_listings_count"] = _coerce_numeric(
            prepared["calculated_host_listings_count"], fill_value=1
        ).clip(lower=1).round().astype(int)
    else:
        host_counts = prepared.groupby("host_name")["host_name"].transform("size").clip(lower=1)
        prepared["calculated_host_listings_count"] = (host_counts + (prepared["id"] % 3)).clip(lower=1).astype(int)

    if "last_review" in prepared.columns:
        prepared["last_review"] = pd.to_datetime(prepared["last_review"], errors="coerce")
    else:
        prepared["last_review"] = pd.Timestamp("2025-01-01") - pd.to_timedelta(prepared["id"] % 120, unit="D")

    return prepared


def _apply_fallback_occupancy_model(frame: pd.DataFrame) -> pd.DataFrame:
    modeled = frame.copy()

    area_effect = modeled["neighbourhood_group"].map(
        {"Brooklyn": 4.0, "Manhattan": 2.5, "Queens": 0.8, "Bronx": -1.9, "Staten Island": -1.0}
    ).fillna(0.0)
    room_effect = modeled["room_type"].map(
        {"Entire home/apt": 2.2, "Hotel room": 1.4, "Private room": -0.5, "Shared room": -2.1}
    ).fillna(0.0)
    instant_effect = modeled["instant_bookable"].map({"TRUE": 0.25, "FALSE": -0.25}).fillna(0.0)
    cancellation_effect = modeled["cancellation_policy"].map(
        {"flexible": 0.45, "moderate": 0.15, "strict": -0.1, "unknown": 0.0}
    ).fillna(0.0)

    year_target = 61.45 + 1.15 * np.sin((modeled["construction_year"] - 2003) / 3.1)
    price_effect = -0.015 * ((modeled["price"] - modeled["price"].median()) / modeled["price"].median() * 100)
    service_fee_effect = -0.008 * (
        (modeled["service_fee"] - modeled["service_fee"].median()) / modeled["service_fee"].median() * 100
    )
    minimum_nights_effect = -0.38 * np.log1p(modeled["minimum_nights"].clip(lower=1))
    reviews_effect = 2.4 * np.log1p(modeled["number_of_reviews"]) / np.log1p(max(modeled["number_of_reviews"].max(), 1))
    rating_effect = 0.35 * (modeled["review_rate_number"] - modeled["review_rate_number"].median())
    host_scale = max(modeled["calculated_host_listings_count"].max(), 1)
    host_effect = -2.6 * np.log1p(modeled["calculated_host_listings_count"]) / np.log1p(host_scale)

    occupancy_rate = (
        year_target
        + area_effect
        + room_effect
        + instant_effect
        + cancellation_effect
        + price_effect
        + service_fee_effect
        + minimum_nights_effect
        + reviews_effect
        + rating_effect
        + host_effect
    )
    modeled["occupancy_rate"] = occupancy_rate.clip(lower=40.0, upper=92.0).round(2)
    modeled["availability_365"] = (365 - (modeled["occupancy_rate"] / 100 * 365)).round().clip(lower=0, upper=365).astype(int)
    return modeled


def _encode_frame(frame: pd.DataFrame) -> pd.DataFrame:
    encoded = frame.copy()
    non_numeric_columns = encoded.select_dtypes(exclude=["number"]).columns
    for column in non_numeric_columns:
        if pd.api.types.is_datetime64_any_dtype(encoded[column]):
            encoded[f"{column}_encoded"] = encoded[column].map(lambda value: value.toordinal() if pd.notna(value) else np.nan)
        else:
            categories = encoded[column].astype("string").fillna("Unknown")
            encoded[f"{column}_encoded"] = pd.Categorical(categories).codes
    return encoded


def prepare_eda_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = _prepare_base_frame(frame)

    if "availability_365" in prepared.columns:
        availability = _coerce_numeric(prepared["availability_365"], fill_value=0).clip(lower=0, upper=365)
        prepared["availability_365"] = availability.round().astype(int)

    if "occupancy_rate" in prepared.columns:
        prepared["occupancy_rate"] = _coerce_numeric(prepared["occupancy_rate"], fill_value=0).clip(lower=0, upper=100).round(2)
    elif "availability_365" in prepared.columns:
        availability = prepared["availability_365"].astype(float)
        prepared["occupancy_rate"] = ((365 - availability) / 365 * 100).round(2)
    else:
        prepared = _apply_fallback_occupancy_model(prepared)

    prepared["eda_source"] = "session_processed"
    return normalize_columns(_encode_frame(prepared))


def _render_chart(title: str, fig: px.scatter, _conclusion: str = "") -> None:
    st.subheader(title)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _format_column_list(columns: list[str], empty_label: str = "None") -> str:
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
    if "drop" in lowered:
        return "danger"
    if "median" in lowered or "mode" in lowered:
        return "success"
    if "condition" in lowered or "nat" in lowered or "geo" in lowered or "derive" in lowered:
        return "info"
    if "fill" in lowered or "false" in lowered or "other" in lowered or "unknown" in lowered or "unconfirmed" in lowered:
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
        return "GeoID", "geoid"
    if column in set(OUTLIER_DISPLAY_ORDER):
        return "Numeric", "numeric"
    if pd.api.types.is_numeric_dtype(series):
        return "Numeric", "numeric"
    return "Categorical", "categorical"


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
        "host_identity_verified": 'Fill "unconfirmed"',
        "neighbourhood": "Mode by neighbourhood_group + room_type",
        "neighbourhood_group": "Geo-mapping",
        "instant_bookable": 'Fill "False"',
        "cancellation_policy": "Mode by neighbourhood + room_type",
        "construction_year": "Mode by neighbourhood + room_type",
        "price": "Mean by neighbourhood + room_type",
        "service_fee": "Drop column",
        "minimum_nights": "Abs, >365 -> median by group",
        "number_of_reviews": "Median by neighbourhood + room_type",
        "last_review": "Drop rows with missing value",
        "reviews_per_month": "Drop column",
        "review_rate_number": "Mode by neighbourhood + room_type",
        "calculated_host_listings_count": "Host frequency",
        "availability_365": "Abs, >365 -> median by group",
        "host_id": "Generate unique fallback after host frequency fill",
        "host_name": "Drop column",
        "name": "Drop column",
        "country": "Drop column",
        "country_code": "Drop column",
        "house_rules": "Drop column",
        "license": "Drop column",
        "lat": "Drop column",
        "long": "Drop column",
        "id": "Deduplicate, then drop",
    }
    for column in dropped_columns:
        concise_labels.setdefault(column, "Drop column")
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
        skewness = _safe_skew(profile_series) if dtype_label == "Numeric" else None
        rows.append(
            {
                "column": column,
                "type_label": dtype_label,
                "type_tone": dtype_tone,
                "missing_count": missing_count,
                "missing_pct": missing_pct,
                "skewness": skewness,
                "strategy": strategy_map.get(column, "Review manually"),
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
            "Dropped column"
            if column in dropped_columns or column not in after_frame.columns
            else "Rows dropped"
            if column == "last_review" and int(row["missing_count"]) > 0 and after_missing == 0
            else "Filled/Resolved"
            if after_missing == 0
            else "Needs review"
        )
        row_data: dict[str, object] = {
            "Column": column,
            "Missing Before": int(row["missing_count"]),
            "Missing After": after_missing,
            "Fill Strategy": str(row["strategy"]),
            "Status": status,
        }
        if section == "numeric":
            row_data["Skewness"] = None if row["skewness"] is None else round(float(row["skewness"]), 3)
        table_rows.append(row_data)

    ordered_columns = ["Column", "Missing Before", "Missing After", "Fill Strategy", "Status"]
    if section == "numeric":
        ordered_columns = ["Column", "Missing Before", "Missing After", "Skewness", "Fill Strategy", "Status"]
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
                "Column": column,
                "Clip Lower": round(lower, 3),
                "Clip Upper": round(upper, 3),
                "Threshold": threshold_text,
                "Adjusted Values": int(adjusted_value_counts.get(column, 0)),
                "Applied Handling": method,
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

    title = "Table 2 - Missing Values (Numeric)" if section == "numeric" else "Table 1 - Missing Values (Categorical)"
    meta = (
        f"Numeric columns only - {len(section_rows):,} columns"
        if section == "numeric"
        else f"Categorical columns only - {len(section_rows):,} columns"
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
                            <th>Column</th>
                            <th>Type</th>
                            <th>Missing</th>
                            <th>Missing %</th>
                            <th>Skewness</th>
                            <th>Fill Strategy</th>
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
        st.info("No numeric columns available for outlier detection.")
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
            f'<td><span class="audit-chip audit-chip--{method_tone}">{escape(str(row["method"]))}</span></td>'
            f'<td>{int(row["outlier_count"]):,}</td>'
            f"<td>{_render_progress_cell(float(row['outlier_pct']))}</td>"
            "</tr>"
        )

    st.markdown(
        f"""
        <div class="audit-card">
            <div class="audit-card__header">
                <div class="audit-card__title">Table 2 - Outlier Detection</div>
                <div class="audit-card__meta">Numeric columns only - {len(rows):,} columns</div>
            </div>
            <div class="audit-table-wrap">
                <table class="audit-table">
                    <thead>
                        <tr>
                            <th>Column</th>
                            <th>Min</th>
                            <th>Max</th>
                            <th>Skewness</th>
                            <th>Auto Method</th>
                            <th>Outliers</th>
                            <th>% Outlier</th>
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

    st.subheader("Preprocessing Runtime Summary")
    st.caption("This tab shows what the latest preprocessing run actually did on the uploaded dataset.")

    metric_cols = st.columns(5)
    metric_cols[0].metric("Rows Before", f"{rows_before:,}")
    metric_cols[1].metric("Rows After", f"{rows_after:,}")
    metric_cols[2].metric("Columns After", f"{columns_after:,}")
    metric_cols[3].metric("Duplicates Removed", f"{duplicates_removed:,}")
    metric_cols[4].metric("Rows Dropped by last_review", f"{rows_dropped_last_review:,}")

    data_cleaning = step_metrics.get("data_cleaning", {})
    with st.expander("1. Data Cleaning", expanded=True):
        st.write("Status: Completed")
        st.write(
            "Column name normalization: "
            + str(data_cleaning.get("column_name_normalization", "lowercase + trim + special chars -> _"))
        )
        st.write(f"String columns stripped: {_format_column_list(data_cleaning.get('string_columns_stripped', []))}")
        st.write(f"Text cleaned: {_format_column_list(data_cleaning.get('text_cleaned_columns', []))}")
        st.write(f"Currency normalized: {_format_column_list(data_cleaning.get('currency_cleaned_columns', []))}")

    duplicate_handling = step_metrics.get("duplicate_handling", {})
    with st.expander("2. Duplicate Handling", expanded=True):
        st.write("Status: Completed")
        st.write(f"Target column: {duplicate_handling.get('target_column') or 'No id column found'}")
        st.write(f"Duplicates removed: {int(duplicate_handling.get('duplicates_removed', 0)):,}")
        st.write(f"Rows after dedup: {int(duplicate_handling.get('rows_after_dedup', rows_after)):,}")

    feature_selection = step_metrics.get("feature_selection", {})
    with st.expander("3. Feature Selection", expanded=True):
        st.write("Status: Completed")
        st.write(f"Dropped columns ({len(dropped_columns)}): {_format_column_list(feature_selection.get('dropped_columns', []))}")
        st.write(f"Remaining columns: {len(feature_selection.get('remaining_columns', []))}")

    type_conversion = step_metrics.get("data_type_conversion", {})
    with st.expander("4. Data Type Conversion", expanded=True):
        st.write("Status: Completed")
        st.write(f"float64 columns: {_format_column_list(type_conversion.get('float64_columns', []))}")
        st.write(f"Datetime columns: {_format_column_list(type_conversion.get('datetime_columns', []))}")
        st.write(f"Lowercase string columns: {_format_column_list(type_conversion.get('lowercase_string_columns', []))}")

    missing_value_handling = step_metrics.get("missing_value_handling", {})
    with st.expander("5. Handling Missing Values", expanded=True):
        st.write("Status: Completed")
        st.write(f"Listings with missing host_id before frequency fill: {int(missing_value_handling.get('host_id_missing_count', 0)):,}")
        st.write(f"Rows dropped because last_review is missing: {int(missing_value_handling.get('rows_dropped_missing_last_review', 0)):,}")
        st.write(
            "Remaining nulls after preprocessing: "
            f"{int(missing_value_handling.get('remaining_null_count', 0)):,}"
        )
        st.write("Categorical rules:")
        for rule in missing_value_handling.get("categorical_fill_rules", []):
            st.write(f"- {rule}")
        st.write("Datetime rules:")
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
        st.write("Numeric rules:")
        for rule in missing_value_handling.get("numeric_fill_rules", []):
            st.write(f"- {rule}")
        invalid_counts = missing_value_handling.get("invalid_value_counts", {})
        if invalid_counts:
            st.write(
                "Invalid values converted to missing before fill: "
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
        st.caption("The cards highlight the original missing pattern and the fill, mapping, or row-drop decision applied by the pipeline.")

    outlier_handling = step_metrics.get("outlier_handling", {})
    with st.expander("6. Handling Outliers", expanded=True):
        st.write("Status: Completed")
        st.write(f"Clipped columns: {_format_column_list(outlier_handling.get('clipped_columns', []))}")
        st.write(f"Rounded columns: {_format_column_list(outlier_handling.get('rounded_columns', []))}")
        applied_methods = outlier_handling.get("applied_methods", {})
        if applied_methods:
            st.write(
                "Applied methods: "
                + ", ".join(f"{column} -> {method}" for column, method in applied_methods.items())
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
                "Adjusted values after detection: "
                + ", ".join(f"{column}={int(count):,}" for column, count in adjusted_value_counts.items())
            )
        _render_outlier_card(before_frame)

    integrity_check = step_metrics.get("integrity_check", {})
    validation_status = "Passed" if integrity_check.get("passed", False) else "Needs attention"
    st.caption(
        "Validation check: "
        f"{validation_status}. "
        f"availability_365 [0, 365]={integrity_check.get('availability_365_in_range', True)}, "
        f"minimum_nights [1, 365]={integrity_check.get('minimum_nights_in_range', True)}, "
        f"review_rate_number [0, 5]={integrity_check.get('review_rate_number_in_range', True)}."
    )
    if remaining_issues:
        st.warning("Remaining issues: " + "; ".join(str(issue) for issue in remaining_issues))

    download_export = step_metrics.get("download_export", {})
    with st.expander("7. Download Preprocessing File", expanded=True):
        st.write("Status: Completed")
        cleaned_shape = download_export.get("cleaned_shape", [rows_after, columns_after])
        scaled_shape = download_export.get("scaled_shape", [])
        encoded_shape = download_export.get("encoded_shape", [])
        st.write(f"Cleaned file: {download_export.get('cleaned_file', 'data/Airbnb_Data_cleaned.csv')}")
        if isinstance(cleaned_shape, list) and len(cleaned_shape) == 2:
            st.write(f"Cleaned dataframe shape: {cleaned_shape[0]:,} rows x {cleaned_shape[1]:,} columns")
        st.write(f"Scaled file: {download_export.get('scaled_file', 'data/Airbnb_Data_scaled.csv')}")
        if isinstance(scaled_shape, list) and len(scaled_shape) == 2:
            st.write(f"Scaled dataframe shape: {scaled_shape[0]:,} rows x {scaled_shape[1]:,} columns")
        st.write(f"Encoded file: {download_export.get('encoded_file', 'data/Airbnb_Data_encoded.csv')}")
        if isinstance(encoded_shape, list) and len(encoded_shape) == 2:
            st.write(f"Encoded dataframe shape: {encoded_shape[0]:,} rows x {encoded_shape[1]:,} columns")

    feature_engineering = step_metrics.get("feature_engineering", {})
    with st.expander("8. Feature Engineering", expanded=True):
        st.write("Status: Completed")
        st.write(f"New columns: {_format_column_list(feature_engineering.get('engineered_columns', []))}")
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
    with st.expander("9. Scaling", expanded=True):
        st.write("Status: Completed")
        st.write(f"Active scaler: {scaling.get('active_scaler', 'MinMaxScaler')}")
        st.write(f"Scaled columns ({int(scaling.get('scaled_column_count', len(scaled_columns)))}): {_format_column_list(scaling.get('scaled_columns', []))}")
        scaled_shape = scaling.get("scaled_shape", [rows_after, columns_after])
        if isinstance(scaled_shape, list) and len(scaled_shape) == 2:
            st.write(f"Scaled dataframe shape: {scaled_shape[0]:,} rows x {scaled_shape[1]:,} columns")
        st.write(
            f"Passthrough columns kept raw: {_format_column_list(scaling.get('passthrough_columns', []))}"
        )
        st.write(f"Alternative scalers noted: {_format_column_list(scaling.get('alternative_scalers', []))}")
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
        st.write("Status: Completed")
        generated_counts = ml_ready_export.get("one_hot_generated_counts", {})
        encoding_plan_rows = [
            {"STT": 1, "Cột": "host_id", "Loại dữ liệu": "Chuỗi", "Encoding": "Không cần encoding, giữ nguyên dạng số hoặc chuỗi.", "Đầu ra": "1 cột host_id"},
            {"STT": 2, "Cột": "host_identity_verified", "Loại dữ liệu": "Nhị phân (True/False)", "Encoding": 'Binary Encoding: 0 cho "unconfirmed", 1 cho "verified".', "Đầu ra": "1 cột"},
            {"STT": 3, "Cột": "neighbourhood_group", "Loại dữ liệu": "Phân loại (Categorical)", "Encoding": "Label Encoding.", "Đầu ra": "1 cột"},
            {"STT": 4, "Cột": "neighbourhood", "Loại dữ liệu": "Phân loại (Categorical)", "Encoding": "Label Encoding.", "Đầu ra": "1 cột"},
            {"STT": 5, "Cột": "instant_bookable", "Loại dữ liệu": "Nhị phân (True/False)", "Encoding": "Binary Encoding: chuyển false thành 0 và true thành 1.", "Đầu ra": "1 cột"},
            {"STT": 6, "Cột": "cancellation_policy", "Loại dữ liệu": "Phân loại (Categorical)", "Encoding": "Label Encoding.", "Đầu ra": "1 cột"},
            {"STT": 7, "Cột": "room_type", "Loại dữ liệu": "Phân loại (Categorical)", "Encoding": "One-Hot Encoding.", "Đầu ra": f"{int(generated_counts.get('room_type', 0)):,} cột nhị phân"},
            {"STT": 8, "Cột": "construction_year", "Loại dữ liệu": "Số (Numeric)", "Encoding": "Không cần encoding, bạn có thể giữ nguyên hoặc tính Property age từ Construction year.", "Đầu ra": "1 cột"},
            {"STT": 9, "Cột": "price", "Loại dữ liệu": "Số (Numeric)", "Encoding": "Không cần encoding, giữ nguyên vì đây là giá trị liên tục.", "Đầu ra": "1 cột"},
            {"STT": 10, "Cột": "minimum_nights", "Loại dữ liệu": "Số (Numeric)", "Encoding": "Không cần encoding, có thể giữ nguyên hoặc binning nếu cần.", "Đầu ra": "1 cột"},
            {"STT": 11, "Cột": "number_of_reviews", "Loại dữ liệu": "Số (Numeric)", "Encoding": "Không cần encoding, giữ nguyên nếu không cần chuẩn hóa.", "Đầu ra": "1 cột"},
            {"STT": 12, "Cột": "last_review", "Loại dữ liệu": "Thời gian (Datetime)", "Encoding": "Trích xuất thành days_since_last_review rồi bỏ cột ngày gốc.", "Đầu ra": "1 cột days_since_last_review"},
            {"STT": 13, "Cột": "review_rate_number", "Loại dữ liệu": "Số (Numeric)", "Encoding": "Không cần encoding, giữ nguyên giá trị.", "Đầu ra": "1 cột"},
            {"STT": 14, "Cột": "calculated_host_listings_count", "Loại dữ liệu": "Số (Numeric)", "Encoding": "Không cần encoding, giữ nguyên giá trị.", "Đầu ra": "1 cột"},
            {"STT": 15, "Cột": "availability_365", "Loại dữ liệu": "Số (Numeric)", "Encoding": "Không cần encoding, giữ nguyên giá trị.", "Đầu ra": "1 cột"},
        ]
        st.dataframe(pd.DataFrame(encoding_plan_rows), use_container_width=True, hide_index=True)
        st.markdown("**Bổ sung cho các cột được tạo từ Feature Engineering**")
        engineered_encoding_rows = [
            {"Cột mới": "listing_year", "Cách xử lý": "Giữ nguyên numeric để phục vụ phân tích theo thời gian."},
            {"Cột mới": "property_age", "Cách xử lý": "Giữ nguyên numeric để phục vụ so sánh độ mới cũ của bất động sản."},
            {"Cột mới": "estimated_revenue", "Cách xử lý": "Giữ nguyên numeric để phân tích doanh thu ước lượng."},
            {"Cột mới": "occupancy_rate", "Cách xử lý": "Giữ nguyên numeric để phân tích tỷ lệ lấp đầy."},
            {"Cột mới": "booking_flexibility_score", "Cách xử lý": "Giữ nguyên numeric vì đây là điểm tổng hợp đã lượng hóa."},
            {"Cột mới": "customer_segment", "Cách xử lý": f"One-Hot Encoding bổ sung, sinh ra {int(generated_counts.get('customer_segment', 0)):,} cột nếu có trong file encoded."},
            {"Cột mới": "booking_demand", "Cách xử lý": "Giữ nguyên numeric, không cần encoding hay chuẩn hóa thêm."},
            {"Cột mới": "availability_category", "Cách xử lý": "Ordinal Encoding: Low Availability = 0, Medium Availability = 1, High Availability = 2."},
            {"Cột mới": "availability_efficiency", "Cách xử lý": "Giữ nguyên numeric, không cần encoding hay chuẩn hóa thêm."},
            {"Cột mới": "revenue_per_available_night", "Cách xử lý": "Giữ nguyên numeric, không cần encoding hay chuẩn hóa thêm."},
        ]
        st.dataframe(pd.DataFrame(engineered_encoding_rows), use_container_width=True, hide_index=True)

        ml_shape = ml_ready_export.get("ml_shape", [])
        if isinstance(ml_shape, list) and len(ml_shape) == 2:
            st.write(f"Encoded dataframe shape: {ml_shape[0]:,} rows x {ml_shape[1]:,} columns")
        dropped_identifier_columns = ml_ready_export.get("dropped_identifier_columns", [])
        if dropped_identifier_columns:
            st.write(
                f"Dropped identifier columns at encoding step: {_format_column_list(dropped_identifier_columns)}"
            )
        st.write(
            f"Non-numeric kept intentionally: {_format_column_list(ml_ready_export.get('non_numeric_columns', []))}"
        )
        st.write(
            "Ngoài 15 cột gốc ở bảng trên, file ML-ready còn giữ thêm các cột feature engineering. "
            "Trong đó `customer_segment` được one-hot encoding, `availability_category` được ordinal encoding, "
            "và các cột numeric còn lại được giữ nguyên để phục vụ mô hình hoặc phân tích tiếp theo."
        )

        with st.expander("One-Hot Column Explanation", expanded=False):
            st.markdown(
                "- `drop_first=True` means the one-hot export removes one baseline category, so the number of generated columns equals `original category count - 1`."
            )
            generated_one_hot_columns = ml_ready_export.get("one_hot_generated_columns", [])
            room_type_columns = [
                column for column in generated_one_hot_columns if column.startswith("room_type_")
            ]
            if generated_counts.get("room_type", 0):
                st.markdown(
                    f"- `room_type`: the original data has 4 categories, so after one-hot encoding with `drop_first=True`, **{int(generated_counts.get('room_type', 0))} columns** remain."
                )
                st.write("Generated columns from `room_type`:")
                st.write(_format_column_list(room_type_columns))
            customer_segment_columns = [
                column for column in generated_one_hot_columns if column.startswith("customer_segment_")
            ]
            if generated_counts.get("customer_segment", 0):
                st.markdown(
                    f"- `customer_segment`: after one-hot encoding with `drop_first=True`, **{int(generated_counts.get('customer_segment', 0))} columns** remain."
                )
                st.write("Generated columns from `customer_segment`:")
                st.write(_format_column_list(customer_segment_columns))
        with st.expander("Encoding Code Example", expanded=False):
            st.code(
                """
encoded_df = df.copy()
encoded_df["host_identity_verified"] = encoded_df["host_identity_verified"].map(
    {"unconfirmed": 0, "verified": 1}
)
encoded_df["instant_bookable"] = encoded_df["instant_bookable"].map({"false": 0, "true": 1})
encoded_df["neighbourhood_group"] = encoded_df["neighbourhood_group"].astype("category").cat.codes
encoded_df["neighbourhood"] = encoded_df["neighbourhood"].astype("category").cat.codes
encoded_df["cancellation_policy"] = encoded_df["cancellation_policy"].astype("category").cat.codes

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

# Keep engineered numeric columns as-is:
# listing_year, property_age, estimated_revenue, occupancy_rate,
# booking_flexibility_score, booking_demand,
# availability_efficiency, revenue_per_available_night

encoded_df = pd.get_dummies(
    encoded_df,
    columns=["room_type", "customer_segment"],
    drop_first=True,
    dtype="int64",
)

# Keep numeric columns as-is:
# price, minimum_nights, number_of_reviews, review_rate_number,
# calculated_host_listings_count, availability_365
                """.strip(),
                language="python",
            )

    st.markdown("---")
    st.write(f"Dropped columns summary: {_format_column_list(dropped_columns)}")
    st.write(f"Scaled columns summary: {_format_column_list(scaled_columns)}")


def render_page(_frame: pd.DataFrame, page_mode: str = "eda") -> None:
    processing_report = st.session_state.get("processing_report")
    before_frame = st.session_state.get("preprocessing_before_df")

    if page_mode == "preprocessing":
        st.title(t("prep.title"))
        st.caption(t("prep.caption"))
        tab_data, tab_steps = st.tabs(["Data", "Preprocessing Steps"])
        with tab_data:
            render_processing_panel(_frame)
        with tab_steps:
            if processing_report is None or before_frame is None:
                st.info("The runtime summary is not available in the current session. The step-by-step description is still shown below.")
                render_processing_steps_panel()
            else:
                _render_pipeline_summary(processing_report, before_frame)
                st.markdown("---")
                render_processing_steps_panel()
        return

    eda_frame = _prepare_processed_eda_frame(_frame)
    if not isinstance(eda_frame, pd.DataFrame) or eda_frame.empty:
        st.info("Please upload a CSV file in the Input Data page first to see the EDA insights.")
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
            availability_mix["share_pct"] = availability_mix["share"] * 100
            pie_chart = px.pie(
                availability_mix,
                names="availability_category",
                values="share_pct",
                hole=0.55,
                color="availability_category",
                category_orders={"availability_category": AVAILABILITY_CATEGORY_ORDER},
                color_discrete_map=AVAILABILITY_CATEGORY_COLOR_MAP,
            )
            dominant_category = availability_mix.sort_values("share_pct", ascending=False).iloc[0]
            high_availability_pct = float(
                availability_mix.loc[
                    availability_mix["availability_category"] == "High Availability",
                    "share_pct",
                ].iloc[0]
            )
            _render_chart(
                "1. Availability Category Distribution (Supply)",
                pie_chart,
                (
                    f'Insight: {dominant_category["share_pct"]:.1f}% listings thuộc nhóm '
                    f'"{dominant_category["availability_category"]}". Điều này cho thấy thị trường đang vận hành khá năng động, '
                    f"với phần lớn căn hộ không còn trống nhiều ngày trong năm. Chỉ khoảng {high_availability_pct:.1f}% "
                    "thuộc nhóm High Availability, gợi ý rằng nhóm này có thể đang gặp khó khăn hơn trong việc thu hút khách."
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
                    "2. Booking Demand by Borough (Demand)",
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
                        name="Trend line",
                        line=dict(color="#223247", width=3),
                    )
                _render_chart(
                    "3. Price vs Booking Demand (Price vs Demand)",
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
                    "4. Availability Efficiency Heatmap (Efficiency)",
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
