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
CANCELLATION_SEQUENCE = ["flexible", "moderate", "strict", "unknown"]
BOROUGH_DISPLAY_ORDER = ["brooklyn", "manhattan", "queens", "bronx", "staten island"]
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

    for column in ("neighbourhood_group", "neighbourhood", "room_type", "cancellation_policy", "customer_segment"):
        if column in prepared.columns:
            prepared[column] = prepared[column].astype("string").str.strip()

    if "instant_bookable" in prepared.columns:
        if pd.api.types.is_bool_dtype(prepared["instant_bookable"]):
            prepared["instant_bookable"] = prepared["instant_bookable"].map({True: "TRUE", False: "FALSE"})
        else:
            prepared["instant_bookable"] = prepared["instant_bookable"].astype("string").str.strip().str.upper()

    if "construction_year" in prepared.columns:
        if pd.api.types.is_datetime64_any_dtype(prepared["construction_year"]):
            prepared["construction_year"] = prepared["construction_year"].dt.year.astype("float64")
        else:
            prepared["construction_year"] = pd.to_numeric(
                prepared["construction_year"].astype("string").str.extract(r"(\d{4})", expand=False),
                errors="coerce",
            )

    for column in (
        "price",
        "minimum_nights",
        "number_of_reviews",
        "review_rate_number",
        "calculated_host_listings_count",
        "availability_365",
        "listing_year",
        "property_age",
        "estimated_revenue",
        "occupancy_rate",
        "booking_flexibility_score",
        "service_fee",
        "reviews_per_month",
        "price_to_neighborhood_ratio",
        "popularity_index",
        "booking_friction",
    ):
        if column not in prepared.columns:
            continue
        if column in {"price", "service_fee"}:
            prepared[column] = _coerce_numeric(coerce_currency(prepared[column]))
        else:
            prepared[column] = _coerce_numeric(prepared[column])

    if "availability_365" in prepared.columns:
        prepared["availability_365"] = prepared["availability_365"].clip(lower=0, upper=365)

    if "listing_year" not in prepared.columns and "last_review" in prepared.columns:
        last_review = pd.to_datetime(prepared["last_review"], errors="coerce")
        prepared["listing_year"] = pd.to_numeric(last_review.dt.year, errors="coerce")

    if "customer_segment" not in prepared.columns and "minimum_nights" in prepared.columns:
        prepared["customer_segment"] = pd.cut(
            prepared["minimum_nights"],
            bins=[0, 3, 7, float("inf")],
            labels=CUSTOMER_SEGMENT_ORDER,
            include_lowest=True,
            right=True,
        ).astype("string")

    if "occupancy_rate" not in prepared.columns and "availability_365" in prepared.columns:
        prepared["occupancy_rate"] = ((365 - prepared["availability_365"]) / 365 * 100).round(2)
    elif "occupancy_rate" in prepared.columns and prepared["occupancy_rate"].dropna().max() <= 1.5:
        prepared["occupancy_rate"] = (prepared["occupancy_rate"] * 100).round(2)

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


def _render_chart(title: str, fig: px.scatter, conclusion: str) -> None:
    st.subheader(title)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(conclusion)


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
        st.write(f"Reference year for property_age: {feature_engineering.get('reference_year', 2022)}")
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
        st.write(f"Alternative scalers noted: {_format_column_list(scaling.get('alternative_scalers', []))}")
        for note in scaling.get("notes", []):
            st.write(f"- {note}")

    ml_ready_export = step_metrics.get("ml_ready_export", {})
    with st.expander("10. Encoding for Machine Learning", expanded=True):
        st.write("Status: Completed")
        generated_counts = ml_ready_export.get("one_hot_generated_counts", {})
        encoding_plan_rows = [
            {"No.": 1, "Original Column": "host_id", "Data Type": "Identifier", "Encoding Method": "No encoding. Keep the raw host ID.", "Output Columns": "1 host_id column"},
            {"No.": 2, "Original Column": "host_identity_verified", "Data Type": "Binary", "Encoding Method": "Label encode: unconfirmed -> 0, verified -> 1.", "Output Columns": "1 column"},
            {"No.": 3, "Original Column": "neighbourhood_group", "Data Type": "Categorical", "Encoding Method": "Label encode.", "Output Columns": "1 column"},
            {"No.": 4, "Original Column": "neighbourhood", "Data Type": "Categorical", "Encoding Method": "Label encode.", "Output Columns": "1 column"},
            {"No.": 5, "Original Column": "instant_bookable", "Data Type": "Binary", "Encoding Method": "Label encode: false -> 0, true -> 1.", "Output Columns": "1 column"},
            {"No.": 6, "Original Column": "cancellation_policy", "Data Type": "Categorical", "Encoding Method": "Label encode.", "Output Columns": "1 column"},
            {"No.": 7, "Original Column": "room_type", "Data Type": "Categorical", "Encoding Method": "One-hot encode.", "Output Columns": f"{int(generated_counts.get('room_type', 0)):,} columns"},
            {"No.": 8, "Original Column": "construction_year", "Data Type": "Numeric", "Encoding Method": "No encoding. Keep numeric.", "Output Columns": "1 column"},
            {"No.": 9, "Original Column": "price", "Data Type": "Numeric", "Encoding Method": "No encoding. Keep numeric.", "Output Columns": "1 column"},
            {"No.": 10, "Original Column": "minimum_nights", "Data Type": "Numeric", "Encoding Method": "No encoding. Keep numeric.", "Output Columns": "1 column"},
            {"No.": 11, "Original Column": "number_of_reviews", "Data Type": "Numeric", "Encoding Method": "No encoding. Keep numeric.", "Output Columns": "1 column"},
            {"No.": 12, "Original Column": "last_review", "Data Type": "Datetime", "Encoding Method": "Convert to days_since_last_review.", "Output Columns": "1 derived column"},
            {"No.": 13, "Original Column": "review_rate_number", "Data Type": "Numeric", "Encoding Method": "No encoding. Keep numeric.", "Output Columns": "1 column"},
            {"No.": 14, "Original Column": "calculated_host_listings_count", "Data Type": "Numeric", "Encoding Method": "No encoding. Keep numeric.", "Output Columns": "1 column"},
            {"No.": 15, "Original Column": "availability_365", "Data Type": "Numeric", "Encoding Method": "No encoding. Keep numeric.", "Output Columns": "1 column"},
            {"No.": 16, "Original Column": "listing_year", "Data Type": "Numeric", "Encoding Method": "No encoding. Keep numeric.", "Output Columns": "1 column"},
            {"No.": 17, "Original Column": "property_age", "Data Type": "Numeric", "Encoding Method": "No encoding. Keep numeric.", "Output Columns": "1 column"},
            {"No.": 18, "Original Column": "estimated_revenue", "Data Type": "Numeric", "Encoding Method": "No encoding. Keep numeric.", "Output Columns": "1 column"},
            {"No.": 19, "Original Column": "occupancy_rate", "Data Type": "Numeric", "Encoding Method": "No encoding. Keep numeric.", "Output Columns": "1 column"},
            {"No.": 20, "Original Column": "booking_flexibility_score", "Data Type": "Numeric", "Encoding Method": "No encoding. Keep numeric.", "Output Columns": "1 column"},
            {"No.": 21, "Original Column": "customer_segment", "Data Type": "Categorical", "Encoding Method": "One-hot encode in the current pipeline.", "Output Columns": f"{int(generated_counts.get('customer_segment', 0)):,} columns"},
        ]
        st.dataframe(pd.DataFrame(encoding_plan_rows), use_container_width=True, hide_index=True)

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

        with st.expander("One-Hot Column Explanation", expanded=False):
            st.markdown(
                "- `drop_first=True` means each one-hot feature drops one baseline category, so the number of generated columns equals `original category count - 1`."
            )
            generated_one_hot_columns = ml_ready_export.get("one_hot_generated_columns", [])
            room_type_columns = [
                column for column in generated_one_hot_columns if column.startswith("room_type_")
            ]
            customer_segment_columns = [
                column for column in generated_one_hot_columns if column.startswith("customer_segment_")
            ]
            if generated_counts.get("room_type", 0):
                st.markdown(
                    f"- `room_type`: the original data has 4 categories, so after one-hot encoding with `drop_first=True`, **{int(generated_counts.get('room_type', 0))} columns** remain."
                )
                st.write("Generated columns from `room_type`:")
                st.write(_format_column_list(room_type_columns))
            if generated_counts.get("customer_segment", 0):
                st.markdown(
                    f"- `customer_segment`: the original data has 3 categories, so after one-hot encoding with `drop_first=True`, **{int(generated_counts.get('customer_segment', 0))} columns** remain."
                )
                st.write("Generated columns from `customer_segment`:")
                st.write(_format_column_list(customer_segment_columns))
        with st.expander("Encoding Code Example", expanded=False):
            st.code(
                """
df["host_identity_verified"] = df["host_identity_verified"].map({"unconfirmed": 0, "verified": 1})
df["neighbourhood_group"] = df["neighbourhood_group"].astype("category").cat.codes
df["neighbourhood"] = df["neighbourhood"].astype("category").cat.codes
df["instant_bookable"] = df["instant_bookable"].map({"false": 0, "true": 1})
df["cancellation_policy"] = df["cancellation_policy"].map({"strict": 0, "moderate": 1, "flexible": 2})
df["days_since_last_review"] = (pd.Timestamp("2022-12-31") - pd.to_datetime(df["last_review"])).dt.days
df = pd.get_dummies(df, columns=["room_type", "customer_segment"], drop_first=True)
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

    st.title(t("eda.title"))
    st.caption(t("eda.caption"))

    eda_frame = _prepare_processed_eda_frame(_frame)
    if not isinstance(eda_frame, pd.DataFrame) or eda_frame.empty:
        st.info("Please upload a CSV file in the Input Data page first to see the EDA insights.")
        return

    with st.container():
        viz_frame = eda_frame.copy()
        if "listing_year" in viz_frame.columns:
            viz_frame["listing_year"] = pd.to_numeric(viz_frame["listing_year"], errors="coerce")
        if "neighbourhood_group" in viz_frame.columns:
            viz_frame["neighbourhood_group"] = (
                viz_frame["neighbourhood_group"].astype("string").str.strip().str.lower()
            )
        if "customer_segment" in viz_frame.columns:
            viz_frame["customer_segment"] = (
                viz_frame["customer_segment"].astype("string").str.strip().str.lower()
            )
        if "room_type" in viz_frame.columns:
            room_type_series = viz_frame["room_type"].astype("string").str.strip()
            viz_frame["room_type"] = room_type_series.str.lower().map(ROOM_TYPE_CANONICAL_MAP).fillna(room_type_series)

        yearly_frame = viz_frame.copy()
        if {"listing_year", "neighbourhood_group"}.issubset(yearly_frame.columns):
            yearly_frame = yearly_frame.loc[
                yearly_frame["listing_year"].between(2012, 2022)
                & yearly_frame["neighbourhood_group"].isin(BOROUGH_DISPLAY_ORDER)
            ].copy()

        if {"price", "listing_year", "neighbourhood_group"}.issubset(yearly_frame.columns) and not yearly_frame.empty:
            price_trend = (
                yearly_frame.groupby(["listing_year", "neighbourhood_group"], as_index=False)["price"]
                .mean()
                .sort_values(["listing_year", "neighbourhood_group"])
            )
            price_trend["listing_year"] = price_trend["listing_year"].round().astype(int)
            price_chart = px.line(
                price_trend,
                x="listing_year",
                y="price",
                color="neighbourhood_group",
                markers=True,
                color_discrete_map=BOROUGH_COLOR_MAP,
                category_orders={"neighbourhood_group": BOROUGH_DISPLAY_ORDER},
            )
            _render_chart(
                "1. Average Price Trend by Neighbourhood Group (2012-2022)",
                price_chart,
                "Insight: Demand for stays across boroughs has increased, and average prices have risen over the last decade as well.",
            )

        if {"occupancy_rate", "listing_year", "neighbourhood_group"}.issubset(yearly_frame.columns) and not yearly_frame.empty:
            occupancy_trend = (
                yearly_frame.groupby(["listing_year", "neighbourhood_group"], as_index=False)["occupancy_rate"]
                .mean()
                .sort_values(["listing_year", "neighbourhood_group"])
            )
            occupancy_trend["listing_year"] = occupancy_trend["listing_year"].round().astype(int)
            occupancy_chart = px.line(
                occupancy_trend,
                x="listing_year",
                y="occupancy_rate",
                color="neighbourhood_group",
                markers=True,
                color_discrete_map=BOROUGH_COLOR_MAP,
                category_orders={"neighbourhood_group": BOROUGH_DISPLAY_ORDER},
            )
            _render_chart(
                "2. Occupancy Rate Trend by Neighbourhood Group (2012-2022)",
                occupancy_chart,
                "Insight: Occupancy levels have generally strengthened across boroughs, pointing to sustained demand over time.",
            )

        if {"listing_year", "estimated_revenue"}.issubset(yearly_frame.columns) and not yearly_frame.empty:
            revenue_year = (
                yearly_frame.groupby("listing_year", as_index=False)["estimated_revenue"]
                .mean()
                .sort_values("listing_year")
            )
            revenue_year["listing_year"] = revenue_year["listing_year"].round().astype(int)
            revenue_year_chart = px.bar(
                revenue_year,
                x="listing_year",
                y="estimated_revenue",
                color_discrete_sequence=[CHART_COLORS[1]],
            )
            _render_chart(
                "3. Average Estimated Revenue by Year (2012-2022)",
                revenue_year_chart,
                "Insight: Average estimated revenue indicates clear Airbnb market growth in New York City during 2012-2022.",
            )

        if {"listing_year", "estimated_revenue", "neighbourhood_group"}.issubset(yearly_frame.columns) and not yearly_frame.empty:
            revenue_area_year = (
                yearly_frame.groupby(["listing_year", "neighbourhood_group"], as_index=False)["estimated_revenue"]
                .mean()
                .sort_values(["listing_year", "neighbourhood_group"])
            )
            revenue_area_year["listing_year"] = revenue_area_year["listing_year"].round().astype(int)
            revenue_area_chart = px.bar(
                revenue_area_year,
                x="listing_year",
                y="estimated_revenue",
                color="neighbourhood_group",
                barmode="group",
                color_discrete_map=BOROUGH_COLOR_MAP,
                category_orders={"neighbourhood_group": BOROUGH_DISPLAY_ORDER},
            )
            _render_chart(
                "4. Average Estimated Revenue by Borough and Year",
                revenue_area_chart,
                "Insight: Average revenue increased across multiple boroughs, reflecting citywide expansion of the Airbnb market.",
            )

        rendered_segment_room_chart = False
        if {"customer_segment", "estimated_revenue"}.issubset(viz_frame.columns):
            segment_frame = viz_frame.loc[
                viz_frame["customer_segment"].isin(CUSTOMER_SEGMENT_ORDER)
            ].copy()
            if not segment_frame.empty:
                revenue_segment = (
                    segment_frame.groupby("customer_segment", as_index=False)["estimated_revenue"]
                    .mean()
                )
                revenue_segment["customer_segment"] = pd.Categorical(
                    revenue_segment["customer_segment"],
                    categories=CUSTOMER_SEGMENT_ORDER,
                    ordered=True,
                )
                revenue_segment = revenue_segment.sort_values("customer_segment")
                revenue_segment["customer_segment_label"] = revenue_segment["customer_segment"].map(translate_customer_segment)
                segment_color_map = {
                    translate_customer_segment(CUSTOMER_SEGMENT_ORDER[0]): "#c95c36",
                    translate_customer_segment(CUSTOMER_SEGMENT_ORDER[1]): "#d8a65d",
                    translate_customer_segment(CUSTOMER_SEGMENT_ORDER[2]): "#1f3c5b",
                }
                segment_chart = px.bar(
                    revenue_segment,
                    x="customer_segment_label",
                    y="estimated_revenue",
                    color="customer_segment_label",
                    labels={
                        "customer_segment_label": t("label.customer_segment"),
                        "estimated_revenue": t("label.estimated_revenue"),
                    },
                    category_orders={
                        "customer_segment_label": [translate_customer_segment(segment) for segment in CUSTOMER_SEGMENT_ORDER],
                    },
                    color_discrete_map=segment_color_map,
                )
                _render_chart(
                    f"5. {t('eda.chart.revenue_by_segment')}",
                    segment_chart,
                    t("eda.chart.revenue_by_segment.insight"),
                )

                if "room_type" in segment_frame.columns and segment_frame["room_type"].isin(ROOM_SEQUENCE).any():
                    segment_frame = segment_frame.loc[segment_frame["room_type"].isin(ROOM_SEQUENCE)].copy()
                    revenue_segment = (
                        segment_frame.groupby(["customer_segment", "room_type"], as_index=False)["estimated_revenue"]
                        .mean()
                    )
                    revenue_segment["customer_segment"] = pd.Categorical(
                        revenue_segment["customer_segment"],
                        categories=CUSTOMER_SEGMENT_ORDER,
                        ordered=True,
                    )
                    revenue_segment["room_type"] = pd.Categorical(
                        revenue_segment["room_type"],
                        categories=ROOM_SEQUENCE,
                        ordered=True,
                    )
                    revenue_segment = revenue_segment.sort_values(["customer_segment", "room_type"])
                    revenue_segment["customer_segment_label"] = revenue_segment["customer_segment"].map(translate_customer_segment)
                    revenue_segment["room_type_label"] = revenue_segment["room_type"].map(translate_room_type)
                    segment_chart = px.bar(
                        revenue_segment,
                        x="customer_segment_label",
                        y="estimated_revenue",
                        color="room_type_label",
                        barmode="group",
                        labels={
                            "customer_segment_label": t("label.customer_segment"),
                            "estimated_revenue": t("label.estimated_revenue"),
                            "room_type_label": t("label.room_type"),
                        },
                        category_orders={
                            "customer_segment_label": [translate_customer_segment(segment) for segment in CUSTOMER_SEGMENT_ORDER],
                            "room_type_label": [translate_room_type(room) for room in ROOM_SEQUENCE],
                        },
                        color_discrete_map={
                            translate_room_type(room): color for room, color in ROOM_TYPE_COLOR_MAP.items()
                        },
                    )
                    _render_chart(
                        f"6. {t('eda.chart.revenue_by_segment_room')}",
                        segment_chart,
                        t("eda.chart.revenue_by_segment_room.insight"),
                    )
                    rendered_segment_room_chart = True

        correlation_columns = [
            column
            for column in ["occupancy_rate", "price", "booking_flexibility_score", "review_rate_number"]
            if column in viz_frame.columns
        ]
        if len(correlation_columns) >= 2:
            corr = viz_frame[correlation_columns].corr(numeric_only=True).round(2)
            correlation_heatmap = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale=["#f4efe8", "#d8a65d", "#c95c36", "#5d2014"],
            )
            strongest_driver = "n/a"
            if "occupancy_rate" in corr.columns:
                occupancy_corr = corr["occupancy_rate"].drop(labels=["occupancy_rate"], errors="ignore").abs().sort_values(ascending=False)
                if not occupancy_corr.empty:
                    strongest_driver = str(occupancy_corr.index[0])
            correlation_chart_number = 7 if rendered_segment_room_chart else 6
            _render_chart(
                f"{correlation_chart_number}. Correlation Between Occupancy Rate and Key Variables",
                correlation_heatmap,
                f"Insight: `{strongest_driver}` is currently the variable most strongly correlated with `occupancy_rate` in this analysis set.",
            )
