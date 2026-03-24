from __future__ import annotations

import random
import re
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from core.config import DATASET_PATH, SAMPLE_SOURCE_LABEL

ML_IDENTIFIER_COLUMNS = {"id"}


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [
        re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", str(column).strip().lower())).strip("_")
        for column in normalized.columns
    ]

    aliases = {
        "neighbourhood_group": ["neighborhood_group"],
        "neighbourhood": ["neighborhood"],
        "number_of_reviews": ["reviews", "review_count"],
        "availability_365": ["availability", "available_days"],
    }
    rename_map: dict[str, str] = {}
    for canonical, options in aliases.items():
        if canonical in normalized.columns:
            continue
        for option in options:
            if option in normalized.columns:
                rename_map[option] = canonical
                break

    return normalized.rename(columns=rename_map)


def coerce_currency(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace({"": None, "nan": None, "None": None})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def build_sample_dataset(rows: int = 240) -> pd.DataFrame:
    rng = random.Random(42)
    area_profiles = {
        "Manhattan": {
            "neighbourhoods": ["Midtown", "Chelsea", "Harlem", "SoHo"],
            "base_price": 220,
        },
        "Brooklyn": {
            "neighbourhoods": ["Williamsburg", "Bushwick", "Park Slope", "DUMBO"],
            "base_price": 165,
        },
        "Queens": {
            "neighbourhoods": ["Astoria", "Flushing", "Long Island City"],
            "base_price": 145,
        },
        "Bronx": {
            "neighbourhoods": ["Mott Haven", "Riverdale", "Fordham"],
            "base_price": 110,
        },
        "Staten Island": {
            "neighbourhoods": ["St. George", "Great Kills", "Tottenville"],
            "base_price": 125,
        },
    }
    room_profiles = {
        "Entire home/apt": {"multiplier": 1.35, "review_bias": 1.05},
        "Private room": {"multiplier": 0.82, "review_bias": 0.92},
        "Shared room": {"multiplier": 0.58, "review_bias": 0.78},
        "Hotel room": {"multiplier": 1.12, "review_bias": 1.18},
    }

    records: list[dict[str, object]] = []
    for listing_id in range(1, rows + 1):
        neighbourhood_group = rng.choice(list(area_profiles))
        neighbourhood = rng.choice(area_profiles[neighbourhood_group]["neighbourhoods"])
        room_type = rng.choices(
            population=list(room_profiles),
            weights=[0.48, 0.31, 0.09, 0.12],
            k=1,
        )[0]
        base_price = area_profiles[neighbourhood_group]["base_price"]
        room_factor = room_profiles[room_type]["multiplier"]
        volatility = rng.uniform(0.75, 1.35)
        price = round(base_price * room_factor * volatility, 2)
        review_bias = room_profiles[room_type]["review_bias"]
        reviews = max(0, int(rng.gauss(52 * review_bias, 18)))
        availability = max(0, min(365, int(rng.gauss(190, 75))))
        records.append(
            {
                "id": listing_id,
                "name": f"{neighbourhood} stay {listing_id}",
                "host_name": f"Host {rng.randint(10, 99)}",
                "neighbourhood_group": neighbourhood_group,
                "neighbourhood": neighbourhood,
                "room_type": room_type,
                "price": f"${price:,.2f}",
                "number_of_reviews": reviews,
                "reviews_per_month": round(max(0.1, rng.gauss(2.4, 1.1)), 2),
                "review_rate_number": rng.randint(3, 5),
                "minimum_nights": max(1, int(rng.gauss(4, 2))),
                "availability_365": availability,
                "last_review": date(2025, 1, 1) + timedelta(days=rng.randint(0, 380)),
            }
        )

    return pd.DataFrame(records)


def build_missing_table(frame: pd.DataFrame) -> pd.DataFrame:
    missing = frame.isna().sum().reset_index()
    missing.columns = ["column", "missing_values"]
    missing["missing_pct"] = (missing["missing_values"] / len(frame) * 100).round(2) if len(frame) else 0.0
    return missing.sort_values(["missing_values", "column"], ascending=[False, True])


def preprocess_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    cleaned = normalize_columns(frame)
    missing_before = build_missing_table(cleaned)
    rows_before = len(cleaned)

    duplicates_removed = int(cleaned.duplicated().sum())
    if duplicates_removed:
        cleaned = cleaned.drop_duplicates()

    numeric_columns = [
        "price",
        "service_fee",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "review_rate_number",
        "calculated_host_listings_count",
        "availability_365",
    ]
    for column in numeric_columns:
        if column not in cleaned.columns:
            continue
        if column in {"price", "service_fee"}:
            cleaned[column] = coerce_currency(cleaned[column])
        else:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if "last_review" in cleaned.columns:
        cleaned["last_review"] = pd.to_datetime(cleaned["last_review"], errors="coerce")

    for column, fill_value in (
        ("neighbourhood_group", "Unknown"),
        ("neighbourhood", "Unknown"),
        ("room_type", "Unknown"),
    ):
        if column not in cleaned.columns:
            continue
        cleaned[column] = cleaned[column].fillna(fill_value)

    for column in ("number_of_reviews", "reviews_per_month", "availability_365"):
        if column not in cleaned.columns:
            continue
        fill_value = 0 if column != "availability_365" else cleaned[column].median()
        fill_value = 0 if pd.isna(fill_value) else fill_value
        cleaned[column] = cleaned[column].fillna(fill_value)

    removed_invalid_price = 0
    if "price" in cleaned.columns:
        valid_price_mask = cleaned["price"].notna() & (cleaned["price"] > 0)
        removed_invalid_price = int((~valid_price_mask).sum())
        cleaned = cleaned.loc[valid_price_mask].copy()

    cleaned = cleaned.reset_index(drop=True)
    missing_after = build_missing_table(cleaned)

    report = {
        "rows_before": rows_before,
        "rows_after": len(cleaned),
        "duplicates_removed": duplicates_removed,
        "invalid_price_removed": removed_invalid_price,
        "missing_before": missing_before,
        "missing_after": missing_after,
    }
    return cleaned, report


def dataset_cache_key() -> str:
    if DATASET_PATH.exists():
        return str(DATASET_PATH.stat().st_mtime_ns)
    return "sample"


@st.cache_data(show_spinner=False)
def load_airbnb_bundle(_cache_key: str) -> tuple[pd.DataFrame, pd.DataFrame, str, dict[str, object]]:
    cleaned_path = DATASET_PATH.parent / "Airbnb_Data_cleaned.csv"
    if DATASET_PATH.exists():
        raw_data = pd.read_csv(DATASET_PATH)
        source_label = str(DATASET_PATH)
        cleaned_data, report = preprocess_data(raw_data)
        return raw_data, cleaned_data, source_label, report
    if cleaned_path.exists():
        cleaned_data = pd.read_csv(cleaned_path)
        source_label = str(cleaned_path)
        report = {
            "rows_before": len(cleaned_data),
            "rows_after": len(cleaned_data),
            "duplicates_removed": 0,
            "invalid_price_removed": 0,
            "missing_before": build_missing_table(cleaned_data),
            "missing_after": build_missing_table(cleaned_data),
        }
        return cleaned_data.copy(), cleaned_data, source_label, report
    else:
        raw_data = build_sample_dataset()
        source_label = SAMPLE_SOURCE_LABEL

    cleaned_data, report = preprocess_data(raw_data)
    return raw_data, cleaned_data, source_label, report


def _coerce_numeric_series(column_name: str, series: pd.Series) -> pd.Series | None:
    if column_name in {"id", "host id", "host_id"} or pd.api.types.is_bool_dtype(series):
        return None

    if pd.api.types.is_datetime64_any_dtype(series):
        if column_name in {"construction_year", "Construction year"}:
            return series.dt.year.astype("float64")
        return None

    if pd.api.types.is_numeric_dtype(series):
        numeric_series = pd.to_numeric(series, errors="coerce")
    else:
        normalized = (
            series.astype("string")
            .str.strip()
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
        )
        numeric_series = pd.to_numeric(normalized, errors="coerce")

    non_null_count = int(series.notna().sum())
    if non_null_count == 0:
        return None

    if numeric_series.notna().sum() / non_null_count < 0.8:
        return None

    return numeric_series


def build_numeric_profile_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns: dict[str, pd.Series] = {}
    for column in frame.columns:
        numeric_series = _coerce_numeric_series(column, frame[column])
        if numeric_series is not None and numeric_series.notna().any():
            numeric_columns[column] = numeric_series
    return pd.DataFrame(numeric_columns, index=frame.index)


def _is_binary_numeric_series(series: pd.Series) -> bool:
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty:
        return False
    return set(numeric_series.unique()).issubset({0, 1})


def build_ml_ready_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    ml_ready = normalize_columns(frame).copy()
    dropped_identifier_columns = [column for column in sorted(ML_IDENTIFIER_COLUMNS) if column in ml_ready.columns]
    if dropped_identifier_columns:
        ml_ready = ml_ready.drop(columns=dropped_identifier_columns)
    kept_identifier_columns = [column for column in ["host_id"] if column in ml_ready.columns]

    def _normalize_text(series: pd.Series) -> pd.Series:
        return (
            series.astype("string")
            .str.strip()
            .str.lower()
            .replace(
                {
                    "": pd.NA,
                    "nan": pd.NA,
                    "none": pd.NA,
                    "<na>": pd.NA,
                }
            )
        )

    label_encoded_columns: list[str] = []
    one_hot_encoded_columns: list[str] = []
    datetime_engineered_columns: list[str] = []

    if "host_id" in ml_ready.columns:
        ml_ready["host_id"] = _normalize_text(ml_ready["host_id"]).fillna("unknown_host")

    if "host_identity_verified" in ml_ready.columns:
        verified = _normalize_text(ml_ready["host_identity_verified"]).fillna("unconfirmed")
        ml_ready["host_identity_verified"] = verified.map({"unconfirmed": 0, "verified": 1}).fillna(0).astype("int64")
        label_encoded_columns.append("host_identity_verified")

    if "instant_bookable" in ml_ready.columns:
        instant_series = ml_ready["instant_bookable"]
        if pd.api.types.is_bool_dtype(instant_series):
            ml_ready["instant_bookable"] = instant_series.astype("int64")
        else:
            normalized_instant = _normalize_text(instant_series).replace({"yes": "true", "no": "false"}).fillna("false")
            ml_ready["instant_bookable"] = normalized_instant.map({"false": 0, "true": 1}).fillna(0).astype("int64")
        label_encoded_columns.append("instant_bookable")

    if "cancellation_policy" in ml_ready.columns:
        cancellation = _normalize_text(ml_ready["cancellation_policy"]).fillna("strict")
        ml_ready["cancellation_policy"] = cancellation.map({"strict": 0, "moderate": 1, "flexible": 2}).fillna(0).astype("int64")
        label_encoded_columns.append("cancellation_policy")

    if "customer_segment" in ml_ready.columns:
        customer_segment = _normalize_text(ml_ready["customer_segment"]).fillna("long stay (>7 nights)")
        ml_ready["customer_segment"] = customer_segment.map(
            {
                "short stay (1-3 nights)": 0,
                "business/leisure (4-7 nights)": 1,
                "long stay (>7 nights)": 2,
            }
        ).fillna(2).astype("int64")
        label_encoded_columns.append("customer_segment")

    if "construction_year" in ml_ready.columns:
        construction_year = pd.to_datetime(ml_ready["construction_year"], errors="coerce")
        ml_ready["construction_year"] = construction_year.dt.year.astype("float64")

    if "last_review" in ml_ready.columns:
        last_review = pd.to_datetime(ml_ready["last_review"], errors="coerce")
        reference_date = pd.Timestamp("2022-12-31")
        ml_ready["days_since_last_review"] = (reference_date - last_review).dt.days.astype("float64")
        ml_ready = ml_ready.drop(columns=["last_review"])
        datetime_engineered_columns = ["days_since_last_review"]

    for column in ("price", "minimum_nights", "number_of_reviews", "review_rate_number", "calculated_host_listings_count", "availability_365", "listing_year", "property_age", "estimated_revenue", "occupancy_rate", "booking_flexibility_score"):
        if column in ml_ready.columns:
            ml_ready[column] = pd.to_numeric(ml_ready[column], errors="coerce").astype("float64")

    one_hot_targets = [
        column
        for column in ["neighbourhood_group", "neighbourhood", "room_type"]
        if column in ml_ready.columns
    ]
    for column in one_hot_targets:
        ml_ready[column] = _normalize_text(ml_ready[column]).fillna("unknown")

    if one_hot_targets:
        ml_ready = pd.get_dummies(
            ml_ready,
            columns=one_hot_targets,
            drop_first=True,
            prefix_sep="__",
            dtype="int64",
        )
        one_hot_encoded_columns = one_hot_targets

    passthrough_numeric_columns = ml_ready.select_dtypes(include="number").columns.tolist()
    metadata = {
        "dropped_identifier_columns": dropped_identifier_columns,
        "kept_identifier_columns": kept_identifier_columns,
        "datetime_engineered_columns": datetime_engineered_columns,
        "label_encoded_columns": label_encoded_columns,
        "one_hot_encoded_columns": one_hot_encoded_columns,
        "passthrough_numeric_columns": passthrough_numeric_columns,
        "ml_shape": list(ml_ready.shape),
        "non_numeric_columns": ml_ready.select_dtypes(exclude="number").columns.tolist(),
    }
    return ml_ready, metadata


def _prepare_null_comparison(before_frame: pd.DataFrame, after_frame: pd.DataFrame) -> pd.DataFrame:
    before_health = build_missing_table(before_frame)[["column", "missing_values"]].rename(
        columns={"column": "column name", "missing_values": "Before Processing"}
    )
    after_health = build_missing_table(after_frame)[["column", "missing_values"]].rename(
        columns={"column": "column name", "missing_values": "After Processing"}
    )
    comparison = before_health.merge(after_health, on="column name", how="outer").fillna(0)
    comparison["max_null"] = comparison[["Before Processing", "After Processing"]].max(axis=1)
    comparison = comparison.sort_values(["max_null", "column name"], ascending=[False, True])
    long_frame = comparison.drop(columns="max_null").melt(
        id_vars="column name",
        var_name="stage",
        value_name="null count",
    )
    long_frame["column name"] = pd.Categorical(
        long_frame["column name"],
        categories=comparison["column name"].tolist()[::-1],
        ordered=True,
    )
    return long_frame


def _prepare_boxplot_comparison(before_frame: pd.DataFrame, after_frame: pd.DataFrame) -> pd.DataFrame:
    stages: list[pd.DataFrame] = []
    for stage_name, frame in (("Before Processing", before_frame), ("After Processing", after_frame)):
        numeric_profile = build_numeric_profile_frame(frame)
        if numeric_profile.empty:
            continue
        melted = numeric_profile.melt(var_name="column name", value_name="value").dropna()
        if melted.empty:
            continue
        stages.append(melted.assign(stage=stage_name))
    if not stages:
        return pd.DataFrame(columns=["column name", "value", "stage"])
    return pd.concat(stages, ignore_index=True)


def build_dtype_distribution_table(frame: pd.DataFrame) -> pd.DataFrame:
    dtype_counts = frame.dtypes.astype(str).value_counts().reset_index()
    dtype_counts.columns = ["dtype", "column_count"]
    return dtype_counts.sort_values(["column_count", "dtype"], ascending=[False, True]).reset_index(drop=True)


def _find_matching_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    candidate_lookup = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        matched = candidate_lookup.get(candidate.strip().lower())
        if matched is not None:
            return matched
    return None


def build_quick_stats_table(frame: pd.DataFrame) -> pd.DataFrame:
    stat_rows: list[dict[str, object]] = []
    quick_stat_columns = {
        "price": ["price"],
        "minimum_nights": ["minimum_nights", "minimum nights"],
        "availability_365": ["availability_365", "availability 365"],
        "number_of_reviews": ["number_of_reviews", "number of reviews"],
    }

    for label, candidates in quick_stat_columns.items():
        column_name = _find_matching_column(frame, candidates)
        if column_name is None:
            continue
        numeric_series = _coerce_numeric_series(column_name, frame[column_name])
        if numeric_series is None:
            continue
        numeric_series = numeric_series.dropna()
        if numeric_series.empty:
            continue
        stat_rows.append(
            {
                "Column": label,
                "Min": round(float(numeric_series.min()), 2),
                "Max": round(float(numeric_series.max()), 2),
                "Mean": round(float(numeric_series.mean()), 2),
            }
        )

    return pd.DataFrame(stat_rows)
