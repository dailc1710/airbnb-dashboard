from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from core.data import build_ml_ready_frame, normalize_columns

DEFAULT_INPUT_PATH = Path("data/Airbnb_Open_Data.csv")
DEFAULT_CLEANED_OUTPUT_PATH = Path("data/Airbnb_Data_cleaned.csv")
DEFAULT_SCALED_OUTPUT_PATH = Path("data/Airbnb_Data_scaled.csv")
DEFAULT_ML_OUTPUT_PATH = Path("data/Airbnb_Data_encoded.csv")
DATASET_REFERENCE_YEAR = 2022
LAST_REVIEW_REFERENCE_END = pd.Timestamp(f"{DATASET_REFERENCE_YEAR}-12-31")

DROP_COLUMNS = [
    "id",
    "name",
    "host_name",
    "lat",
    "long",
    "country",
    "country_code",
    "service_fee",
    "reviews_per_month",
    "house_rules",
    "license",
]
STRING_COLUMNS = [
    "host_id",
    "host_identity_verified",
    "neighbourhood_group",
    "neighbourhood",
    "instant_bookable",
    "cancellation_policy",
    "room_type",
]
NUMERIC_COLUMNS = [
    "price",
    "minimum_nights",
    "number_of_reviews",
    "review_rate_number",
    "calculated_host_listings_count",
    "availability_365",
]
DATETIME_COLUMNS = [
    "construction_year",
    "last_review",
]
OUTLIER_COLUMNS = [
    "price",
    "minimum_nights",
    "number_of_reviews",
    "review_rate_number",
    "calculated_host_listings_count",
    "availability_365",
]
ENGINEERED_NUMERIC_COLUMNS = [
    "listing_year",
    "property_age",
    "estimated_revenue",
    "occupancy_rate",
    "booking_flexibility_score",
    "booking_demand",
    "availability_efficiency",
    "revenue_per_available_night",
]
ENGINEERED_CATEGORICAL_COLUMNS = [
    "customer_segment",
    "availability_category",
]
ENGINEERED_COLUMNS = ENGINEERED_NUMERIC_COLUMNS + ENGINEERED_CATEGORICAL_COLUMNS
CUSTOMER_SEGMENT_LABELS = [
    "short stay (1-3 nights)",
    "business/leisure (4-7 nights)",
    "long stay (>7 nights)",
]
AVAILABILITY_CATEGORY_LABELS = [
    "Low Availability",
    "Medium Availability",
    "High Availability",
]
AVAILABILITY_CATEGORY_BINS = [-1, 150, 300, 365]
NON_SCALED_NUMERIC_COLUMNS = [
    "booking_demand",
    "availability_efficiency",
    "revenue_per_available_night",
]
EXPECTED_COLUMNS = sorted(
    set(
        DROP_COLUMNS
        + STRING_COLUMNS
        + NUMERIC_COLUMNS
        + DATETIME_COLUMNS
        + ["service_fee", "reviews_per_month"]
    )
)
NEIGHBOURHOOD_GROUP_FIXES = {
    "brookln": "brooklyn",
    "brookyn": "brooklyn",
    "manhatan": "manhattan",
    "manhatten": "manhattan",
}
BOOLEAN_TEXT_MAP = {
    "true": "true",
    "t": "true",
    "yes": "true",
    "y": "true",
    "1": "true",
    "false": "false",
    "f": "false",
    "no": "false",
    "n": "false",
    "0": "false",
}
CANCELLATION_POLICY_SCORES = {
    "flexible": 2.0,
    "moderate": 1.0,
    "strict": 0.0,
}
OUTLIER_METHODS = {
    "price": "Percentile Capping (1%, 99%)",
    "minimum_nights": "0 -> 1, then IQR Capping",
    "number_of_reviews": "IQR Capping",
    "review_rate_number": "Clip [0, 5]",
    "calculated_host_listings_count": "IQR Capping",
    "availability_365": "Clip [0, 365]",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess the Airbnb NYC dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to the raw Airbnb CSV file. Defaults to {DEFAULT_INPUT_PATH}.",
    )
    parser.add_argument(
        "--cleaned-output",
        type=Path,
        default=DEFAULT_CLEANED_OUTPUT_PATH,
        help=f"Path for the cleaned CSV output. Defaults to {DEFAULT_CLEANED_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--scaled-output",
        type=Path,
        default=DEFAULT_SCALED_OUTPUT_PATH,
        help=f"Path for the scaled CSV output. Defaults to {DEFAULT_SCALED_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--ml-output",
        type=Path,
        default=DEFAULT_ML_OUTPUT_PATH,
        help=f"Path for the encoded CSV output used for machine learning. Defaults to {DEFAULT_ML_OUTPUT_PATH}.",
    )
    return parser.parse_args()


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    completed = frame.copy()
    for column in columns:
        if column not in completed.columns:
            completed[column] = pd.NA
    return completed


def _normalize_string_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return cleaned.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "none": pd.NA,
            "<NA>": pd.NA,
            "<na>": pd.NA,
            "NaT": pd.NA,
            "nat": pd.NA,
        }
    )


def _clean_special_text_series(series: pd.Series) -> pd.Series:
    cleaned = _normalize_string_series(series)
    cleaned = (
        cleaned.str.replace(r"<[^>]+>", " ", regex=True)
        .str.replace(r"&[a-zA-Z0-9#]+;", " ", regex=True)
        .str.replace(r"[\U00010000-\U0010ffff]", " ", regex=True)
        .str.replace(r"[^\w\s/&.,'\-]", " ", regex=True)
        .str.replace("_", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return _normalize_string_series(cleaned)


def _coerce_currency(series: pd.Series) -> pd.Series:
    cleaned = (
        _normalize_string_series(series)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def _to_datetime_year(series: pd.Series) -> pd.Series:
    extracted_year = _normalize_string_series(series).str.extract(r"(\d{4})", expand=False)
    return pd.to_datetime(extracted_year, format="%Y", errors="coerce")


def _first_mode(series: pd.Series) -> object:
    mode_values = series.dropna().mode()
    if not mode_values.empty:
        return mode_values.iloc[0]
    non_null = series.dropna()
    if not non_null.empty:
        return non_null.iloc[0]
    return pd.NA


def _safe_mean(series: pd.Series) -> float:
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty:
        return float("nan")
    return float(numeric_series.mean())


def _safe_median(series: pd.Series) -> float:
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty:
        return float("nan")
    return float(numeric_series.median())


def _group_stat_series(
    frame: pd.DataFrame,
    target_col: str,
    group_cols: list[str],
    statistic: str,
) -> pd.Series:
    if target_col not in frame.columns:
        return pd.Series(index=frame.index, dtype="object")

    valid_group_cols = [column for column in group_cols if column in frame.columns]
    if not valid_group_cols:
        return pd.Series(index=frame.index, dtype="object")

    grouped = frame.groupby(valid_group_cols, dropna=False)[target_col]
    if statistic == "mode":
        return grouped.transform(_first_mode)
    if statistic == "mean":
        return grouped.transform(lambda values: pd.to_numeric(values, errors="coerce").dropna().mean())
    if statistic == "median":
        return grouped.transform(lambda values: pd.to_numeric(values, errors="coerce").dropna().median())
    raise ValueError(f"Unsupported statistic: {statistic}")


def _fill_numeric_with_group_stat(
    frame: pd.DataFrame,
    target_col: str,
    group_cols: list[str],
    statistic: str,
    fallback: float,
) -> pd.Series:
    filled = pd.to_numeric(frame[target_col], errors="coerce")
    grouped_fill = pd.to_numeric(_group_stat_series(frame, target_col, group_cols, statistic), errors="coerce")
    filled = filled.fillna(grouped_fill)
    if pd.isna(fallback):
        fallback = 0.0
    return filled.fillna(float(fallback)).astype("float64")


def _fill_object_with_group_mode(
    frame: pd.DataFrame,
    target_col: str,
    group_cols: list[str],
    fallback: object,
) -> pd.Series:
    filled = frame[target_col].copy()
    grouped_fill = _group_stat_series(frame, target_col, group_cols, "mode")
    filled = filled.fillna(grouped_fill)
    if pd.isna(fallback):
        return filled
    return filled.where(filled.notna(), fallback)


def _safe_skew(series: pd.Series) -> float | None:
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric_series) < 3 or numeric_series.nunique() <= 1:
        return None
    skewness = float(numeric_series.skew())
    if pd.isna(skewness):
        return None
    return skewness


def _percentile_cap(series: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    non_null = pd.to_numeric(series, errors="coerce").dropna()
    if non_null.empty:
        return pd.to_numeric(series, errors="coerce").astype("float64")
    lower_bound = float(non_null.quantile(lower_q))
    upper_bound = float(non_null.quantile(upper_q))
    return pd.to_numeric(series, errors="coerce").clip(lower=lower_bound, upper=upper_bound).astype("float64")


def _iqr_cap(series: pd.Series) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors="coerce").astype("float64")
    non_null = numeric_series.dropna()
    if non_null.empty:
        return numeric_series
    q1 = float(non_null.quantile(0.25))
    q3 = float(non_null.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0:
        return numeric_series
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return numeric_series.clip(lower=lower_bound, upper=upper_bound)


def _build_scaled_dataframe(df_cleaned: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df_scaled = df_cleaned.copy()
    scaled_columns = [
        column
        for column in df_scaled.select_dtypes(include="number").columns.tolist()
        if column not in NON_SCALED_NUMERIC_COLUMNS
    ]
    if not scaled_columns or df_scaled.empty:
        return df_scaled, scaled_columns

    scaler = MinMaxScaler()
    df_scaled[scaled_columns] = scaler.fit_transform(df_scaled[scaled_columns].astype("float64"))
    return df_scaled, scaled_columns


def _build_processing_report(
    before_frame: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    df_scaled: pd.DataFrame,
    df_ml_ready: pd.DataFrame,
    ml_metadata: dict[str, object],
    *,
    string_columns_stripped: list[str],
    text_cleaned_columns: list[str],
    currency_cleaned_columns: list[str],
    duplicate_target: str | None,
    duplicates_removed: int,
    dropped_columns: list[str],
    host_id_missing_count: int,
    rows_dropped_missing_last_review: int,
    invalid_future_last_review_count: int,
    invalid_minimum_nights_count: int,
    invalid_availability_count: int,
    outlier_adjustments: dict[str, int],
    skewness_before: dict[str, float | None],
) -> dict[str, object]:
    remaining_nulls = df_cleaned.isna().sum()
    remaining_nulls = remaining_nulls[remaining_nulls > 0]
    remaining_issues: list[str] = []
    if not remaining_nulls.empty:
        remaining_issues.append(
            "Missing values remain after preprocessing: "
            + ", ".join(f"{column}={int(count)}" for column, count in remaining_nulls.items())
        )
    if "minimum_nights" in df_cleaned.columns and not df_cleaned["minimum_nights"].between(1, 365).all():
        remaining_issues.append("minimum_nights contains values outside [1, 365]")
    if "availability_365" in df_cleaned.columns and not df_cleaned["availability_365"].between(0, 365).all():
        remaining_issues.append("availability_365 contains values outside [0, 365]")
    if "review_rate_number" in df_cleaned.columns and not df_cleaned["review_rate_number"].between(0, 5).all():
        remaining_issues.append("review_rate_number contains values outside [0, 5]")

    type_conversion_summary = {
        "float64_columns": NUMERIC_COLUMNS,
        "datetime_columns": DATETIME_COLUMNS,
        "lowercase_string_columns": STRING_COLUMNS,
    }
    missing_value_summary = {
        "categorical_fill_rules": [
            "host_identity_verified -> fill 'unconfirmed'",
            "neighbourhood_group -> geo-mapping from neighbourhood",
            "neighbourhood -> mode by neighbourhood_group + room_type",
            "instant_bookable -> fill 'false'",
            "cancellation_policy -> mode by neighbourhood + room_type",
        ],
        "datetime_fill_rules": [
            "construction_year -> mode by neighbourhood + room_type",
            "last_review > 2022-12-31 -> convert to missing",
            "last_review -> drop rows where value is missing",
        ],
        "numeric_fill_rules": [
            "price -> mean by neighbourhood + room_type",
            "minimum_nights -> abs, values > 365 to NaN, then median by neighbourhood + room_type",
            "number_of_reviews -> median by neighbourhood + room_type",
            "review_rate_number -> mode by neighbourhood + room_type",
            "calculated_host_listings_count -> host_id frequency, fallback 1 if host_id missing",
            "availability_365 -> abs, values > 365 to NaN, then median by neighbourhood + room_type",
        ],
        "host_id_missing_count": host_id_missing_count,
        "rows_dropped_missing_last_review": rows_dropped_missing_last_review,
        "invalid_value_counts": {
            "last_review_gt_2022_to_na": invalid_future_last_review_count,
            "minimum_nights_gt_365_to_na": invalid_minimum_nights_count,
            "availability_365_gt_365_to_na": invalid_availability_count,
        },
        "remaining_null_count": int(remaining_nulls.sum()),
        "remaining_null_columns": {column: int(count) for column, count in remaining_nulls.items()},
        "last_review_nat_count": int(df_cleaned["last_review"].isna().sum()) if "last_review" in df_cleaned.columns else 0,
    }
    outlier_handling_summary = {
        "clipped_columns": OUTLIER_COLUMNS,
        "applied_methods": OUTLIER_METHODS,
        "adjusted_value_counts": outlier_adjustments,
        "skewness_before": skewness_before,
        "rounded_columns": [
            "minimum_nights",
            "number_of_reviews",
            "review_rate_number",
            "calculated_host_listings_count",
            "availability_365",
        ],
    }
    integrity_summary = {
        "passed": not remaining_issues,
        "availability_365_in_range": bool(df_cleaned["availability_365"].between(0, 365).all()) if "availability_365" in df_cleaned.columns else True,
        "minimum_nights_in_range": bool(df_cleaned["minimum_nights"].between(1, 365).all()) if "minimum_nights" in df_cleaned.columns else True,
        "review_rate_number_in_range": bool(df_cleaned["review_rate_number"].between(0, 5).all()) if "review_rate_number" in df_cleaned.columns else True,
        "remaining_issues": remaining_issues,
    }
    scaled_numeric_columns = [
        column
        for column in df_scaled.select_dtypes(include="number").columns.tolist()
        if column not in NON_SCALED_NUMERIC_COLUMNS
    ]
    scaling_summary = {
        "scaled_columns": scaled_numeric_columns,
        "scaled_column_count": len(scaled_numeric_columns),
        "scaled_shape": list(df_scaled.shape),
        "active_scaler": "MinMaxScaler (visualization-only columns)",
        "alternative_scalers": ["StandardScaler", "RobustScaler", "Log Scaling"],
        "passthrough_columns": [column for column in NON_SCALED_NUMERIC_COLUMNS if column in df_cleaned.columns],
        "recommended_by_column": {
            "price": "RobustScaler or log1p + StandardScaler for ML; MinMaxScaler only for dashboard comparison.",
            "minimum_nights": "RobustScaler when modeling because the distribution stays right-skewed after capping.",
            "number_of_reviews": "log1p + StandardScaler or RobustScaler when modeling because review counts are highly skewed.",
            "review_rate_number": "No scaling is usually required because the feature already lives in the fixed 0-5 range.",
            "calculated_host_listings_count": "RobustScaler or log1p + StandardScaler for ML because host counts remain uneven.",
            "availability_365": "MinMaxScaler is acceptable for visualization; raw values are also interpretable for business analysis.",
            "booking_demand": "No scaling recommended; keep raw because the unit already represents booked nights.",
            "availability_category": "Do not scale. Use ordinal encoding only: Low=0, Medium=1, High=2.",
            "availability_efficiency": "No scaling recommended for business analysis because the raw value is directly interpretable.",
            "revenue_per_available_night": "No scaling recommended because the ratio is already in an interpretable monetary unit.",
        },
        "notes": [
            "The scaled export is kept for visualization and compares only the selected numeric columns on a common 0-1 range.",
            "The new engineered business metrics stay in raw units: booking_demand, availability_efficiency, revenue_per_available_night.",
            "availability_category is not scaled; it is ordinal-encoded only in the machine-learning export.",
            "RobustScaler is the safer fallback when strong skew or residual outliers remain.",
        ],
    }

    return {
        "rows_before": len(before_frame),
        "rows_after": len(df_cleaned),
        "columns_after": df_cleaned.shape[1],
        "duplicates_removed": duplicates_removed,
        "rows_dropped_missing_last_review": rows_dropped_missing_last_review,
        "dropped_columns": dropped_columns,
        "scaled_columns": scaling_summary["scaled_columns"],
        "remaining_issues": remaining_issues,
        "outlier_adjustments": outlier_adjustments,
        "step_metrics": {
            "data_cleaning": {
                "column_name_normalization": "lowercase + trim + special chars -> _",
                "string_columns_stripped": string_columns_stripped,
                "text_cleaned_columns": text_cleaned_columns,
                "currency_cleaned_columns": currency_cleaned_columns,
            },
            "duplicate_handling": {
                "target_column": duplicate_target,
                "duplicates_removed": duplicates_removed,
                "rows_after_dedup": len(before_frame) - duplicates_removed,
            },
            "feature_selection": {
                "dropped_columns": dropped_columns,
                "remaining_columns": df_cleaned.columns.tolist(),
            },
            "data_type_conversion": type_conversion_summary,
            "missing_value_handling": missing_value_summary,
            "outlier_handling": outlier_handling_summary,
            "integrity_check": integrity_summary,
            "download_export": {
                "cleaned_file": str(DEFAULT_CLEANED_OUTPUT_PATH),
                "scaled_file": str(DEFAULT_SCALED_OUTPUT_PATH),
                "encoded_file": str(DEFAULT_ML_OUTPUT_PATH),
                "cleaned_shape": list(df_cleaned.shape),
                "scaled_shape": list(df_scaled.shape),
                "encoded_shape": list(df_ml_ready.shape),
            },
            "feature_engineering": {
                "engineered_columns": ENGINEERED_COLUMNS,
                "non_scaled_engineered_columns": NON_SCALED_NUMERIC_COLUMNS,
                "ordinal_encoded_engineered_columns": ["availability_category"],
                "definitions": [
                    "listing_year = year(last_review)",
                    "property_age = 2022 - construction_year",
                    "estimated_revenue = (365 - availability_365) * price",
                    "occupancy_rate = (365 - availability_365) / 365",
                    "booking_flexibility_score = instant_bookable score + cancellation_policy score",
                    "customer_segment = binning minimum_nights into short stay / business-leisure / long stay",
                    "booking_demand = 365 - availability_365",
                    "availability_category = cut(availability_365, [-1, 150, 300, 365]) -> Low / Medium / High Availability",
                    "availability_efficiency = price * (365 - availability_365)",
                    "revenue_per_available_night = price * (365 - availability_365) / 365",
                ],
                "details": [
                    {
                        "name": "Listing Year",
                        "logic": "Trích xuất năm từ thời điểm đánh giá gần nhất để hỗ trợ phân tích xu hướng theo thời gian.",
                        "formula": "listing_year = year(last_review)",
                    },
                    {
                        "name": "Property Age",
                        "logic": "Ước lượng tuổi của bất động sản từ năm xây dựng để phân tích ảnh hưởng của độ mới cũ.",
                        "formula": "property_age = 2022 - construction_year",
                    },
                    {
                        "name": "Estimated Revenue",
                        "logic": "Ước lượng doanh thu tiềm năng dựa trên giá và số đêm đã được đặt.",
                        "formula": "estimated_revenue = (365 - availability_365) * price",
                    },
                    {
                        "name": "Occupancy Rate",
                        "logic": "Đo tỷ lệ lấp đầy của listing dựa trên số ngày còn trống trong năm.",
                        "formula": "occupancy_rate = (365 - availability_365) / 365",
                    },
                    {
                        "name": "Booking Flexibility Score",
                        "logic": "Tổng hợp mức linh hoạt khi đặt phòng từ khả năng đặt ngay và chính sách hủy.",
                        "formula": "booking_flexibility_score = instant_bookable score + cancellation_policy score",
                    },
                    {
                        "name": "Customer Segment",
                        "logic": "Phân nhóm khách hàng theo thời lượng lưu trú tối thiểu để hỗ trợ đọc insight theo hành vi.",
                        "formula": "customer_segment = binning minimum_nights into short stay / business-leisure / long stay",
                    },
                    {
                        "name": "Booking Demand (Nhu cầu đặt phòng)",
                        "logic": "Tính toán nhu cầu đặt phòng dựa trên số đêm không sẵn có.",
                        "formula": "booking_demand = 365 - availability_365",
                    },
                    {
                        "name": "Availability Category (Phân loại mức độ sẵn có)",
                        "logic": "Phân chia các căn hộ thành 3 nhóm Low Availability, Medium Availability và High Availability dựa trên số đêm có sẵn.",
                        "formula": "availability_category = pd.cut(availability_365, bins=[-1, 150, 300, 365], labels=['Low Availability', 'Medium Availability', 'High Availability'])",
                    },
                    {
                        "name": "Availability Efficiency (Hiệu quả sẵn có)",
                        "logic": "Đánh giá hiệu quả sử dụng các đêm có sẵn dựa trên price và availability_365.",
                        "formula": "availability_efficiency = price * (365 - availability_365)",
                    },
                    {
                        "name": "Revenue per Available Night (Doanh thu mỗi đêm có sẵn)",
                        "logic": "Đánh giá doanh thu mỗi đêm có sẵn khi giá và mức độ sẵn có liên quan đến nhau.",
                        "formula": "revenue_per_available_night = price * (365 - availability_365) / 365",
                    },
                ],
            },
            "scaling": scaling_summary,
            "ml_ready_export": {
                "dropped_identifier_columns": ml_metadata["dropped_identifier_columns"],
                "kept_identifier_columns": ml_metadata["kept_identifier_columns"],
                "datetime_engineered_columns": ml_metadata["datetime_engineered_columns"],
                "label_encoded_columns": ml_metadata["label_encoded_columns"],
                "ordinal_encoded_columns": ml_metadata["ordinal_encoded_columns"],
                "ordinal_mappings": ml_metadata["ordinal_mappings"],
                "one_hot_encoded_columns": ml_metadata["one_hot_encoded_columns"],
                "one_hot_generated_columns": ml_metadata["one_hot_generated_columns"],
                "one_hot_generated_counts": ml_metadata["one_hot_generated_counts"],
                "passthrough_numeric_columns": ml_metadata["passthrough_numeric_columns"],
                "ml_shape": ml_metadata["ml_shape"],
                "non_numeric_columns": ml_metadata["non_numeric_columns"],
            },
        },
    }


def run_preprocessing_pipeline(
    raw_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    normalized = normalize_columns(raw_frame)
    before_frame = normalized.copy()
    processed = _ensure_columns(normalized.copy(), EXPECTED_COLUMNS)

    actual_string_columns = processed.select_dtypes(include=["object", "string"]).columns.tolist()
    string_columns_stripped = actual_string_columns.copy()
    text_cleaned_columns = [column for column in actual_string_columns if column not in {"price", "service_fee"}]
    currency_cleaned_columns = [column for column in ("price", "service_fee") if column in processed.columns]
    duplicate_target = "id" if "id" in normalized.columns else None

    for column in actual_string_columns:
        processed[column] = _normalize_string_series(processed[column])
    for column in text_cleaned_columns:
        processed[column] = _clean_special_text_series(processed[column])
    for column in currency_cleaned_columns:
        processed[column] = _coerce_currency(processed[column])

    duplicates_removed = 0
    if duplicate_target == "id":
        processed["id"] = _normalize_string_series(processed["id"])
        rows_before_dedup = len(processed)
        processed = processed.drop_duplicates(subset=["id"], keep="first").copy()
        duplicates_removed = rows_before_dedup - len(processed)

    dropped_columns = [column for column in DROP_COLUMNS if column in processed.columns]
    processed = processed.drop(columns=DROP_COLUMNS, errors="ignore").reset_index(drop=True)

    processed["host_id"] = _normalize_string_series(processed["host_id"]).str.lower()
    host_id_missing_count = int(processed["host_id"].isna().sum())

    for column in STRING_COLUMNS:
        processed[column] = _normalize_string_series(processed[column]).str.lower()

    processed["neighbourhood_group"] = processed["neighbourhood_group"].replace(NEIGHBOURHOOD_GROUP_FIXES)
    processed["instant_bookable"] = processed["instant_bookable"].replace(BOOLEAN_TEXT_MAP)

    for column in NUMERIC_COLUMNS:
        processed[column] = pd.to_numeric(processed[column], errors="coerce").astype("float64")

    processed["construction_year"] = _to_datetime_year(processed["construction_year"])
    processed["last_review"] = pd.to_datetime(processed["last_review"], errors="coerce")
    invalid_future_last_review_count = int(processed["last_review"].gt(LAST_REVIEW_REFERENCE_END).sum())
    processed.loc[processed["last_review"] > LAST_REVIEW_REFERENCE_END, "last_review"] = pd.NaT

    host_verified_fallback = "unconfirmed"
    room_type_fallback = _first_mode(processed["room_type"])
    if pd.isna(room_type_fallback):
        room_type_fallback = "unknown"
    processed["host_identity_verified"] = processed["host_identity_verified"].fillna(host_verified_fallback)
    processed["room_type"] = processed["room_type"].fillna(room_type_fallback)

    neighbourhood_group_lookup = (
        processed.loc[processed["neighbourhood"].notna() & processed["neighbourhood_group"].notna(), ["neighbourhood", "neighbourhood_group"]]
        .groupby("neighbourhood", dropna=False)["neighbourhood_group"]
        .agg(_first_mode)
        .to_dict()
    )
    neighbourhood_group_fallback = _first_mode(processed["neighbourhood_group"])
    if pd.isna(neighbourhood_group_fallback):
        neighbourhood_group_fallback = "unknown"
    processed["neighbourhood_group"] = (
        processed["neighbourhood_group"]
        .fillna(processed["neighbourhood"].map(neighbourhood_group_lookup))
        .fillna(neighbourhood_group_fallback)
    )

    neighbourhood_fallback = _first_mode(processed["neighbourhood"])
    if pd.isna(neighbourhood_fallback):
        neighbourhood_fallback = "unknown"
    processed["neighbourhood"] = _fill_object_with_group_mode(
        processed,
        "neighbourhood",
        ["neighbourhood_group", "room_type"],
        neighbourhood_fallback,
    )

    processed["instant_bookable"] = processed["instant_bookable"].fillna("false")

    cancellation_policy_fallback = _first_mode(processed["cancellation_policy"])
    if pd.isna(cancellation_policy_fallback):
        cancellation_policy_fallback = "strict"
    processed["cancellation_policy"] = _fill_object_with_group_mode(
        processed,
        "cancellation_policy",
        ["neighbourhood", "room_type"],
        cancellation_policy_fallback,
    )

    construction_year_fallback = _first_mode(processed["construction_year"])
    if pd.isna(construction_year_fallback):
        construction_year_fallback = pd.Timestamp("2015-01-01")
    processed["construction_year"] = _fill_object_with_group_mode(
        processed,
        "construction_year",
        ["neighbourhood", "room_type"],
        construction_year_fallback,
    )
    processed["construction_year"] = pd.to_datetime(processed["construction_year"], errors="coerce").fillna(pd.Timestamp("2015-01-01"))

    rows_before_last_review_drop = len(processed)
    processed = processed.loc[processed["last_review"].notna()].copy().reset_index(drop=True)
    rows_dropped_missing_last_review = rows_before_last_review_drop - len(processed)

    processed["price"] = processed["price"].where(processed["price"] > 0)
    price_fallback = _safe_mean(processed["price"])
    processed["price"] = _fill_numeric_with_group_stat(
        processed,
        "price",
        ["neighbourhood", "room_type"],
        "mean",
        0.0 if pd.isna(price_fallback) else float(price_fallback),
    )

    processed["minimum_nights"] = processed["minimum_nights"].abs()
    invalid_minimum_nights_count = int(processed["minimum_nights"].gt(365).sum())
    processed.loc[processed["minimum_nights"] > 365, "minimum_nights"] = pd.NA
    minimum_nights_fallback = _safe_median(processed["minimum_nights"])
    processed["minimum_nights"] = _fill_numeric_with_group_stat(
        processed,
        "minimum_nights",
        ["neighbourhood", "room_type"],
        "median",
        1.0 if pd.isna(minimum_nights_fallback) else float(minimum_nights_fallback),
    )

    number_of_reviews_fallback = _safe_median(processed["number_of_reviews"])
    processed["number_of_reviews"] = _fill_numeric_with_group_stat(
        processed,
        "number_of_reviews",
        ["neighbourhood", "room_type"],
        "median",
        0.0 if pd.isna(number_of_reviews_fallback) else float(number_of_reviews_fallback),
    ).clip(lower=0)

    review_rate_fallback = _first_mode(processed["review_rate_number"])
    if pd.isna(review_rate_fallback):
        review_rate_fallback = 0.0
    processed["review_rate_number"] = pd.to_numeric(
        _fill_object_with_group_mode(
            processed,
            "review_rate_number",
            ["neighbourhood", "room_type"],
            review_rate_fallback,
        ),
        errors="coerce",
    ).astype("float64")

    host_listing_counts = processed.groupby("host_id")["host_id"].transform("size").astype("float64")
    processed["calculated_host_listings_count"] = (
        processed["calculated_host_listings_count"]
        .fillna(host_listing_counts)
        .fillna(1.0)
        .astype("float64")
    )
    missing_host_mask = processed["host_id"].isna()
    if missing_host_mask.any():
        generated_host_ids = pd.Series(
            [f"generated_host_{index + 1}" for index in range(len(processed))],
            index=processed.index,
            dtype="string",
        )
        processed.loc[missing_host_mask, "host_id"] = generated_host_ids.loc[missing_host_mask]

    processed["availability_365"] = processed["availability_365"].abs()
    invalid_availability_count = int(processed["availability_365"].gt(365).sum())
    processed.loc[processed["availability_365"] > 365, "availability_365"] = pd.NA
    availability_fallback = _safe_median(processed["availability_365"])
    processed["availability_365"] = _fill_numeric_with_group_stat(
        processed,
        "availability_365",
        ["neighbourhood", "room_type"],
        "median",
        0.0 if pd.isna(availability_fallback) else float(availability_fallback),
    )

    skewness_before = {
        column: _safe_skew(processed[column]) for column in OUTLIER_COLUMNS if column in processed.columns
    }
    outlier_adjustments: dict[str, int] = {}

    price_before = processed["price"].copy()
    processed["price"] = _percentile_cap(processed["price"], 0.01, 0.99)
    outlier_adjustments["price"] = int((price_before != processed["price"]).fillna(False).sum())

    minimum_nights_before = processed["minimum_nights"].copy()
    processed["minimum_nights"] = processed["minimum_nights"].replace(0, 1)
    processed["minimum_nights"] = _iqr_cap(processed["minimum_nights"]).clip(lower=1)
    outlier_adjustments["minimum_nights"] = int((minimum_nights_before != processed["minimum_nights"]).fillna(False).sum())

    number_of_reviews_before = processed["number_of_reviews"].copy()
    processed["number_of_reviews"] = _iqr_cap(processed["number_of_reviews"]).clip(lower=0)
    outlier_adjustments["number_of_reviews"] = int(
        (number_of_reviews_before != processed["number_of_reviews"]).fillna(False).sum()
    )

    review_rate_before = processed["review_rate_number"].copy()
    processed["review_rate_number"] = processed["review_rate_number"].clip(lower=0, upper=5)
    outlier_adjustments["review_rate_number"] = int((review_rate_before != processed["review_rate_number"]).fillna(False).sum())

    host_count_before = processed["calculated_host_listings_count"].copy()
    processed["calculated_host_listings_count"] = _iqr_cap(processed["calculated_host_listings_count"]).clip(lower=1)
    outlier_adjustments["calculated_host_listings_count"] = int(
        (host_count_before != processed["calculated_host_listings_count"]).fillna(False).sum()
    )

    availability_before = processed["availability_365"].copy()
    processed["availability_365"] = processed["availability_365"].clip(lower=0, upper=365)
    outlier_adjustments["availability_365"] = int((availability_before != processed["availability_365"]).fillna(False).sum())

    for column in ("minimum_nights", "number_of_reviews", "review_rate_number", "calculated_host_listings_count", "availability_365"):
        processed[column] = processed[column].round().astype("float64")

    processed["listing_year"] = processed["last_review"].dt.year.astype("float64")
    construction_year_value = processed["construction_year"].dt.year.astype("float64")
    processed["property_age"] = (DATASET_REFERENCE_YEAR - construction_year_value).clip(lower=0).astype("float64")
    processed["estimated_revenue"] = ((365.0 - processed["availability_365"]) * processed["price"]).astype("float64")
    processed["occupancy_rate"] = ((365.0 - processed["availability_365"]) / 365.0).astype("float64")
    processed["booking_flexibility_score"] = (
        processed["instant_bookable"].map({"true": 1.0, "false": 0.0}).fillna(0.0)
        + processed["cancellation_policy"].map(CANCELLATION_POLICY_SCORES).fillna(0.0)
    ).astype("float64")
    processed["customer_segment"] = pd.cut(
        processed["minimum_nights"],
        bins=[0, 3, 7, float("inf")],
        labels=CUSTOMER_SEGMENT_LABELS,
        include_lowest=True,
        right=True,
    ).astype("string")

    processed["booking_demand"] = (365.0 - processed["availability_365"]).clip(lower=0).astype("float64")
    processed["availability_category"] = pd.cut(
        processed["availability_365"],
        bins=AVAILABILITY_CATEGORY_BINS,
        labels=AVAILABILITY_CATEGORY_LABELS,
        include_lowest=True,
        right=True,
    ).astype("string")
    processed["availability_efficiency"] = (processed["price"] * processed["booking_demand"]).astype("float64")
    processed["revenue_per_available_night"] = (
        processed["availability_efficiency"] / 365.0
    ).astype("float64")

    processed[NUMERIC_COLUMNS + ENGINEERED_NUMERIC_COLUMNS] = processed[NUMERIC_COLUMNS + ENGINEERED_NUMERIC_COLUMNS].astype("float64")
    df_cleaned = processed.reset_index(drop=True).copy()
    df_scaled, scaled_columns = _build_scaled_dataframe(df_cleaned)
    df_ml_ready, ml_metadata = build_ml_ready_frame(df_cleaned)

    report = _build_processing_report(
        before_frame,
        df_cleaned,
        df_scaled,
        df_ml_ready,
        ml_metadata,
        string_columns_stripped=string_columns_stripped,
        text_cleaned_columns=text_cleaned_columns,
        currency_cleaned_columns=currency_cleaned_columns,
        duplicate_target=duplicate_target,
        duplicates_removed=duplicates_removed,
        dropped_columns=dropped_columns,
        host_id_missing_count=host_id_missing_count,
        rows_dropped_missing_last_review=rows_dropped_missing_last_review,
        invalid_future_last_review_count=invalid_future_last_review_count,
        invalid_minimum_nights_count=invalid_minimum_nights_count,
        invalid_availability_count=invalid_availability_count,
        outlier_adjustments=outlier_adjustments,
        skewness_before=skewness_before,
    )
    report["scaled_columns"] = scaled_columns
    return before_frame, df_cleaned, df_scaled, df_ml_ready, report


def preprocess_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    _, cleaned_frame, _, _, report = run_preprocessing_pipeline(frame)
    if report["remaining_issues"]:
        raise ValueError(str(report["remaining_issues"][0]))
    return cleaned_frame


def save_outputs(
    df_cleaned: pd.DataFrame,
    df_scaled: pd.DataFrame,
    df_ml_ready: pd.DataFrame,
    cleaned_output_path: Path,
    scaled_output_path: Path,
    ml_output_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cleaned_output_path.parent.mkdir(parents=True, exist_ok=True)
    scaled_output_path.parent.mkdir(parents=True, exist_ok=True)
    ml_output_path.parent.mkdir(parents=True, exist_ok=True)

    df_cleaned.to_csv(cleaned_output_path, index=False)
    print(f"Saved: {cleaned_output_path}")

    df_scaled.to_csv(scaled_output_path, index=False)
    print(f"Saved: {scaled_output_path}")

    df_ml_ready.to_csv(ml_output_path, index=False)
    print(f"Saved: {ml_output_path}")

    return df_cleaned, df_scaled, df_ml_ready


def load_dataset(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found at {input_path}. Add Airbnb_Open_Data.csv to the data directory "
            "or pass --input with a valid path."
        )
    return pd.read_csv(input_path)


def run_pipeline(
    input_path: Path = DEFAULT_INPUT_PATH,
    cleaned_output_path: Path = DEFAULT_CLEANED_OUTPUT_PATH,
    scaled_output_path: Path = DEFAULT_SCALED_OUTPUT_PATH,
    ml_output_path: Path = DEFAULT_ML_OUTPUT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_frame = load_dataset(input_path)
    _, df_cleaned, df_scaled, df_ml_ready, report = run_preprocessing_pipeline(raw_frame)
    if report["remaining_issues"]:
        raise ValueError(str(report["remaining_issues"][0]))
    return save_outputs(df_cleaned, df_scaled, df_ml_ready, cleaned_output_path, scaled_output_path, ml_output_path)


def main() -> None:
    args = parse_args()
    try:
        run_pipeline(
            input_path=args.input,
            cleaned_output_path=args.cleaned_output,
            scaled_output_path=args.scaled_output,
            ml_output_path=args.ml_output,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
