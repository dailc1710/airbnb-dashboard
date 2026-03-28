from __future__ import annotations

import pandas as pd

from core.formatting import format_currency
from core.i18n import t, translate_room_type

ROOM_TYPE_CANONICAL_MAP = {
    "entire home/apt": "Entire home/apt",
    "private room": "Private room",
    "shared room": "Shared room",
    "hotel room": "Hotel room",
}
AREA_CANONICAL_MAP = {
    "brooklyn": "Brooklyn",
    "manhattan": "Manhattan",
    "queens": "Queens",
    "bronx": "Bronx",
    "staten island": "Staten Island",
}


def _canonicalize_series(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    lookup = {key.lower(): value for key, value in mapping.items()}
    cleaned = series.astype("string").str.strip()
    return cleaned.map(lambda value: lookup.get(str(value).lower(), str(value)) if pd.notna(value) else value)


def _prepare_chat_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()

    if "room_type" in prepared.columns:
        prepared["room_type"] = _canonicalize_series(prepared["room_type"], ROOM_TYPE_CANONICAL_MAP)

    if "neighbourhood_group" in prepared.columns:
        prepared["neighbourhood_group"] = _canonicalize_series(prepared["neighbourhood_group"], AREA_CANONICAL_MAP)

    if "price" in prepared.columns:
        prepared["price"] = pd.to_numeric(
            prepared["price"].astype("string").str.replace(r"[\$,]", "", regex=True),
            errors="coerce",
        )

    for column in ("number_of_reviews", "availability_365", "booking_demand"):
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    if "booking_demand" not in prepared.columns and "availability_365" in prepared.columns:
        prepared["booking_demand"] = 365 - prepared["availability_365"]

    return prepared


def build_chat_context(frame: pd.DataFrame) -> str:
    prepared = _prepare_chat_frame(frame)
    if prepared.empty:
        return "No dataset is currently available."

    context_lines = [
        f"Current cleaned dataset rows: {len(prepared):,}",
        f"Current cleaned dataset columns: {prepared.shape[1]}",
        "Current cleaned dataset insights:",
    ]

    if "price" in prepared.columns and prepared["price"].notna().any():
        context_lines.append(
            f"- Price median: {format_currency(prepared['price'].median(), fallback='N/A')}; mean: {format_currency(prepared['price'].mean(), fallback='N/A')}"
        )

    if {"neighbourhood_group", "price"}.issubset(prepared.columns):
        area_prices = prepared.groupby("neighbourhood_group")["price"].median().sort_values(ascending=False)
        if not area_prices.empty:
            top_areas = ", ".join(
                f"{area}: {format_currency(value, fallback='N/A')}"
                for area, value in area_prices.head(3).items()
            )
            context_lines.append(f"- Top areas by median price: {top_areas}")

    if "room_type" in prepared.columns:
        room_mix = prepared["room_type"].value_counts(normalize=True).mul(100).round(1)
        if not room_mix.empty:
            room_mix_text = ", ".join(
                f"{room}: {share:.1f}%"
                for room, share in room_mix.items()
            )
            context_lines.append(f"- Room type mix: {room_mix_text}")

    if {"neighbourhood_group", "booking_demand"}.issubset(prepared.columns):
        demand_rank = prepared.groupby("neighbourhood_group")["booking_demand"].median().sort_values(ascending=False)
        if not demand_rank.empty:
            demand_text = ", ".join(
                f"{area}: {value:.0f} nights"
                for area, value in demand_rank.head(3).items()
            )
            context_lines.append(f"- Top areas by median booking demand: {demand_text}")

    if "number_of_reviews" in prepared.columns and prepared["number_of_reviews"].notna().any():
        context_lines.append(
            f"- Reviews median: {prepared['number_of_reviews'].median():.0f}; 90th percentile: {prepared['number_of_reviews'].quantile(0.9):.0f}"
        )

    if "availability_365" in prepared.columns and prepared["availability_365"].notna().any():
        context_lines.append(
            f"- Availability median: {prepared['availability_365'].median():.0f} days; mean: {prepared['availability_365'].mean():.0f} days"
        )

        availability_category = pd.cut(
            prepared["availability_365"],
            bins=[-1, 150, 300, 365],
            labels=["Low Availability", "Medium Availability", "High Availability"],
            include_lowest=True,
            right=True,
        )
        availability_mix = availability_category.value_counts(normalize=True).mul(100).round(1)
        if not availability_mix.empty:
            mix_text = ", ".join(
                f"{label}: {share:.1f}%"
                for label, share in availability_mix.items()
            )
            context_lines.append(f"- Availability category mix: {mix_text}")

    return "\n".join(context_lines)


def insight_sentences(frame: pd.DataFrame) -> list[str]:
    frame = _prepare_chat_frame(frame)
    insights: list[str] = []
    if frame.empty:
        return [t("insight.no_data")]

    if "price" in frame.columns:
        insights.append(
            t(
                "insight.typical_price",
                median=format_currency(frame["price"].median(), fallback=t("common.na")),
                mean=format_currency(frame["price"].mean(), fallback=t("common.na")),
            )
        )

    if {"neighbourhood_group", "price"}.issubset(frame.columns):
        area_prices = frame.groupby("neighbourhood_group", dropna=False)["price"].median().sort_values(ascending=False)
        if not area_prices.empty:
            insights.append(
                t(
                    "insight.top_area",
                    area=area_prices.index[0],
                    price=format_currency(area_prices.iloc[0], fallback=t("common.na")),
                )
            )

    if "room_type" in frame.columns:
        room_mix = frame["room_type"].value_counts(normalize=True)
        if not room_mix.empty:
            insights.append(
                t(
                    "insight.room_mix",
                    room_type=translate_room_type(room_mix.index[0]),
                    share=f"{room_mix.iloc[0] * 100:.1f}",
                )
            )

    if {"room_type", "number_of_reviews"}.issubset(frame.columns):
        review_rank = frame.groupby("room_type", dropna=False)["number_of_reviews"].median().sort_values(ascending=False)
        if not review_rank.empty:
            insights.append(
                t(
                    "insight.review_rank",
                    room_type=translate_room_type(review_rank.index[0]),
                    reviews=f"{review_rank.iloc[0]:.0f}",
                )
            )

    if {"availability_365", "price"}.issubset(frame.columns) and len(frame) > 1:
        correlation = frame[["availability_365", "price"]].corr(numeric_only=True).iloc[0, 1]
        if pd.notna(correlation):
            strength_key = "weak"
            if abs(correlation) >= 0.5:
                strength_key = "strong"
            elif abs(correlation) >= 0.25:
                strength_key = "moderate"
            direction_key = "positive" if correlation >= 0 else "negative"
            insights.append(
                t(
                    "insight.availability_correlation",
                    strength=t(f"insight.strength.{strength_key}"),
                    direction=t(f"insight.direction.{direction_key}"),
                    correlation=f"{correlation:.2f}",
                )
            )

    return insights


def answer_chat_question(question: str, frame: pd.DataFrame) -> str:
    frame = _prepare_chat_frame(frame)
    prompt = question.lower()
    insights = insight_sentences(frame)

    if frame.empty:
        return insights[0]

    if any(token in prompt for token in ("neighbourhood", "neighborhood", "area", "borough", "khu", "quận", "vùng")) and {"neighbourhood_group", "price"}.issubset(frame.columns):
        area_prices = frame.groupby("neighbourhood_group")["price"].median().sort_values(ascending=False)
        top_two = ", ".join(
            f"{area}: {format_currency(value, fallback=t('common.na'))}"
            for area, value in area_prices.head(2).items()
        )
        return t("chat.answer.top_areas", areas=top_two)

    if any(token in prompt for token in ("price", "cost", "expensive", "cheap", "giá", "đắt", "rẻ", "chi phí")) and "price" in frame.columns:
        return t(
            "chat.answer.price",
            median=format_currency(frame["price"].median(), fallback=t("common.na")),
            mean=format_currency(frame["price"].mean(), fallback=t("common.na")),
            percentile=format_currency(frame["price"].quantile(0.75), fallback=t("common.na")),
        )

    if any(token in prompt for token in ("demand", "booking", "nhu cầu", "đặt phòng", "booked")) and {"booking_demand", "neighbourhood_group"}.issubset(frame.columns):
        demand_rank = frame.groupby("neighbourhood_group")["booking_demand"].median().sort_values(ascending=False)
        top_two = ", ".join(
            f"{area}: {value:.0f}"
            for area, value in demand_rank.head(2).items()
        )
        return t(
            "chat.answer.demand",
            areas=top_two,
            median=f"{frame['booking_demand'].median():.0f}",
        )

    if any(token in prompt for token in ("room", "type", "phòng", "loại")) and "room_type" in frame.columns:
        room_mix = frame["room_type"].value_counts(normalize=True).mul(100).round(1)
        room_summary = ", ".join(f"{translate_room_type(room)}: {share:.1f}%" for room, share in room_mix.items())
        return t("chat.answer.room_mix", summary=room_summary)

    if any(token in prompt for token in ("review", "rating", "đánh giá")) and "number_of_reviews" in frame.columns:
        return t(
            "chat.answer.reviews",
            median=f"{frame['number_of_reviews'].median():.0f}",
            top_decile=f"{frame['number_of_reviews'].quantile(0.9):.0f}",
        )

    if any(token in prompt for token in ("availability", "book", "open", "trống", "khả dụng", "available")) and "availability_365" in frame.columns:
        return t(
            "chat.answer.availability",
            median=f"{frame['availability_365'].median():.0f}",
            mean=f"{frame['availability_365'].mean():.0f}",
        )

    return " ".join(insights[:2])
