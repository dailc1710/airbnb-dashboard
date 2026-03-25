from __future__ import annotations

from html import escape

import pandas as pd
import plotly.express as px
import streamlit as st

from core.config import CHART_COLORS
from core.formatting import format_currency
from core.i18n import t, translate_room_type

ROOM_TYPE_CANONICAL_MAP = {
    "entire home/apt": "Entire home/apt",
    "private room": "Private room",
    "shared room": "Shared room",
    "hotel room": "Hotel room",
    "unknown": "Unknown",
}


def _inject_conclusion_styles() -> None:
    st.markdown(
        """
        <style>
            .conclusion-hero {
                margin-bottom: 1.2rem;
                padding: 1.7rem 1.8rem 1.35rem;
                border-radius: 24px;
                background:
                    radial-gradient(circle at top left, rgba(255, 237, 214, 0.24), transparent 32%),
                    linear-gradient(135deg, #18314b 0%, #415167 40%, #c56d41 100%);
                color: #f8f4ee;
                box-shadow: 0 22px 48px rgba(24, 49, 75, 0.18);
            }
            .conclusion-hero__badge {
                display: inline-flex;
                align-items: center;
                padding: 0.4rem 0.78rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.16);
                font-size: 0.74rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            .conclusion-hero h1 {
                margin: 0.85rem 0 0;
                font-size: clamp(2rem, 3vw, 2.7rem);
                line-height: 1.05;
                color: #fffaf4;
            }
            .conclusion-hero p {
                margin: 0.75rem 0 0;
                max-width: 52rem;
                color: rgba(248, 244, 238, 0.9);
                line-height: 1.65;
                font-size: 0.98rem;
            }
            .conclusion-section-head {
                margin: 1.35rem 0 0.85rem;
            }
            .conclusion-section-head span {
                display: inline-block;
                color: #8a6f61;
                font-size: 0.76rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            .conclusion-section-head h2 {
                margin: 0.34rem 0 0;
                color: #223247;
                font-size: 1.26rem;
            }
            .conclusion-section-head p {
                margin: 0.38rem 0 0;
                color: #68717f;
                line-height: 1.58;
            }
            .conclusion-metric-card,
            .conclusion-callout,
            .conclusion-action-card {
                background: rgba(255, 255, 255, 0.76);
                border: 1px solid rgba(31, 60, 91, 0.08);
                border-radius: 18px;
                backdrop-filter: blur(12px);
                box-shadow: 0 14px 28px rgba(31, 60, 91, 0.06);
            }
            .conclusion-metric-card {
                padding: 1rem 1rem 0.95rem;
                min-height: 9.2rem;
            }
            .conclusion-metric-card span {
                display: block;
                color: #7a818d;
                font-size: 0.8rem;
                font-weight: 700;
                letter-spacing: 0.05em;
                text-transform: uppercase;
            }
            .conclusion-metric-card strong {
                display: block;
                margin-top: 0.5rem;
                color: #223247;
                font-size: 1.78rem;
                line-height: 1.08;
            }
            .conclusion-metric-card p {
                margin: 0.48rem 0 0;
                color: #66707f;
                line-height: 1.56;
                font-size: 0.9rem;
            }
            .conclusion-callout,
            .conclusion-action-card {
                padding: 1rem 1.05rem;
                height: 100%;
            }
            .conclusion-callout__eyebrow,
            .conclusion-action-card__eyebrow {
                display: inline-flex;
                align-items: center;
                padding: 0.28rem 0.58rem;
                border-radius: 999px;
                background: rgba(201, 92, 54, 0.1);
                color: #9b4a2e;
                font-size: 0.74rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }
            .conclusion-callout h3,
            .conclusion-action-card h3 {
                margin: 0.72rem 0 0;
                color: #223247;
                font-size: 1.02rem;
                line-height: 1.35;
            }
            .conclusion-callout p,
            .conclusion-action-card p {
                margin: 0.46rem 0 0;
                color: #606977;
                line-height: 1.62;
                font-size: 0.93rem;
            }
            .conclusion-chart-card {
                background: rgba(255, 255, 255, 0.78);
                border: 1px solid rgba(31, 60, 91, 0.08);
                border-radius: 20px;
                padding: 1rem 1rem 0.5rem;
                box-shadow: 0 14px 28px rgba(31, 60, 91, 0.05);
                height: 100%;
            }
            .conclusion-chart-card h3 {
                margin: 0;
                color: #223247;
                font-size: 1.02rem;
            }
            .conclusion-chart-card p {
                margin: 0.42rem 0 0.8rem;
                color: #67707f;
                line-height: 1.56;
                font-size: 0.9rem;
            }
            @media (max-width: 640px) {
                .conclusion-hero {
                    padding: 1.2rem 1rem 1rem;
                    border-radius: 20px;
                }
                .conclusion-metric-card {
                    min-height: auto;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _display_area(value: object) -> str:
    text = str(value).replace("_", " ").strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return t("common.na")
    return text.title()


def _display_room_type(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return t("common.na")
    canonical = ROOM_TYPE_CANONICAL_MAP.get(text.lower(), text)
    return translate_room_type(canonical)


def _format_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return t("common.na")
    return f"{value:.1f}%"


def _occupancy_series(frame: pd.DataFrame) -> pd.Series:
    if "occupancy_rate" in frame.columns:
        occupancy = _coerce_numeric(frame["occupancy_rate"])
        if occupancy.dropna().max() <= 1.5:
            occupancy = occupancy * 100
        return occupancy.clip(lower=0, upper=100)

    if "availability_365" in frame.columns:
        availability = _coerce_numeric(frame["availability_365"]).clip(lower=0, upper=365)
        return ((365 - availability) / 365 * 100).clip(lower=0, upper=100)

    return pd.Series(dtype="float64")


def _metric_card(title: str, value: str, note: str) -> str:
    return (
        '<div class="conclusion-metric-card">'
        f"<span>{escape(title)}</span>"
        f"<strong>{escape(value)}</strong>"
        f"<p>{escape(note)}</p>"
        "</div>"
    )


def _section_head(kicker: str, title: str, body: str) -> str:
    return (
        '<div class="conclusion-section-head">'
        f"<span>{escape(kicker)}</span>"
        f"<h2>{escape(title)}</h2>"
        f"<p>{escape(body)}</p>"
        "</div>"
    )


def _callout_card(eyebrow: str, title: str, body: str) -> str:
    return (
        '<div class="conclusion-callout">'
        f'<span class="conclusion-callout__eyebrow">{escape(eyebrow)}</span>'
        f"<h3>{escape(title)}</h3>"
        f"<p>{escape(body)}</p>"
        "</div>"
    )


def _action_card(eyebrow: str, title: str, body: str) -> str:
    return (
        '<div class="conclusion-action-card">'
        f'<span class="conclusion-action-card__eyebrow">{escape(eyebrow)}</span>'
        f"<h3>{escape(title)}</h3>"
        f"<p>{escape(body)}</p>"
        "</div>"
    )


def render_page(frame: pd.DataFrame) -> None:
    _inject_conclusion_styles()

    prepared = frame.copy()
    for column in ("price", "number_of_reviews", "availability_365", "estimated_revenue", "occupancy_rate"):
        if column in prepared.columns:
            prepared[column] = _coerce_numeric(prepared[column])

    if prepared.empty:
        st.title(t("conclusion.title"))
        st.caption(t("conclusion.caption"))
        st.info(t("insight.no_data"))
        return

    listing_count = len(prepared)
    price_series = prepared["price"].dropna() if "price" in prepared.columns else pd.Series(dtype="float64")
    revenue_series = prepared["estimated_revenue"].dropna() if "estimated_revenue" in prepared.columns else pd.Series(dtype="float64")
    occupancy_series = _occupancy_series(prepared).dropna()

    area_mix = (
        prepared["neighbourhood_group"]
        .astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
        .dropna()
        .value_counts(normalize=True)
        if "neighbourhood_group" in prepared.columns
        else pd.Series(dtype="float64")
    )
    room_mix = (
        prepared["room_type"]
        .astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
        .dropna()
        .value_counts(normalize=True)
        if "room_type" in prepared.columns
        else pd.Series(dtype="float64")
    )

    area_price = (
        prepared.groupby("neighbourhood_group", dropna=False)["price"].median().dropna().sort_values(ascending=False)
        if {"neighbourhood_group", "price"}.issubset(prepared.columns)
        else pd.Series(dtype="float64")
    )
    area_occupancy = (
        prepared.assign(_occupancy_pct=_occupancy_series(prepared))
        .groupby("neighbourhood_group", dropna=False)["_occupancy_pct"]
        .mean()
        .dropna()
        .sort_values(ascending=False)
        if "neighbourhood_group" in prepared.columns and not occupancy_series.empty
        else pd.Series(dtype="float64")
    )
    room_reviews = (
        prepared.groupby("room_type", dropna=False)["number_of_reviews"].median().dropna().sort_values(ascending=False)
        if {"room_type", "number_of_reviews"}.issubset(prepared.columns)
        else pd.Series(dtype="float64")
    )

    top_price_area = _display_area(area_price.index[0]) if not area_price.empty else t("common.na")
    top_price_value = format_currency(area_price.iloc[0], fallback=t("common.na")) if not area_price.empty else t("common.na")
    top_occupancy_area = _display_area(area_occupancy.index[0]) if not area_occupancy.empty else t("common.na")
    top_occupancy_value = _format_percent(float(area_occupancy.iloc[0])) if not area_occupancy.empty else t("common.na")
    dominant_room_type = _display_room_type(room_mix.index[0]) if not room_mix.empty else t("common.na")
    dominant_room_share = f"{room_mix.iloc[0] * 100:.1f}%" if not room_mix.empty else t("common.na")
    review_leader_room = _display_room_type(room_reviews.index[0]) if not room_reviews.empty else t("common.na")
    review_leader_value = f"{room_reviews.iloc[0]:.0f}" if not room_reviews.empty else t("common.na")

    lead_areas = [_display_area(area) for area in area_mix.index[:2]]
    lead_areas_label = ", ".join(lead_areas) if lead_areas else t("common.na")
    lead_areas_share = f"{area_mix.iloc[:2].sum() * 100:.1f}%" if not area_mix.empty else t("common.na")

    st.markdown(
        f"""
        <section class="conclusion-hero">
            <span class="conclusion-hero__badge">{escape(t("conclusion.hero_badge"))}</span>
            <h1>{escape(t("conclusion.title"))}</h1>
            <p>{escape(t("conclusion.hero_body", listings=f"{listing_count:,}", median_price=format_currency(price_series.median(), fallback=t("common.na")), occupancy=_format_percent(float(occupancy_series.mean())) if not occupancy_series.empty else t("common.na"), top_area=top_price_area))}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    metric_cards = [
        _metric_card(
            t("conclusion.kpi.listings"),
            f"{listing_count:,}",
            t("conclusion.kpi.listings_note"),
        ),
        _metric_card(
            t("conclusion.kpi.price"),
            format_currency(price_series.median(), fallback=t("common.na")),
            t("conclusion.kpi.price_note", area=top_price_area),
        ),
        _metric_card(
            t("conclusion.kpi.revenue"),
            format_currency(revenue_series.median(), fallback=t("common.na")),
            t("conclusion.kpi.revenue_note"),
        ),
        _metric_card(
            t("conclusion.kpi.occupancy"),
            _format_percent(float(occupancy_series.mean())) if not occupancy_series.empty else t("common.na"),
            t("conclusion.kpi.occupancy_note", area=top_occupancy_area),
        ),
    ]
    for column, card in zip(metric_columns, metric_cards):
        with column:
            st.markdown(card, unsafe_allow_html=True)

    st.markdown(
        _section_head(
            t("conclusion.takeaways_kicker"),
            t("conclusion.takeaways_title"),
            t("conclusion.takeaways_caption"),
        ),
        unsafe_allow_html=True,
    )

    takeaway_columns = st.columns(3)
    takeaway_cards = [
        _callout_card(
            t("conclusion.takeaway.pricing_eyebrow"),
            t("conclusion.takeaway.pricing_title"),
            t("conclusion.takeaway.pricing_body", area=top_price_area, price=top_price_value),
        ),
        _callout_card(
            t("conclusion.takeaway.inventory_eyebrow"),
            t("conclusion.takeaway.inventory_title"),
            t(
                "conclusion.takeaway.inventory_body",
                room_type=dominant_room_type,
                share=dominant_room_share,
                areas=lead_areas_label,
                area_share=lead_areas_share,
            ),
        ),
        _callout_card(
            t("conclusion.takeaway.demand_eyebrow"),
            t("conclusion.takeaway.demand_title"),
            t(
                "conclusion.takeaway.demand_body",
                area=top_occupancy_area,
                occupancy=top_occupancy_value,
                room_type=review_leader_room,
                reviews=review_leader_value,
            ),
        ),
    ]
    for column, card in zip(takeaway_columns, takeaway_cards):
        with column:
            st.markdown(card, unsafe_allow_html=True)

    st.markdown(
        _section_head(
            t("conclusion.visual_kicker"),
            t("conclusion.visual_title"),
            t("conclusion.visual_caption"),
        ),
        unsafe_allow_html=True,
    )

    chart_left, chart_right = st.columns((1.15, 0.85))

    with chart_left:
        st.markdown(
            f"""
            <div class="conclusion-chart-card">
                <h3>{escape(t("conclusion.chart.area_title"))}</h3>
                <p>{escape(t("conclusion.chart.area_caption"))}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not area_price.empty:
            area_chart = area_price.reset_index()
            area_chart.columns = ["neighbourhood_group", "median_price"]
            area_chart["neighbourhood_group"] = area_chart["neighbourhood_group"].map(_display_area)
            area_chart = area_chart.sort_values("median_price", ascending=True)
            bar = px.bar(
                area_chart,
                x="median_price",
                y="neighbourhood_group",
                orientation="h",
                color="neighbourhood_group",
                color_discrete_sequence=CHART_COLORS,
                text=area_chart["median_price"].map(lambda value: format_currency(value, fallback=t("common.na"))),
            )
            bar.update_traces(textposition="outside", marker_line_width=0, hovertemplate="%{y}<br>%{text}<extra></extra>")
            bar.update_layout(
                margin=dict(l=12, r=16, t=8, b=8),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title=None,
                yaxis_title=None,
                showlegend=False,
                height=320,
            )
            bar.update_xaxes(showgrid=True, gridcolor="rgba(31,60,91,0.08)")
            bar.update_yaxes(showgrid=False)
            st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info(t("common.na"))

    with chart_right:
        st.markdown(
            f"""
            <div class="conclusion-chart-card">
                <h3>{escape(t("conclusion.chart.room_title"))}</h3>
                <p>{escape(t("conclusion.chart.room_caption"))}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not room_mix.empty:
            room_chart = room_mix.reset_index()
            room_chart.columns = ["room_type", "share"]
            room_chart["room_type"] = room_chart["room_type"].map(_display_room_type)
            room_chart["share_pct"] = room_chart["share"] * 100
            pie = px.pie(
                room_chart,
                names="room_type",
                values="share_pct",
                hole=0.62,
                color="room_type",
                color_discrete_sequence=CHART_COLORS,
            )
            pie.update_traces(
                texttemplate="%{percent}",
                hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>",
                marker=dict(line=dict(color="rgba(255,255,255,0.9)", width=2)),
            )
            pie.update_layout(
                margin=dict(l=12, r=12, t=8, b=8),
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                legend_title_text="",
                height=320,
            )
            st.plotly_chart(pie, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info(t("common.na"))

    st.markdown(
        _section_head(
            t("conclusion.actions_kicker"),
            t("conclusion.actions_title"),
            t("conclusion.actions_caption"),
        ),
        unsafe_allow_html=True,
    )

    action_columns = st.columns(3)
    action_cards = [
        _action_card(
            t("conclusion.action.pricing_eyebrow"),
            t("conclusion.action.pricing_title"),
            t("conclusion.action.pricing_body", area=top_price_area, price=top_price_value),
        ),
        _action_card(
            t("conclusion.action.portfolio_eyebrow"),
            t("conclusion.action.portfolio_title"),
            t("conclusion.action.portfolio_body", room_type=dominant_room_type, share=dominant_room_share),
        ),
        _action_card(
            t("conclusion.action.operations_eyebrow"),
            t("conclusion.action.operations_title"),
            t("conclusion.action.operations_body", area=top_occupancy_area, occupancy=top_occupancy_value, room_type=review_leader_room),
        ),
    ]
    for column, card in zip(action_columns, action_cards):
        with column:
            st.markdown(card, unsafe_allow_html=True)
