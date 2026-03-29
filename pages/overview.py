from __future__ import annotations

from html import escape

import pandas as pd
import plotly.express as px
import streamlit as st

from core.config import CHART_COLORS, SAMPLE_SOURCE_LABEL
from core.formatting import format_currency
from core.i18n import (
    display_source_label,
    get_language,
    nav_label,
    t,
    translate_room_type,
)


def _inject_overview_styles() -> None:
    st.markdown(
        """
        <style>
            .overview-hero-shell {
                margin-bottom: 1.25rem;
                padding: 1.7rem 1.8rem 1.2rem;
                border-radius: 24px;
                background:
                    radial-gradient(circle at top left, rgba(255, 237, 214, 0.22), transparent 28%),
                    linear-gradient(135deg, #18314b 0%, #46536a 38%, #b95a36 100%);
                color: #f9f6f0;
                box-shadow: 0 22px 48px rgba(24, 49, 75, 0.22);
            }
            .overview-hero-grid {
                display: grid;
                grid-template-columns: minmax(0, 1.4fr) minmax(280px, 0.9fr);
                gap: 1rem;
                align-items: stretch;
            }
            .overview-badge {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                padding: 0.38rem 0.72rem;
                border-radius: 999px;
                border: 1px solid rgba(255, 255, 255, 0.18);
                background: rgba(255, 255, 255, 0.08);
                font-size: 0.76rem;
                font-weight: 700;
                letter-spacing: 0.06em;
                text-transform: uppercase;
            }
            .overview-hero-copy h1 {
                margin: 0.85rem 0 0;
                font-size: clamp(2rem, 3vw, 2.6rem);
                line-height: 1.05;
            }
            .overview-hero-copy p {
                margin: 0.75rem 0 0;
                max-width: 46rem;
                color: rgba(249, 246, 240, 0.88);
                line-height: 1.65;
                font-size: 1rem;
            }
            .overview-hero-panel {
                border-radius: 20px;
                padding: 1.1rem 1.1rem 1rem;
                background: rgba(255, 255, 255, 0.12);
                border: 1px solid rgba(255, 255, 255, 0.12);
                backdrop-filter: blur(10px);
            }
            .overview-hero-panel__kicker {
                font-size: 0.77rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: rgba(255, 245, 232, 0.72);
                font-weight: 700;
            }
            .overview-hero-panel__title {
                margin-top: 0.55rem;
                font-size: 1.02rem;
                line-height: 1.55;
                color: #fff8f1;
                font-weight: 600;
            }
            .overview-hero-panel__metrics {
                margin-top: 1rem;
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.75rem;
            }
            .overview-hero-panel__metric {
                padding: 0.85rem 0.9rem;
                border-radius: 16px;
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
            .overview-hero-panel__metric span {
                display: block;
                font-size: 0.78rem;
                color: rgba(255, 245, 232, 0.72);
            }
            .overview-hero-panel__metric strong {
                display: block;
                margin-top: 0.22rem;
                font-size: 1.18rem;
                color: #fffaf4;
            }
            .overview-chip-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.6rem;
                margin-top: 1rem;
            }
            .overview-chip {
                display: inline-flex;
                align-items: center;
                padding: 0.48rem 0.82rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.14);
                color: #fff6eb;
                font-size: 0.82rem;
                font-weight: 600;
            }
            .overview-section-head {
                margin: 1.4rem 0 0.75rem;
            }
            .overview-section-head h2 {
                margin: 0;
                font-size: 1.28rem;
                color: #223247;
            }
            .overview-section-head p {
                margin: 0.35rem 0 0;
                color: #6a7280;
                line-height: 1.58;
            }
            .overview-metric-grid,
            .overview-card-grid {
                display: grid;
                gap: 0.9rem;
            }
            .overview-metric-grid {
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            }
            .overview-card-grid {
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            }
            .overview-metric-card,
            .overview-card,
            .overview-insight-card {
                background: rgba(255, 255, 255, 0.74);
                border: 1px solid rgba(31, 60, 91, 0.08);
                border-radius: 18px;
                backdrop-filter: blur(12px);
                box-shadow: 0 14px 28px rgba(31, 60, 91, 0.06);
            }
            .overview-metric-card {
                padding: 1rem 1rem 0.95rem;
            }
            .overview-metric-card span {
                display: block;
                font-size: 0.8rem;
                letter-spacing: 0.05em;
                text-transform: uppercase;
                color: #7a818d;
                font-weight: 700;
            }
            .overview-metric-card strong {
                display: block;
                margin-top: 0.45rem;
                font-size: 1.7rem;
                line-height: 1.1;
                color: #223247;
            }
            .overview-card {
                padding: 1rem 1.05rem;
            }
            .overview-card__eyebrow {
                display: inline-block;
                padding: 0.26rem 0.52rem;
                border-radius: 999px;
                background: rgba(201, 92, 54, 0.1);
                color: #9b4a2e;
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }
            .overview-card h3 {
                margin: 0.72rem 0 0;
                color: #223247;
                font-size: 1.05rem;
            }
            .overview-card p {
                margin: 0.48rem 0 0;
                color: #606977;
                line-height: 1.58;
                font-size: 0.94rem;
            }
            .overview-insight-stack {
                display: grid;
                gap: 0.75rem;
            }
            .overview-insight-card {
                padding: 0.95rem 1rem;
            }
            .overview-insight-card p {
                margin: 0;
                color: #2b384b;
                line-height: 1.62;
            }
            @media (max-width: 980px) {
                .overview-hero-grid {
                    grid-template-columns: 1fr;
                }
                .overview-hero-panel__metrics {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }
            @media (max-width: 640px) {
                .overview-hero-shell {
                    padding: 1.2rem 1rem 1rem;
                    border-radius: 20px;
                }
                .overview-hero-panel__metrics {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _year_coverage_label(frame: pd.DataFrame) -> str:
    if "listing_year" in frame.columns:
        year_series = pd.to_numeric(frame["listing_year"], errors="coerce").dropna()
    elif "last_review" in frame.columns:
        year_series = pd.to_datetime(frame["last_review"], errors="coerce").dt.year.dropna()
    else:
        return t("common.na")

    if year_series.empty:
        return t("common.na")
    return f"{int(year_series.min())} - {int(year_series.max())}"


def _title_case_label(value: object) -> str:
    return str(value).replace("_", " ").strip().title()


def _build_metric_card(title: str, value: str) -> str:
    return (
        '<div class="overview-metric-card">'
        f"<span>{escape(title)}</span>"
        f"<strong>{escape(value)}</strong>"
        "</div>"
    )


def _build_feature_card(*, eyebrow: str, title: str, body: str) -> str:
    return (
        '<div class="overview-card">'
        f'<div class="overview-card__eyebrow">{escape(eyebrow)}</div>'
        f"<h3>{escape(title)}</h3>"
        f"<p>{escape(body)}</p>"
        "</div>"
    )


def _localized_output_eyebrows() -> tuple[str, str, str]:
    if get_language() == "vi":
        return ("CSV", "Chuẩn hóa", "Học máy")
    return ("CSV", "Scale", "ML")


def render_page(frame: pd.DataFrame, source_label: str) -> None:
    _inject_overview_styles()
    source_text = display_source_label(source_label) or t("common.na")
    price_series = frame["price"] if "price" in frame.columns else pd.Series(dtype=float)
    borough_count = int(frame["neighbourhood_group"].nunique()) if "neighbourhood_group" in frame.columns else 0
    room_type_count = int(frame["room_type"].nunique()) if "room_type" in frame.columns else 0

    hero_metric_cards = [
        _build_metric_card(t("overview.metric.listings"), f"{len(frame):,}"),
        _build_metric_card(t("overview.kpi.year_coverage"), _year_coverage_label(frame)),
        _build_metric_card(t("overview.kpi.borough_groups"), f"{borough_count:,}" if borough_count else t("common.na")),
        _build_metric_card(t("overview.kpi.room_types"), f"{room_type_count:,}" if room_type_count else t("common.na")),
    ]
    hero_chips = [
        t("overview.hero_chip.clean"),
        t("overview.hero_chip.features"),
        t("overview.hero_chip.eda"),
        t("overview.hero_chip.ml"),
    ]

    st.markdown(
        f"""
        <div class="overview-hero-shell">
            <div class="overview-hero-grid">
                <div class="overview-hero-copy">
                    <div class="overview-badge">{escape(t("overview.hero_badge"))}</div>
                    <h1>{escape(t("app.title"))}</h1>
                    <p>{t("overview.hero_body", source=escape(source_text))}</p>
                </div>
                <div class="overview-hero-panel">
                    <div class="overview-hero-panel__kicker">{escape(t("overview.hero_panel.kicker"))}</div>
                    <div class="overview-hero-panel__title">{escape(t("overview.hero_panel.title"))}</div>
                    <div class="overview-hero-panel__metrics">
                        {''.join(hero_metric_cards)}
                    </div>
                </div>
            </div>
            <div class="overview-chip-row">
                {''.join(f'<span class="overview-chip">{escape(chip)}</span>' for chip in hero_chips)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if source_label == SAMPLE_SOURCE_LABEL:
        st.info(t("overview.sample_info"))
    elif frame.empty:
        st.info(t("common.upload_required"))
        st.caption(t("common.upload_required_detail"))

    st.markdown(
        f"""
        <div class="overview-section-head">
            <h2>{escape(t("overview.section.questions"))}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    question_cards = [
        _build_feature_card(
            eyebrow="01",
            title=t("overview.question.price.title"),
            body=t("overview.question.price.body"),
        ),
        _build_feature_card(
            eyebrow="02",
            title=t("overview.question.demand.title"),
            body=t("overview.question.demand.body"),
        ),
        _build_feature_card(
            eyebrow="03",
            title=t("overview.question.revenue.title"),
            body=t("overview.question.revenue.body"),
        ),
    ]
    st.markdown(f'<div class="overview-card-grid">{"".join(question_cards)}</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="overview-section-head">
            <h2>{escape(t("overview.section.workflow"))}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    workflow_cards = [
        _build_feature_card(
            eyebrow="01",
            title=nav_label("overview"),
            body=t("overview.workflow.overview.body"),
        ),
        _build_feature_card(
            eyebrow="02",
            title=nav_label("data_raw"),
            body=t("overview.workflow.data_raw.body"),
        ),
        _build_feature_card(
            eyebrow="03",
            title=nav_label("preprocessing"),
            body=t("overview.workflow.preprocessing.body"),
        ),
        _build_feature_card(
            eyebrow="04",
            title=nav_label("eda"),
            body=t("overview.workflow.eda.body"),
        ),
        _build_feature_card(
            eyebrow="05",
            title=nav_label("conclusion"),
            body=t("overview.workflow.conclusion.body"),
        ),
        _build_feature_card(
            eyebrow="06",
            title=nav_label("chatbot"),
            body=t("overview.workflow.chatbot.body"),
        ),
    ]
    st.markdown(f'<div class="overview-card-grid">{"".join(workflow_cards)}</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="overview-section-head">
            <h2>{escape(t("overview.section.outputs"))}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    output_cards = [
        _build_feature_card(
            eyebrow=_localized_output_eyebrows()[0],
            title=t("overview.output.clean.title"),
            body=t("overview.output.clean.body"),
        ),
        _build_feature_card(
            eyebrow=_localized_output_eyebrows()[1],
            title=t("overview.output.scaled.title"),
            body=t("overview.output.scaled.body"),
        ),
        _build_feature_card(
            eyebrow=_localized_output_eyebrows()[2],
            title=t("overview.output.encoded.title"),
            body=t("overview.output.encoded.body"),
        ),
    ]
    st.markdown(f'<div class="overview-card-grid">{"".join(output_cards)}</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="overview-section-head">
            <h2>{escape(t("overview.section.snapshot"))}</h2>
            <p>{escape(t("overview.section.snapshot_body"))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    snapshot_cards = [
        _build_metric_card(t("overview.metric.listings"), f"{len(frame):,}"),
        _build_metric_card(
            t("overview.metric.median_price"),
            format_currency(price_series.median() if not price_series.empty else None, fallback=t("common.na")),
        ),
        _build_metric_card(t("overview.kpi.borough_groups"), f"{borough_count:,}" if borough_count else t("common.na")),
        _build_metric_card(t("overview.kpi.room_types"), f"{room_type_count:,}" if room_type_count else t("common.na")),
        _build_metric_card(t("overview.kpi.year_coverage"), _year_coverage_label(frame)),
    ]
    st.markdown(f'<div class="overview-metric-grid">{"".join(snapshot_cards)}</div>', unsafe_allow_html=True)

    chart_col, mix_col = st.columns([1.15, 0.85], gap="large")
    with chart_col:
        if {"neighbourhood_group", "price"}.issubset(frame.columns):
            area_summary = (
                frame.groupby("neighbourhood_group", dropna=False)["price"]
                .median()
                .sort_values(ascending=False)
                .reset_index(name="median_price")
            )
            area_summary["borough_label"] = area_summary["neighbourhood_group"].map(_title_case_label)
            area_summary["median_price_label"] = area_summary["median_price"].round(0).fillna(0).astype("int64").astype(str)
            bar_chart = px.bar(
                area_summary,
                x="borough_label",
                y="median_price",
                text="median_price_label",
                color="borough_label",
                title=t("overview.chart.area_price"),
                color_discrete_sequence=CHART_COLORS,
            )
            bar_chart.update_layout(
                margin=dict(l=10, r=10, t=54, b=10),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            bar_chart.update_traces(textposition="outside", cliponaxis=False)
            bar_chart.update_xaxes(title=None)
            bar_chart.update_yaxes(title=None)
            st.plotly_chart(bar_chart, use_container_width=True)
        else:
            st.info(t("common.na"))

    with mix_col:
        if "room_type" in frame.columns:
            room_mix = frame["room_type"].value_counts().reset_index()
            room_mix.columns = ["room_type", "count"]
            room_mix["room_type_label"] = room_mix["room_type"].map(translate_room_type)
            donut = px.pie(
                room_mix,
                names="room_type_label",
                values="count",
                hole=0.58,
                title=t("overview.chart.room_mix"),
                color_discrete_sequence=CHART_COLORS,
            )
            donut.update_layout(
                margin=dict(l=10, r=10, t=54, b=10),
                legend_title_text="",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(donut, use_container_width=True)
        else:
            st.info(t("common.na"))
