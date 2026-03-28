from __future__ import annotations

import hashlib
import io
import pandas as pd
import plotly.express as px
import streamlit as st

from core.data import (
    build_missing_table,
    normalize_columns,
)
from core.i18n import localize_dataframe_for_display, t, translate_room_type
from pages.preprocessing import run_processing_pipeline, store_processed_outputs

COLUMN_MEANINGS_VI = {
    "id": "Mã định danh duy nhất của mỗi listing.",
    "name": "Tên hiển thị của listing trên Airbnb.",
    "host_id": "Mã định danh của chủ nhà (host).",
    "host_name": "Tên của chủ nhà.",
    "neighbourhood_group": "Khu vực lớn hoặc quận nơi listing tọa lạc, ví dụ Manhattan hoặc Brooklyn.",
    "neighbourhood": "Khu phố hoặc địa bàn cụ thể của listing bên trong neighbourhood_group.",
    "lat": "Vĩ độ của vị trí listing.",
    "long": "Kinh độ của vị trí listing.",
    "country": "Quốc gia của listing.",
    "country_code": "Mã quốc gia viết tắt của listing.",
    "room_type": "Loại chỗ ở được cho thuê, ví dụ Entire home/apt hoặc Private room.",
    "price": "Giá niêm yết của listing, thường là theo đêm.",
    "service_fee": "Phí dịch vụ đi kèm với booking.",
    "minimum_nights": "Số đêm tối thiểu khách phải đặt.",
    "number_of_reviews": "Tổng số lượt đánh giá mà listing đã nhận.",
    "reviews_per_month": "Số lượt đánh giá trung bình mỗi tháng.",
    "last_review": "Ngày gần nhất listing nhận được đánh giá.",
    "review_rate_number": "Điểm đánh giá tổng quan của listing, thường theo thang điểm 1-5.",
    "calculated_host_listings_count": "Số listing mà cùng một host đang sở hữu hoặc quản lý trong dữ liệu.",
    "availability_365": "Số ngày listing còn trống hoặc có thể đặt trong 365 ngày gần nhất.",
    "instant_bookable": "Cho biết listing có thể đặt ngay mà không cần host phê duyệt hay không.",
    "cancellation_policy": "Chính sách hủy đặt phòng áp dụng cho listing.",
    "construction_year": "Năm xây dựng hoặc năm hoàn thành của bất động sản.",
    "house_rules": "Nội quy mà khách phải tuân theo khi ở tại listing.",
    "license": "Mã giấy phép hoặc thông tin cấp phép vận hành listing.",
    "host_identity_verified": "Trạng thái xác minh danh tính của host trên nền tảng.",
    "listing_year": "Năm tham chiếu của listing dùng trong các phân tích theo thời gian.",
    "property_age": "Tuổi của bất động sản, thường được tính từ năm xây dựng.",
    "estimated_revenue": "Doanh thu ước tính của listing dựa trên giá và mức độ khai thác.",
    "occupancy_rate": "Tỷ lệ lấp đầy hoặc mức độ được đặt của listing.",
    "booking_flexibility_score": "Điểm tổng hợp phản ánh độ linh hoạt khi booking.",
    "customer_segment": "Nhóm khách hàng hoặc kiểu lưu trú mà listing phù hợp.",
    "days_since_last_review": "Số ngày tính từ lần đánh giá gần nhất đến mốc phân tích.",
    "price_to_neighborhood_ratio": "Tỷ lệ giữa giá listing và mức giá trung bình của khu vực.",
    "popularity_index": "Chỉ số tổng hợp phản ánh mức độ phổ biến của listing.",
    "booking_friction": "Chỉ số tổng hợp phản ánh mức độ khó hoặc rào cản khi đặt listing.",
}


def _clear_preprocessing_session_state() -> None:
    for key in (
        "preprocessing_before_df",
        "processing_report",
        "processed_df",
        "processed_scaled_df",
        "processed_ml_df",
        "cleaned_data",
    ):
        st.session_state.pop(key, None)


def _normalize_single_column_name(column_name: str) -> str:
    normalized = normalize_columns(pd.DataFrame(columns=[column_name]))
    return str(normalized.columns[0]) if len(normalized.columns) else str(column_name)


def _get_column_meaning(column_name: str) -> str:
    normalized_name = _normalize_single_column_name(str(column_name))
    return COLUMN_MEANINGS_VI.get(
        normalized_name,
        "Trường dữ liệu gốc từ file tải lên. Dashboard hiện chưa có mô tả riêng cho cột này.",
    )


def _build_missing_table_with_meaning(frame: pd.DataFrame) -> pd.DataFrame:
    health_table = build_missing_table(frame).copy()
    health_table.insert(1, "Ý nghĩa", health_table["column"].astype(str).map(_get_column_meaning))
    return health_table


def filter_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    filtered = frame.copy()
    filter_cols = st.columns(4)

    if "neighbourhood_group" in frame.columns:
        area_options = sorted(frame["neighbourhood_group"].dropna().astype(str).unique().tolist())
        selected_areas = filter_cols[0].multiselect(t("raw.filter.neighborhood_group"), area_options)
        if selected_areas:
            filtered = filtered[filtered["neighbourhood_group"].isin(selected_areas)]

    if "room_type" in frame.columns:
        room_options = sorted(frame["room_type"].dropna().astype(str).unique().tolist())
        selected_rooms = filter_cols[1].multiselect(
            t("raw.filter.room_type"),
            room_options,
            format_func=translate_room_type,
        )
        if selected_rooms:
            filtered = filtered[filtered["room_type"].isin(selected_rooms)]

    if "price" in frame.columns and not filtered.empty and not filtered["price"].dropna().empty:
        min_price = float(filtered["price"].min())
        max_price = float(filtered["price"].max())
        if min_price != max_price:
            price_range = filter_cols[2].slider(
                t("raw.filter.price_range"),
                min_price,
                max_price,
                (min_price, max_price),
            )
            filtered = filtered[filtered["price"].between(price_range[0], price_range[1])]

    if "number_of_reviews" in frame.columns and not filtered.empty and not filtered["number_of_reviews"].dropna().empty:
        review_cap = int(filtered["number_of_reviews"].max())
        review_threshold = filter_cols[3].slider(t("raw.filter.minimum_reviews"), 0, review_cap, 0)
        filtered = filtered[filtered["number_of_reviews"] >= review_threshold]

    return filtered


def render_page(raw_frame: pd.DataFrame, cleaned_frame: pd.DataFrame) -> None:
    _ = cleaned_frame
    st.title(t("raw.title"))
    st.caption(t("raw.caption"))

    if not isinstance(raw_frame, pd.DataFrame):
        raw_frame = cleaned_frame.copy() if isinstance(cleaned_frame, pd.DataFrame) else pd.DataFrame()

    session_raw_frame = st.session_state.get("raw_df")
    session_raw_name = st.session_state.get("raw_df_name")
    audit_frame = session_raw_frame.copy() if isinstance(session_raw_frame, pd.DataFrame) else raw_frame.copy()
    uploaded_file = st.file_uploader(
        t("raw.upload.label"),
        type=["csv"],
        help=t("raw.upload.help"),
    )
    if uploaded_file is not None:
        try:
            uploaded_bytes = uploaded_file.getvalue()
            upload_token = hashlib.md5(uploaded_bytes).hexdigest()
            if st.session_state.get("raw_upload_token") != upload_token:
                audit_frame = pd.read_csv(io.BytesIO(uploaded_bytes))
                st.session_state["raw_df"] = audit_frame.copy()
                st.session_state["raw_df_name"] = uploaded_file.name
                st.session_state["raw_upload_token"] = upload_token
                _clear_preprocessing_session_state()
                before_frame, df_cleaned, df_scaled, df_ml_ready, processing_report = run_processing_pipeline(audit_frame)
                store_processed_outputs(before_frame, df_cleaned, df_scaled, df_ml_ready, processing_report)
                st.rerun()
            else:
                audit_frame = session_raw_frame.copy() if isinstance(session_raw_frame, pd.DataFrame) else pd.read_csv(io.BytesIO(uploaded_bytes))
            st.caption(t("raw.source.uploaded", file_name=uploaded_file.name))
        except Exception as exc:
            audit_frame = raw_frame
            st.error(t("raw.upload.error", error=str(exc)))
    elif isinstance(session_raw_frame, pd.DataFrame):
        st.caption(t("raw.source.uploaded", file_name=session_raw_name or "CSV đã tải lên"))
    else:
        st.session_state["raw_df"] = raw_frame.copy()
        st.session_state["raw_df_name"] = None
        st.caption(t("raw.source.default"))

    tab_overview, tab_sort_data = st.tabs(
        [
            t("raw.tab.overview"),
            t("raw.tab.cleaned"),
        ]
    )

    with tab_overview:
        overview_metrics = st.columns(3)
        overview_metrics[0].metric(t("raw.metric.rows"), f"{len(audit_frame):,}")
        overview_metrics[1].metric(t("raw.metric.columns"), f"{audit_frame.shape[1]:,}")
        overview_metrics[2].metric("Tổng giá trị thiếu", f"{int(audit_frame.isna().sum().sum()):,}")

        st.subheader(t("raw.preview.title"))
        st.dataframe(localize_dataframe_for_display(audit_frame.head(50)), use_container_width=True, height=360)

        st.subheader(t("raw.missing.title"))
        health_table = _build_missing_table_with_meaning(audit_frame)
        st.dataframe(
            localize_dataframe_for_display(health_table),
            use_container_width=True,
            hide_index=True,
            height=620,
        )

        null_chart = px.bar(
            health_table.sort_values(["missing_values", "column"], ascending=[False, True]),
            x="missing_values",
            y="column",
            orientation="h",
            color="missing_pct",
            title="Số lượng giá trị thiếu theo từng cột",
            color_continuous_scale=["#f3dcc0", "#c95c36", "#7e3120"],
        )
        null_chart.update_layout(
            coloraxis_colorbar_title_text="Tỷ lệ thiếu (%)",
            margin=dict(l=10, r=10, t=50, b=10),
            yaxis=dict(categoryorder="total ascending"),
        )
        st.plotly_chart(null_chart, use_container_width=True)

        st.markdown("---")
        st.header(t("raw.audit.title"))

        missing_data = (
            health_table.rename(columns={"missing_pct": "missing_percent"})
            .loc[lambda frame: frame["missing_percent"] > 0]
            .sort_values(["missing_percent", "column"], ascending=[False, True])
        )

        fig1 = px.bar(
            missing_data,
            x="column",
            y="missing_percent",
            title="Phần trăm giá trị thiếu theo từng cột",
            labels={"column": "Tên cột", "missing_percent": "Phần trăm (%)"},
            text=missing_data["missing_percent"].apply(lambda x: f"{x:.2f}%"),
            color_discrete_sequence=["#440154"]
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Ma trận giá trị thiếu")
        sample_frame = audit_frame.sample(n=min(len(audit_frame), 1000), random_state=42).sort_index()
        null_matrix = sample_frame.isnull().astype(int)
        fig2 = px.imshow(
            null_matrix,
            labels=dict(x="Tên cột", y="Số dòng dữ liệu", color="Thiếu dữ liệu"),
            color_continuous_scale=["#440154", "#fde725"]
        )
        fig2.update_traces(hovertemplate='Cột: %{x}<br>Dòng: %{y}<br>Thiếu dữ liệu: %{z}')
        fig2.update_layout(
            title_text="Ma trận giá trị thiếu",
        )
        st.plotly_chart(fig2, use_container_width=True)

        numerical_cols = [
            "price", "minimum_nights", "number_of_reviews", "service_fee",
            "reviews_per_month", "calculated_host_listings_count", "availability_365"
        ]
        available_numerical_cols = [col for col in numerical_cols if col in audit_frame.columns and pd.api.types.is_numeric_dtype(audit_frame[col])]

        if available_numerical_cols:
            melted_frame = audit_frame[available_numerical_cols].melt()
            fig3 = px.box(
                melted_frame,
                x="value",
                y="variable",
                title="Boxplot nhận diện ngoại lệ",
                labels={"variable": "Cột", "value": "Giá trị"},
                orientation="h",
                color="variable",
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Không tìm thấy cột số để nhận diện ngoại lệ.")
    
    with tab_sort_data:
        st.subheader(t("raw.cleaned.title"))
        processed_frame = st.session_state.get("processed_df")
        if not isinstance(processed_frame, pd.DataFrame):
            st.info("Hãy tải CSV để chạy pipeline tiền xử lý tự động. Phần chi tiết nằm ở tab Tiền xử lý trên sidebar.")
            return

        filtered = filter_dataframe(processed_frame)
        st.metric(t("raw.metric.filtered_rows"), f"{len(filtered):,}")
        st.dataframe(localize_dataframe_for_display(filtered), use_container_width=True, height=420)
