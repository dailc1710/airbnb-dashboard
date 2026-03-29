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
from core.i18n import get_language, localize_dataframe_for_display, t, translate_room_type
from pages.preprocessing import run_processing_pipeline, store_processed_outputs

COLUMN_MEANINGS = {
    "id": {"en": "Unique identifier for each listing.", "vi": "Mã định danh duy nhất của mỗi listing."},
    "name": {"en": "Listing title shown on Airbnb.", "vi": "Tên hiển thị của listing trên Airbnb."},
    "host_id": {"en": "Identifier of the host account.", "vi": "Mã định danh của chủ nhà (host)."},
    "host_name": {"en": "Name of the host.", "vi": "Tên của chủ nhà."},
    "neighbourhood_group": {
        "en": "Large area or borough where the listing is located, such as Manhattan or Brooklyn.",
        "vi": "Khu vực lớn hoặc quận nơi listing tọa lạc, ví dụ Manhattan hoặc Brooklyn.",
    },
    "neighbourhood": {
        "en": "Specific neighborhood inside the neighbourhood_group.",
        "vi": "Khu phố hoặc địa bàn cụ thể của listing bên trong neighbourhood_group.",
    },
    "lat": {"en": "Latitude of the listing location.", "vi": "Vĩ độ của vị trí listing."},
    "long": {"en": "Longitude of the listing location.", "vi": "Kinh độ của vị trí listing."},
    "country": {"en": "Country of the listing.", "vi": "Quốc gia của listing."},
    "country_code": {"en": "Short country code of the listing.", "vi": "Mã quốc gia viết tắt của listing."},
    "room_type": {
        "en": "Accommodation type being rented, for example Entire home/apt or Private room.",
        "vi": "Loại chỗ ở được cho thuê, ví dụ Entire home/apt hoặc Private room.",
    },
    "price": {"en": "Listed price of the listing, typically per night.", "vi": "Giá niêm yết của listing, thường là theo đêm."},
    "service_fee": {"en": "Service fee attached to the booking.", "vi": "Phí dịch vụ đi kèm với booking."},
    "minimum_nights": {"en": "Minimum number of nights guests must book.", "vi": "Số đêm tối thiểu khách phải đặt."},
    "number_of_reviews": {"en": "Total number of reviews received by the listing.", "vi": "Tổng số lượt đánh giá mà listing đã nhận."},
    "reviews_per_month": {"en": "Average number of reviews per month.", "vi": "Số lượt đánh giá trung bình mỗi tháng."},
    "last_review": {"en": "Most recent review date for the listing.", "vi": "Ngày gần nhất listing nhận được đánh giá."},
    "review_rate_number": {"en": "Overall review score, usually on a 1-5 scale.", "vi": "Điểm đánh giá tổng quan của listing, thường theo thang điểm 1-5."},
    "calculated_host_listings_count": {
        "en": "Number of listings owned or managed by the same host in the dataset.",
        "vi": "Số listing mà cùng một host đang sở hữu hoặc quản lý trong dữ liệu.",
    },
    "availability_365": {
        "en": "Number of days the listing is available within the latest 365 days.",
        "vi": "Số ngày listing còn trống hoặc có thể đặt trong 365 ngày gần nhất.",
    },
    "instant_bookable": {
        "en": "Whether the listing can be booked instantly without host approval.",
        "vi": "Cho biết listing có thể đặt ngay mà không cần host phê duyệt hay không.",
    },
    "cancellation_policy": {"en": "Cancellation policy applied to the listing.", "vi": "Chính sách hủy đặt phòng áp dụng cho listing."},
    "construction_year": {"en": "Year when the property was built or completed.", "vi": "Năm xây dựng hoặc năm hoàn thành của bất động sản."},
    "house_rules": {"en": "House rules guests must follow during the stay.", "vi": "Nội quy mà khách phải tuân theo khi ở tại listing."},
    "license": {"en": "License code or operating permit information for the listing.", "vi": "Mã giấy phép hoặc thông tin cấp phép vận hành listing."},
    "host_identity_verified": {"en": "Verification status of the host identity on the platform.", "vi": "Trạng thái xác minh danh tính của host trên nền tảng."},
    "listing_year": {"en": "Reference year used for time-based analysis.", "vi": "Năm tham chiếu của listing dùng trong các phân tích theo thời gian."},
    "property_age": {"en": "Age of the property, typically derived from construction year.", "vi": "Tuổi của bất động sản, thường được tính từ năm xây dựng."},
    "estimated_revenue": {"en": "Estimated listing revenue based on price and utilization.", "vi": "Doanh thu ước tính của listing dựa trên giá và mức độ khai thác."},
    "occupancy_rate": {"en": "Occupancy rate or booking intensity of the listing.", "vi": "Tỷ lệ lấp đầy hoặc mức độ được đặt của listing."},
    "booking_flexibility_score": {"en": "Composite score reflecting booking flexibility.", "vi": "Điểm tổng hợp phản ánh độ linh hoạt khi booking."},
    "customer_segment": {"en": "Customer segment or stay pattern that fits the listing.", "vi": "Nhóm khách hàng hoặc kiểu lưu trú mà listing phù hợp."},
    "days_since_last_review": {"en": "Number of days from the latest review to the analysis date.", "vi": "Số ngày tính từ lần đánh giá gần nhất đến mốc phân tích."},
    "price_to_neighborhood_ratio": {"en": "Ratio between listing price and the neighborhood average price.", "vi": "Tỷ lệ giữa giá listing và mức giá trung bình của khu vực."},
    "popularity_index": {"en": "Composite index reflecting listing popularity.", "vi": "Chỉ số tổng hợp phản ánh mức độ phổ biến của listing."},
    "booking_friction": {"en": "Composite index reflecting booking difficulty or barriers.", "vi": "Chỉ số tổng hợp phản ánh mức độ khó hoặc rào cản khi đặt listing."},
}

RAW_TEXT = {
    "unknown_column": {
        "en": "Original field from the uploaded file. The dashboard does not have a dedicated description for this column yet.",
        "vi": "Trường dữ liệu gốc từ file tải lên. Dashboard hiện chưa có mô tả riêng cho cột này.",
    },
    "uploaded_csv_fallback": {"en": "Uploaded CSV", "vi": "CSV đã tải lên"},
    "total_missing_values": {"en": "Total missing values", "vi": "Tổng giá trị thiếu"},
    "missing_count_by_column": {"en": "Missing values by column", "vi": "Số lượng giá trị thiếu theo từng cột"},
    "missing_pct_axis": {"en": "Missing rate (%)", "vi": "Tỷ lệ thiếu (%)"},
    "missing_pct_by_column": {"en": "Missing rate by column", "vi": "Phần trăm giá trị thiếu theo từng cột"},
    "column_name": {"en": "Column name", "vi": "Tên cột"},
    "missing_matrix": {"en": "Missing-value matrix", "vi": "Ma trận giá trị thiếu"},
    "data_row_count": {"en": "Data rows", "vi": "Số dòng dữ liệu"},
    "missing_data": {"en": "Missing data", "vi": "Thiếu dữ liệu"},
    "missing_matrix_hover": {"en": "Column: %{x}<br>Row: %{y}<br>Missing data: %{z}", "vi": "Cột: %{x}<br>Dòng: %{y}<br>Thiếu dữ liệu: %{z}"},
    "outlier_boxplot": {"en": "Outlier detection boxplot", "vi": "Boxplot nhận diện ngoại lệ"},
    "value": {"en": "Value", "vi": "Giá trị"},
    "no_numeric_outlier": {"en": "No numeric columns were found for outlier detection.", "vi": "Không tìm thấy cột số để nhận diện ngoại lệ."},
    "upload_first_notice": {
        "en": "Upload a CSV to run the preprocessing pipeline automatically. Detailed results are available in the Preprocessing tab on the sidebar.",
        "vi": "Hãy tải CSV để chạy pipeline tiền xử lý tự động. Phần chi tiết nằm ở tab Tiền xử lý trên sidebar.",
    },
    "no_csv_uploaded": {
        "en": "No CSV has been uploaded in this session yet. Upload a file above to start the dashboard workflow.",
        "vi": "Bạn chưa tải CSV nào trong session này. Hãy tải tệp ở phía trên để bắt đầu luồng dashboard.",
    },
}


def _raw_text(key: str, **kwargs: object) -> str:
    lang = get_language()
    template = RAW_TEXT.get(key, {}).get(lang) or RAW_TEXT.get(key, {}).get("en") or key
    if kwargs:
        return template.format(**kwargs)
    return template


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
    meaning = COLUMN_MEANINGS.get(normalized_name, {})
    return meaning.get(get_language()) or meaning.get("en") or _raw_text("unknown_column")


def _build_missing_table_with_meaning(frame: pd.DataFrame) -> pd.DataFrame:
    health_table = build_missing_table(frame).copy()
    health_table.insert(1, t("meaning"), health_table["column"].astype(str).map(_get_column_meaning))
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
        st.caption(t("raw.source.uploaded", file_name=session_raw_name or _raw_text("uploaded_csv_fallback")))
    else:
        st.session_state["raw_df"] = raw_frame.copy()
        st.session_state["raw_df_name"] = None
        st.caption(_raw_text("no_csv_uploaded"))

    if audit_frame.empty:
        st.info(_raw_text("no_csv_uploaded"))
        return

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
        overview_metrics[2].metric(_raw_text("total_missing_values"), f"{int(audit_frame.isna().sum().sum()):,}")

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
            title=_raw_text("missing_count_by_column"),
            color_continuous_scale=["#f3dcc0", "#c95c36", "#7e3120"],
        )
        null_chart.update_layout(
            coloraxis_colorbar_title_text=_raw_text("missing_pct_axis"),
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
            title=_raw_text("missing_pct_by_column"),
            labels={"column": _raw_text("column_name"), "missing_percent": _raw_text("missing_pct_axis")},
            text=missing_data["missing_percent"].apply(lambda x: f"{x:.2f}%"),
            color_discrete_sequence=["#440154"]
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader(_raw_text("missing_matrix"))
        sample_frame = audit_frame.sample(n=min(len(audit_frame), 1000), random_state=42).sort_index()
        null_matrix = sample_frame.isnull().astype(int)
        fig2 = px.imshow(
            null_matrix,
            labels=dict(x=_raw_text("column_name"), y=_raw_text("data_row_count"), color=_raw_text("missing_data")),
            color_continuous_scale=["#440154", "#fde725"]
        )
        fig2.update_traces(hovertemplate=_raw_text("missing_matrix_hover"))
        fig2.update_layout(
            title_text=_raw_text("missing_matrix"),
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
                title=_raw_text("outlier_boxplot"),
                labels={"variable": t("column"), "value": _raw_text("value")},
                orientation="h",
                color="variable",
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info(_raw_text("no_numeric_outlier"))
    
    with tab_sort_data:
        st.subheader(t("raw.cleaned.title"))
        processed_frame = st.session_state.get("processed_df")
        if not isinstance(processed_frame, pd.DataFrame):
            st.info(_raw_text("upload_first_notice"))
            return

        filtered = filter_dataframe(processed_frame)
        st.metric(t("raw.metric.filtered_rows"), f"{len(filtered):,}")
        st.dataframe(localize_dataframe_for_display(filtered), use_container_width=True, height=420)
