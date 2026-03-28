from __future__ import annotations

import streamlit as st

from core.i18n import t

ACHIEVEMENT_POINTS = [
    "Hoàn thành các mục tiêu tiền xử lý quan trọng như xử lý giá trị thiếu, loại bỏ hoặc kiểm soát ngoại lệ, và mã hóa dữ liệu phân loại để dữ liệu trở nên nhất quán hơn.",
    "Các bước làm sạch và chuẩn hóa giúp nâng cao chất lượng bộ dữ liệu, giảm nhiễu trong phân tích và tạo nền tảng tốt hơn cho các mô hình học máy ở giai đoạn tiếp theo.",
    "Toàn bộ pipeline đã biến dữ liệu thô thành dữ liệu sẵn sàng cho EDA, trực quan hóa và trích xuất insight phục vụ cho việc ra quyết định.",
]

BENEFIT_POINTS = [
    "Ứng dụng giúp đơn giản hóa quy trình xử lý dữ liệu bằng cách gom các bước clean data, feature engineering, visualization và kết luận vào cùng một dashboard.",
    "Cách trình bày trực quan giúp quy trình phân tích trở nên dễ tiếp cận hơn đối với sinh viên, người mới học dữ liệu và cả những ai cần đọc nhanh insight kinh doanh.",
    "Nhờ đó, người dùng có thể tập trung nhiều hơn vào việc diễn giải kết quả thay vì phải xử lý thủ công từng bước trên nhiều công cụ khác nhau.",
]

INSIGHT_GROUPS = [
    {
        "title": "Yếu tố Vị trí và Loại hình phòng",
        "weight": "Trọng tâm nhất - Khoảng 40%",
        "summary": 'Đây là yếu tố then chốt dẫn đến kết luận rằng thị trường cạnh tranh bằng "sự phù hợp" thay vì chỉ cạnh tranh bằng giá cả.',
        "evidence": 'Biểu đồ Heatmap chỉ ra các "điểm ngọt" (sweet spot) cụ thể như Hotel Room tại Brooklyn hoặc Shared Room tại Staten Island mang lại hiệu quả cao nhất.',
        "meaning": "Kết luận chiến lược là nhà đầu tư nên chọn đúng cặp bài trùng giữa khu vực và loại hình lưu trú để tối ưu hiệu quả khai thác.",
    },
    {
        "title": "Yếu tố Cầu (Booking Demand)",
        "weight": "Khoảng 25%",
        "summary": 'Yếu tố này xác định các khu vực "vàng" có khả năng duy trì dòng khách ổn định.',
        "evidence": "Biểu đồ Boxplot cho thấy Brooklyn và Manhattan là hai khu vực dẫn đầu về nhu cầu đặt phòng với mức trung vị rất cao, gần 300 đêm.",
        "meaning": "Điều này củng cố nhận định rằng khách hàng ưu tiên vị trí trung tâm và khả năng tiếp cận hơn là chỉ nhìn vào mức giá.",
    },
    {
        "title": "Yếu tố Giá (Price)",
        "weight": "Khoảng 20%",
        "summary": 'Yếu tố này đóng vai trò loại trừ giả thuyết rằng thị trường chủ yếu cạnh tranh bằng giá thấp.',
        "evidence": "Biểu đồ Scatter cho thấy đường hồi quy gần như nằm ngang, minh chứng rằng giá không tỷ lệ nghịch rõ rệt với nhu cầu trong phân khúc dưới 1,200 USD.",
        "meaning": "Từ đó có thể rút ra rằng khách hàng không quá nhạy cảm với giá, nên chiến lược cạnh tranh cần dịch chuyển sang tiện ích, vị trí và độ phù hợp của sản phẩm.",
    },
    {
        "title": "Yếu tố Cung (Availability)",
        "weight": "Khoảng 15%",
        "summary": "Yếu tố này phản ánh trạng thái sức khỏe chung của thị trường lưu trú.",
        "evidence": '59.5% niêm yết thuộc nhóm "Low Availability", cho thấy nhiều phòng thường xuyên có khách đặt.',
        "meaning": 'Kết quả này cho thấy thị trường đang ở trạng thái hấp thụ tốt, tạo cơ hội cho các căn hộ mới hoặc các phân khúc ngách nếu được định vị đúng.',
    },
]


def _inject_conclusion_styles() -> None:
    st.markdown(
        """
        <style>
            .conclusion-section {
                margin-bottom: 1rem;
                padding: 1.05rem 1.15rem;
                border: 1px solid rgba(31, 60, 91, 0.08);
                border-radius: 18px;
                background: rgba(255, 255, 255, 0.78);
            }
            .conclusion-section h3 {
                margin: 0 0 0.72rem;
                color: #223247;
                font-size: 1.06rem;
            }
            .conclusion-section ul {
                margin: 0;
                padding-left: 1.2rem;
                color: #546071;
                line-height: 1.72;
            }
            .conclusion-insight-card {
                margin-bottom: 0.95rem;
                padding: 1rem 1.1rem;
                border: 1px solid rgba(31, 60, 91, 0.08);
                border-radius: 18px;
                background: rgba(255, 255, 255, 0.76);
            }
            .conclusion-insight-card h4 {
                margin: 0 0 0.32rem;
                color: #223247;
                font-size: 1rem;
            }
            .conclusion-insight-weight {
                margin-bottom: 0.72rem;
                color: #6b7280;
                font-size: 0.86rem;
                font-weight: 600;
            }
            .conclusion-insight-card ul {
                margin: 0;
                padding-left: 1.2rem;
                color: #546071;
                line-height: 1.72;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_point_section(title: str, points: list[str]) -> None:
    items = "".join(f"<li>{point}</li>" for point in points)
    st.markdown(
        f"""
        <section class="conclusion-section">
            <h3>{title}</h3>
            <ul>{items}</ul>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_insight_card(index: int, group: dict[str, str]) -> None:
    st.markdown(
        f"""
        <section class="conclusion-insight-card">
            <h4>{index}. {group["title"]}</h4>
            <div class="conclusion-insight-weight">{group["weight"]}</div>
            <ul>
                <li>{group["summary"]}</li>
                <li><strong>Dẫn chứng:</strong> {group["evidence"]}</li>
                <li><strong>Ý nghĩa:</strong> {group["meaning"]}</li>
            </ul>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_page(_frame) -> None:
    _inject_conclusion_styles()
    st.title(t("conclusion.title"))

    _render_point_section("Tóm tắt các thành tựu chính", ACHIEVEMENT_POINTS)
    _render_point_section("Nhấn mạnh lợi ích của ứng dụng", BENEFIT_POINTS)

    st.subheader("Đúc kết từ các insight")
    for index, group in enumerate(INSIGHT_GROUPS, start=1):
        _render_insight_card(index, group)
