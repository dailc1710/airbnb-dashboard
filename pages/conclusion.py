from __future__ import annotations

import streamlit as st

from core.i18n import t

ACHIEVEMENT_POINTS = [
    "Nêu bật việc hoàn thành các mục tiêu lớn, chẳng hạn như xử lý giá trị thiếu, loại bỏ ngoại lệ và mã hóa dữ liệu phân loại. Đề cập đến tác động của từng bước đến chất lượng bộ dữ liệu và sự sẵn sàng của nó cho các phân tích hoặc mô hình học máy tiếp theo.",
]

BENEFIT_POINTS = [
    "Bạn có thể tập trung vào cách ứng dụng giúp đơn giản hóa quy trình xử lý dữ liệu, làm cho nó trở nên dễ tiếp cận hơn cho các nhà khoa học dữ liệu và sinh viên.",
]

INSIGHT_GROUPS = [
    {
        "title": "Yếu tố Vị trí và Loại hình phòng",
        "weight": "Trọng tâm nhất - Khoảng 40%",
        "summary": 'Đây là yếu tố then chốt dẫn đến kết luận rằng thị trường cạnh tranh bằng "sự phù hợp" thay vì giá cả.',
        "evidence": 'Biểu đồ Heatmap chỉ ra các "điểm ngọt" (sweet spot) cụ thể như Hotel Room tại Brooklyn hoặc Shared Room tại Staten Island mang lại hiệu quả cao nhất.',
        "meaning": "Kết luận chiến lược tập trung vào việc nhà đầu tư nên chọn đúng cặp bài trùng giữa khu vực và loại hình lưu trú.",
    },
    {
        "title": "Yếu tố Cầu (Booking Demand)",
        "weight": "Khoảng 25%",
        "summary": 'Yếu tố này xác định các khu vực "vàng" để đảm bảo dòng khách ổn định.',
        "evidence": "Biểu đồ Boxplot xác định Brooklyn và Manhattan là hai khu vực dẫn đầu về nhu cầu đặt phòng với mức trung vị rất cao (gần 300 đêm).",
        "meaning": "Củng cố cho kết luận rằng khách hàng ưu tiên vị trí trung tâm bất kể giá cả.",
    },
    {
        "title": "Yếu tố Giá (Price)",
        "weight": "Khoảng 20%",
        "summary": 'Yếu tố này đóng vai trò "loại trừ" giả thuyết cạnh tranh bằng giá.',
        "evidence": "Biểu đồ Scatter cho thấy đường hồi quy nằm ngang, minh chứng rằng giá không tỷ lệ nghịch với nhu cầu trong phân khúc dưới 1200 USD.",
        "meaning": "Giúp rút ra nhận định quan trọng là khách hàng không quá nhạy cảm về giá, từ đó dịch chuyển chiến lược sang tập trung vào tiện ích và vị trí.",
    },
    {
        "title": "Yếu tố Cung (Availability)",
        "weight": "Khoảng 15%",
        "summary": "Yếu tố này xác định trạng thái sức khỏe của thị trường.",
        "evidence": '59.5% niêm yết thuộc nhóm "Low Availability", cho thấy phòng thường xuyên có khách.',
        "meaning": 'Dẫn đến kết luận thị trường đang ở trạng thái "khát" phòng (High occupancy), tạo cơ hội cho các căn hộ mới hoặc các phân khúc ngách.',
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
