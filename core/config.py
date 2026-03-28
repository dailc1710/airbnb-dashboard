from __future__ import annotations

from pathlib import Path
from textwrap import dedent

APP_TITLE = "Airbnb Analytics Dashboard"
DATASET_PATH = Path("data/Airbnb_Open_Data.csv")
SAMPLE_SOURCE_LABEL = "Bundled sample data"
NAVIGATION_PAGES = [
    "overview",
    "data_raw",
    "preprocessing",
    "eda",
    "conclusion",
    "chatbot",
]
CHART_COLORS = ["#1f3c5b", "#c95c36", "#d8a65d", "#6d8f71"]

PREPROCESSING_PIPELINE_STEPS = [
    {
        "title": "A.I. Làm sạch dữ liệu",
        "summary": dedent(
            """
            - Chuẩn hóa tên cột: lowercase, trim khoảng trắng thừa, thay ký tự đặc biệt bằng dấu `_`.
            - Xóa khoảng trắng thừa ở tất cả cột dạng chuỗi.
            - Xóa emoji, ký tự lạ, HTML tags và HTML entities.
            - Xóa ký tự `$` và dấu `,` trong `price` và `service_fee`.
            """
        ).strip(),
        "code": dedent(
            """
            df = normalize_columns(df)

            string_cols = df.select_dtypes(include=["object", "string"]).columns
            df[string_cols] = df[string_cols].apply(lambda col: col.astype("string").str.strip())

            for col in string_cols:
                if col not in {"price", "service_fee"}:
                    df[col] = (
                        df[col]
                        .astype("string")
                        .str.replace(r"<[^>]+>", " ", regex=True)
                        .str.replace(r"&[a-zA-Z0-9#]+;", " ", regex=True)
                        .str.replace(r"[\\U00010000-\\U0010ffff]", " ", regex=True)
                        .str.replace(r"[^\\w\\s/&.,'\\-]", " ", regex=True)
                        .str.replace(r"\\s+", " ", regex=True)
                        .str.strip()
                    )

            for col in ["price", "service_fee"]:
                if col in df.columns:
                    df[col] = (
                        df[col]
                        .astype("string")
                        .str.replace("$", "", regex=False)
                        .str.replace(",", "", regex=False)
                    )
            """
        ).strip(),
    },
    {
        "title": "A.II. Xử lý dữ liệu trùng lặp",
        "summary": dedent(
            """
            - Xóa các dòng trùng lặp dựa trên `id`.
            - Giữ lại bản ghi đầu tiên cho mỗi `id`.
            """
        ).strip(),
        "code": dedent(
            """
            df = df.drop_duplicates(subset=["id"], keep="first")
            """
        ).strip(),
    },
    {
        "title": "A.III. Chọn đặc trưng",
        "summary": dedent(
            """
            - Xóa các cột không dùng trong phân tích: `id`, `name`, `host_name`, `lat`, `long`,
              `country`, `country_code`, `service_fee`, `reviews_per_month`, `house_rules`, `license`,
              `instant_bookable`, `cancellation_policy`.
            - Mục tiêu là giảm khối lượng dữ liệu và số cột phải xử lý trong project.
            """
        ).strip(),
        "code": dedent(
            """
            df = df.drop(
                columns=[
                    "id", "name", "host_name", "lat", "long", "country",
                    "country_code", "service_fee", "reviews_per_month",
                    "house_rules", "license", "instant_bookable",
                    "cancellation_policy"
                ],
                errors="ignore",
            )
            """
        ).strip(),
    },
    {
        "title": "A.IV. Chuyển đổi kiểu dữ liệu",
        "summary": dedent(
            """
            - Chuyển các cột numeric chính về `float64`.
            - Chuyển `construction_year` và `last_review` sang `datetime`.
            - Chuyển các cột chuỗi về lowercase.
            - Chuẩn hóa lỗi chính tả `brookln -> brooklyn`, `manhatan -> manhattan`.
            """
        ).strip(),
        "code": dedent(
            """
            float_cols = [
                "price", "minimum_nights", "number_of_reviews",
                "review_rate_number", "calculated_host_listings_count",
                "availability_365"
            ]
            for col in float_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

            df["construction_year"] = pd.to_datetime(
                df["construction_year"].astype("string").str.extract(r"(\\d{4})", expand=False),
                format="%Y",
                errors="coerce",
            )
            df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

            lowercase_cols = [
                "host_id", "host_identity_verified", "neighbourhood_group",
                "neighbourhood", "room_type"
            ]
            for col in lowercase_cols:
                df[col] = df[col].astype("string").str.lower().str.strip()

            df["neighbourhood_group"] = df["neighbourhood_group"].replace({"brookln": "brooklyn", "manhatan": "manhattan"})
            """
        ).strip(),
    },
    {
        "title": "A.V.1. Xử lý giá trị thiếu - Dữ liệu phân loại",
        "summary": dedent(
            """
            - `host_identity_verified`: điền `unconfirmed`.
            - `neighbourhood_group`: ánh xạ từ `neighbourhood`.
            - `neighbourhood`: điền mode theo `neighbourhood_group + room_type`.
            """
        ).strip(),
        "code": dedent(
            """
            df["host_identity_verified"] = df["host_identity_verified"].fillna("unconfirmed")

            neighbourhood_group_map = (
                df.loc[df["neighbourhood"].notna() & df["neighbourhood_group"].notna(), ["neighbourhood", "neighbourhood_group"]]
                .groupby("neighbourhood")["neighbourhood_group"]
                .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA)
                .to_dict()
            )
            df["neighbourhood_group"] = df["neighbourhood_group"].fillna(df["neighbourhood"].map(neighbourhood_group_map))
            df["neighbourhood"] = df["neighbourhood"].fillna(
                df.groupby(["neighbourhood_group", "room_type"])["neighbourhood"].transform(
                    lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA
                )
            )
            """
        ).strip(),
    },
    {
        "title": "A.V.2. Xử lý giá trị thiếu - Ngày giờ",
        "summary": dedent(
            """
            - `construction_year`: điền mode theo `neighbourhood + room_type`.
            - `last_review`: giá trị lớn hơn `2022-12-31` được xem là không hợp lệ và đổi thành missing.
            - `last_review`: xóa toàn bộ dòng bị thiếu vì đây là biến nhạy cảm và không phù hợp để nội suy.
            """
        ).strip(),
        "code": dedent(
            """
            df["construction_year"] = df["construction_year"].fillna(
                df.groupby(["neighbourhood", "room_type"])["construction_year"].transform(
                    lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NaT
                )
            )
            df.loc[df["last_review"] > pd.Timestamp("2022-12-31"), "last_review"] = pd.NaT
            df = df.dropna(subset=["last_review"]).copy()
            """
        ).strip(),
    },
    {
        "title": "A.V.3. Xử lý giá trị thiếu - Dữ liệu số",
        "summary": dedent(
            """
            - `price`: mean theo `neighbourhood + room_type`.
            - `minimum_nights`: lấy trị tuyệt đối, đổi giá trị `> 365` thành missing, rồi điền median theo nhóm.
            - `number_of_reviews`: điền median theo `neighbourhood + room_type`.
            - `review_rate_number`: điền mode theo `neighbourhood + room_type`.
            - `calculated_host_listings_count`: điền theo tần suất xuất hiện của `host_id`, fallback `1` nếu `host_id` thiếu.
            - `availability_365`: lấy trị tuyệt đối, đổi giá trị `> 365` thành missing, rồi điền median theo nhóm.
            """
        ).strip(),
        "code": dedent(
            """
            df["price"] = df["price"].fillna(df.groupby(["neighbourhood", "room_type"])["price"].transform("mean"))

            df["minimum_nights"] = df["minimum_nights"].abs()
            df.loc[df["minimum_nights"] > 365, "minimum_nights"] = pd.NA
            df["minimum_nights"] = df["minimum_nights"].fillna(
                df.groupby(["neighbourhood", "room_type"])["minimum_nights"].transform("median")
            )

            df["number_of_reviews"] = df["number_of_reviews"].fillna(
                df.groupby(["neighbourhood", "room_type"])["number_of_reviews"].transform("median")
            )
            df["review_rate_number"] = df["review_rate_number"].fillna(
                df.groupby(["neighbourhood", "room_type"])["review_rate_number"].transform(
                    lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA
                )
            )

            host_frequency = df.groupby("host_id")["host_id"].transform("size")
            df["calculated_host_listings_count"] = (
                df["calculated_host_listings_count"]
                .fillna(host_frequency)
                .fillna(1.0)
            )

            df["availability_365"] = df["availability_365"].abs()
            df.loc[df["availability_365"] > 365, "availability_365"] = pd.NA
            df["availability_365"] = df["availability_365"].fillna(
                df.groupby(["neighbourhood", "room_type"])["availability_365"].transform("median")
            )
            """
        ).strip(),
    },
    {
        "title": "A.VI. Xử lý ngoại lệ",
        "summary": dedent(
            """
            - `price`: Percentile Capping ở ngưỡng 1% và 99%.
            - `minimum_nights`: đổi `0 -> 1`, sau đó dùng IQR Capping.
            - `number_of_reviews`: IQR Capping.
            - `review_rate_number`: `clip(0, 5)`.
            - `calculated_host_listings_count`: IQR Capping.
            - `availability_365`: `clip(0, 365)`.
            - Skewness được tính trước để giải thích vì sao mỗi cột dùng một chiến lược khác nhau.
            - Sau outlier handling, các cột dạng đếm/ngày được làm tròn để tránh số lẻ khó đọc.
            """
        ).strip(),
        "code": dedent(
            """
            skew_profile = {col: float(df[col].skew()) for col in [
                "price", "minimum_nights", "number_of_reviews",
                "review_rate_number", "calculated_host_listings_count",
                "availability_365"
            ]}

            df["price"] = df["price"].clip(
                lower=df["price"].quantile(0.01),
                upper=df["price"].quantile(0.99),
            )

            df["minimum_nights"] = df["minimum_nights"].replace(0, 1)
            q1 = df["minimum_nights"].quantile(0.25)
            q3 = df["minimum_nights"].quantile(0.75)
            iqr = q3 - q1
            df["minimum_nights"] = df["minimum_nights"].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

            for col in ["number_of_reviews", "calculated_host_listings_count"]:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

            df["review_rate_number"] = df["review_rate_number"].clip(0, 5)
            df["availability_365"] = df["availability_365"].clip(0, 365)

            for col in [
                "minimum_nights",
                "number_of_reviews",
                "review_rate_number",
                "calculated_host_listings_count",
                "availability_365",
            ]:
                df[col] = df[col].round().astype("float64")
            """
        ).strip(),
    },
    {
        "title": "B. Tải file preprocessing",
        "summary": dedent(
            """
            - Xuất file `Airbnb_Data_cleaned.csv` sau toàn bộ bước preprocessing.
            - Ngoài ra dashboard vẫn hỗ trợ tải thêm bản `scaled` cho visualization và `encoded` cho học máy.
            """
        ).strip(),
        "code": dedent(
            """
            df.to_csv("data/Airbnb_Data_cleaned.csv", index=False)
            df_scaled.to_csv("data/Airbnb_Data_scaled.csv", index=False)
            encoded_df.to_csv("data/Airbnb_Data_encoded.csv", index=False)
            """
        ).strip(),
    },
    {
        "title": "C. Tạo đặc trưng",
        "summary": dedent(
            """
            - `Booking Demand (Nhu cầu đặt phòng)`
              Logic: tính toán nhu cầu đặt phòng từ số đêm không sẵn có.
              Cách thực hiện: `booking_demand = 365 - availability_365`
            - `Availability Category (Phân loại mức độ sẵn có)`
              Logic: chia listing thành 3 nhóm Low / Medium / High Availability dựa trên số đêm còn trống.
              Cách thực hiện:
              `availability_category = pd.cut(availability_365, bins=[-1, 150, 300, 365], labels=['Low Availability', 'Medium Availability', 'High Availability'])`
            - `Availability Efficiency (Hiệu quả sẵn có)`
              Logic: đo hiệu quả khai thác dựa trên giá và số đêm đã được đặt.
              Cách thực hiện: `availability_efficiency = price * (365 - availability_365)`
            - `Revenue per Available Night (Doanh thu mỗi đêm có sẵn)`
              Logic: quy đổi doanh thu ước tính về trung bình trên mỗi đêm khả dụng trong năm.
              Cách thực hiện: `revenue_per_available_night = price * (365 - availability_365) / 365`
            """
        ).strip(),
        "code": dedent(
            """
            df["booking_demand"] = 365 - df["availability_365"]
            df["availability_category"] = pd.cut(
                df["availability_365"],
                bins=[-1, 150, 300, 365],
                labels=[
                    "Low Availability",
                    "Medium Availability",
                    "High Availability",
                ],
                include_lowest=True,
                right=True,
            )
            df["availability_efficiency"] = df["price"] * (365 - df["availability_365"])
            df["revenue_per_available_night"] = df["availability_efficiency"] / 365
            """
        ).strip(),
    },
    {
        "title": "D. Chuẩn hóa cho trực quan hóa",
        "summary": dedent(
            """
            - File `scaled` vẫn dùng `MinMaxScaler` để phục vụ dashboard visualization.
            - Chỉ scale các cột numeric dùng để so sánh trực quan trên cùng thang đo.
            - Không scale các feature mới:
              `booking_demand`, `availability_efficiency`, `revenue_per_available_night`.
            - `availability_category` không scale, vì đây là biến thứ bậc và sẽ được ordinal encoding ở file ML-ready.
            - Gợi ý tham chiếu cho modeling:
              `price` -> `RobustScaler` hoặc `log1p + StandardScaler`,
              `minimum_nights`, `number_of_reviews`, `calculated_host_listings_count` -> `RobustScaler`,
              `review_rate_number` -> thường có thể giữ nguyên.

            Hiện tại dashboard export file `scaled` bằng `MinMaxScaler`; các lựa chọn còn lại được giữ như phương án tham chiếu cho modeling.
            """
        ).strip(),
        "code": dedent(
            """
            numeric_cols = [
                col for col in df.select_dtypes(include="number").columns
                if col not in {
                    "booking_demand",
                    "availability_efficiency",
                    "revenue_per_available_night",
                }
            ]

            minmax_scaler = MinMaxScaler()
            df_scaled = df.copy()
            df_scaled[numeric_cols] = minmax_scaler.fit_transform(df_scaled[numeric_cols])

            # Keep raw engineered metrics for interpretation:
            # - booking_demand
            # - availability_efficiency
            # - revenue_per_available_night
            # availability_category is handled by ordinal encoding in the ML-ready export
            """
        ).strip(),
    },
    {
        "title": "E. Trực quan hóa",
        "summary": dedent(
            """
            - Pie Chart: tỷ lệ `availability_category` để đọc trạng thái cung của thị trường.
            - Boxplot: phân bổ `booking_demand` theo `neighbourhood_group` để xem cầu theo khu vực.
            - Scatter Plot: `price` vs `booking_demand`, kèm đường xu hướng để kiểm tra mức độ nhạy cảm về giá.
            - Heatmap: trung bình `availability_efficiency` theo `room_type` và `neighbourhood_group` để tìm "sweet spot" kinh doanh.
            - 4 biểu đồ được đọc theo logic Cung -> Cầu -> Giá -> Hiệu quả.
            """
        ).strip(),
        "code": dedent(
            """
            availability_mix = (
                df["availability_category"]
                .value_counts(normalize=True)
                .rename_axis("availability_category")
                .reset_index(name="share")
            )

            booking_demand_by_area = df.groupby("neighbourhood_group", as_index=False)["booking_demand"].median()

            scatter_df = df.loc[df["price"].between(0, 1200), ["price", "booking_demand"]].dropna()

            efficiency_heatmap = df.pivot_table(
                index="room_type",
                columns="neighbourhood_group",
                values="availability_efficiency",
                aggfunc="mean",
            )
            """
        ).strip(),
    },
    {
        "title": "F. Mã hóa cho học máy",
        "summary": dedent(
            """
            - Giữ nguyên `host_id` vì đây là cột định danh phục vụ theo dõi host, không phải biến phân loại cần label encoding.
            - Mã hóa nhị phân `host_identity_verified`: `unconfirmed -> 0`, `verified -> 1`.
            - Label Encoding cho `neighbourhood_group` và `neighbourhood` vì đây là các cột phân loại nhiều mức, nếu one-hot sẽ làm số cột tăng rất mạnh.
            - One-Hot Encoding cho `room_type` vì đây là biến phân loại danh nghĩa ít mức và không có thứ bậc.
            - Giữ nguyên các biến numeric như `price`, `minimum_nights`, `number_of_reviews`,
              `review_rate_number`, `calculated_host_listings_count`, `availability_365`.
            - `construction_year` được chuyển về năm số.
            - `last_review` được chuyển thành `days_since_last_review`, sau đó loại bỏ cột ngày gốc.
            - Các cột tạo mới từ Feature Engineering vẫn được giữ trong file ML-ready:
              `booking_demand`, `availability_efficiency`, `revenue_per_available_night`,
              còn `availability_category` được ordinal encoding với `Low=0`, `Medium=1`, `High=2`.
            """
        ).strip(),
        "code": dedent(
            """
            encoded_df = df.copy()
            encoded_df["host_identity_verified"] = encoded_df["host_identity_verified"].map({"unconfirmed": 0, "verified": 1})

            def label_encode_text(series, fill_value="unknown"):
                normalized = (
                    series.astype("string")
                    .str.strip()
                    .str.lower()
                    .fillna(fill_value)
                )
                categories = sorted(normalized.unique().tolist())
                mapping = {value: index for index, value in enumerate(categories)}
                return normalized.map(mapping).astype("int64")

            encoded_df["neighbourhood_group"] = label_encode_text(encoded_df["neighbourhood_group"])
            encoded_df["neighbourhood"] = label_encode_text(encoded_df["neighbourhood"])
            encoded_df["construction_year"] = pd.to_datetime(
                encoded_df["construction_year"],
                errors="coerce",
            ).dt.year
            encoded_df["days_since_last_review"] = (
                pd.Timestamp.today().normalize() - pd.to_datetime(encoded_df["last_review"])
            ).dt.days
            encoded_df = encoded_df.drop(columns=["last_review"])
            encoded_df["availability_category"] = encoded_df["availability_category"].map({
                "Low Availability": 0,
                "Medium Availability": 1,
                "High Availability": 2,
            })
            encoded_df = pd.get_dummies(
                encoded_df,
                columns=["room_type"],
                drop_first=True,
                prefix_sep="__",
                dtype="int64",
            )
            """
        ).strip(),
    },
]
