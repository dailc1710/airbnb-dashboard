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
        "title": "A.I. Data Cleaning",
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
        "title": "A.II. Handling Duplicates",
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
        "title": "A.III. Feature Selection",
        "summary": dedent(
            """
            - Xóa các cột không dùng trong phân tích: `id`, `name`, `host_name`, `lat`, `long`,
              `country`, `country_code`, `service_fee`, `reviews_per_month`, `house_rules`, `license`.
            - Mục tiêu là giảm khối lượng dữ liệu và số cột phải xử lý trong project.
            """
        ).strip(),
        "code": dedent(
            """
            df = df.drop(
                columns=[
                    "id", "name", "host_name", "lat", "long", "country",
                    "country_code", "service_fee", "reviews_per_month",
                    "house_rules", "license"
                ],
                errors="ignore",
            )
            """
        ).strip(),
    },
    {
        "title": "A.IV. Data Type Conversion",
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
                "neighbourhood", "instant_bookable", "cancellation_policy", "room_type"
            ]
            for col in lowercase_cols:
                df[col] = df[col].astype("string").str.lower().str.strip()

            df["neighbourhood_group"] = df["neighbourhood_group"].replace({"brookln": "brooklyn", "manhatan": "manhattan"})
            """
        ).strip(),
    },
    {
        "title": "A.V.1. Handling Missing Values - Categorical",
        "summary": dedent(
            """
            - `host_identity_verified`: điền `unconfirmed`.
            - `neighbourhood_group`: ánh xạ từ `neighbourhood`.
            - `neighbourhood`: điền mode theo `neighbourhood_group + room_type`.
            - `instant_bookable`: điền `false`.
            - `cancellation_policy`: điền mode theo `neighbourhood + room_type`.
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
            df["instant_bookable"] = df["instant_bookable"].fillna("false")
            df["cancellation_policy"] = df["cancellation_policy"].fillna(
                df.groupby(["neighbourhood", "room_type"])["cancellation_policy"].transform(
                    lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA
                )
            )
            """
        ).strip(),
    },
    {
        "title": "A.V.2. Handling Missing Values - Date-time",
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
        "title": "A.V.3. Handling Missing Values - Numerical",
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
        "title": "A.VI. Handling Outliers",
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
        "title": "B. Download Preprocessing File",
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
        "title": "C. Feature Engineering",
        "summary": dedent(
            """
            - `listing_year = year(last_review)`
            - `property_age = 2022 - construction_year`
            - `estimated_revenue = (365 - availability_365) * price`
            - `occupancy_rate = (365 - availability_365) / 365`
            - `booking_flexibility_score`:
              `instant_bookable(true=1, false=0)` + `cancellation_policy(flexible=2, moderate=1, strict=0)`
            - `customer_segment`: gom nhóm `minimum_nights` thành 3 tệp khách hàng.
            """
        ).strip(),
        "code": dedent(
            """
            df["listing_year"] = df["last_review"].dt.year.astype("float64")
            construction_year_value = df["construction_year"].dt.year.astype("float64")
            df["property_age"] = (2022 - construction_year_value).clip(lower=0)
            df["estimated_revenue"] = (365 - df["availability_365"]) * df["price"]
            df["occupancy_rate"] = (365 - df["availability_365"]) / 365
            df["booking_flexibility_score"] = (
                df["instant_bookable"].map({"true": 1, "false": 0}).fillna(0)
                + df["cancellation_policy"].map({"flexible": 2, "moderate": 1, "strict": 0}).fillna(0)
            )
            df["customer_segment"] = pd.cut(
                df["minimum_nights"],
                bins=[0, 3, 7, float("inf")],
                labels=[
                    "short stay (1-3 nights)",
                    "business/leisure (4-7 nights)",
                    "long stay (>7 nights)",
                ],
                include_lowest=True,
            )
            """
        ).strip(),
    },
    {
        "title": "D. Scaling for Visualization",
        "summary": dedent(
            """
            1. `MinMaxScaler`: đưa toàn bộ numeric feature về khoảng 0-1, dễ so sánh trên biểu đồ.
            2. `StandardScaler`: phù hợp khi muốn xem biến lệch khỏi mean bao nhiêu.
            3. `RobustScaler`: phù hợp khi còn nhiều outlier.
            4. `Log Scaling`: thường áp dụng trực tiếp lên trục biểu đồ cho dữ liệu lệch mạnh.

            Hiện tại dashboard export file `scaled` bằng `MinMaxScaler`; 3 lựa chọn còn lại được giữ như phương án tham chiếu cho visualization.
            """
        ).strip(),
        "code": dedent(
            """
            numeric_cols = df.select_dtypes(include="number").columns

            minmax_scaler = MinMaxScaler()
            df_scaled = df.copy()
            df_scaled[numeric_cols] = minmax_scaler.fit_transform(df_scaled[numeric_cols])

            # Alternative references for visualization:
            # - StandardScaler: compare deviation from mean (z-score)
            # - RobustScaler: stable when outliers are still present
            # - Log scaling: usually applied on chart axis, not by replacing raw data
            """
        ).strip(),
    },
    {
        "title": "E. Visualization",
        "summary": dedent(
            """
            - Multi-line chart: giá trung bình theo `listing_year` và `neighbourhood_group` trong giai đoạn 2012-2022.
            - Multi-line chart: `occupancy_rate` theo `listing_year` và `neighbourhood_group` trong giai đoạn 2012-2022.
            - Bar chart: doanh thu trung bình theo `listing_year`.
            - Grouped bar chart: doanh thu trung bình theo `listing_year` và `neighbourhood_group`.
            - Bar chart: doanh thu trung bình theo `customer_segment`.
            - Correlation heatmap: `occupancy_rate`, `price`, `booking_flexibility_score`, `review_rate_number`.
            """
        ).strip(),
        "code": dedent(
            """
            price_trend = (
                df.loc[df["listing_year"].between(2012, 2022)]
                .groupby(["listing_year", "neighbourhood_group"], as_index=False)["price"]
                .mean()
            )
            occupancy_trend = (
                df.loc[df["listing_year"].between(2012, 2022)]
                .groupby(["listing_year", "neighbourhood_group"], as_index=False)["occupancy_rate"]
                .mean()
            )
            revenue_year = df.groupby("listing_year", as_index=False)["estimated_revenue"].mean()
            revenue_area_year = df.groupby(["listing_year", "neighbourhood_group"], as_index=False)["estimated_revenue"].mean()
            revenue_segment = df.groupby("customer_segment", as_index=False)["estimated_revenue"].mean()
            corr = df[["occupancy_rate", "price", "booking_flexibility_score", "review_rate_number"]].corr()
            """
        ).strip(),
    },
    {
        "title": "F. Encoding for Machine Learning",
        "summary": dedent(
            """
            - Giữ nguyên `host_id`.
            - Label encode: `host_identity_verified`, `neighbourhood_group`, `neighbourhood`, `instant_bookable`, `cancellation_policy`.
            - One-hot encode: `room_type`, `customer_segment`.
            - Giữ nguyên các biến numeric.
            - Chuyển `last_review` thành `days_since_last_review`.
            - Sau khi one-hot, chuẩn hóa lại tên cột để bỏ khoảng trắng và ký tự gây khó chịu khi code.
            """
        ).strip(),
        "code": dedent(
            """
            encoded_df = df.copy()
            encoded_df["host_identity_verified"] = encoded_df["host_identity_verified"].map({"unconfirmed": 0, "verified": 1})
            encoded_df["neighbourhood_group"] = encoded_df["neighbourhood_group"].astype("category").cat.codes
            encoded_df["neighbourhood"] = encoded_df["neighbourhood"].astype("category").cat.codes
            encoded_df["instant_bookable"] = encoded_df["instant_bookable"].map({"false": 0, "true": 1})
            encoded_df["cancellation_policy"] = encoded_df["cancellation_policy"].map({"strict": 0, "moderate": 1, "flexible": 2})
            encoded_df["days_since_last_review"] = (pd.Timestamp("2022-12-31") - pd.to_datetime(encoded_df["last_review"])).dt.days
            encoded_df = encoded_df.drop(columns=["last_review"])
            encoded_df = pd.get_dummies(
                encoded_df,
                columns=["room_type", "customer_segment"],
                drop_first=True,
                dtype="int64",
            )
            encoded_df = normalize_columns(encoded_df)
            """
        ).strip(),
    },
]
