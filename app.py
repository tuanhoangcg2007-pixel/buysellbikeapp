import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Scikit-learn chuyên sâu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ==========================================
# CẤU HÌNH TRANG & GIAO DIỆN
# ==========================================
st.set_page_config(page_title="AI Motorbike Price Predictor", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #FF4B4B; color: white; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# HÀM XỬ LÝ DỮ LIỆU (THEO ĐÚNG ĐỀ BÀI)
# ==========================================
@st.cache_data
def load_and_preprocess_data(file_path):
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path)
    
    # 1. Xử lý Giá bán (Xử lý định dạng 19.800.000.00 hoặc 30.800.000,00)
    def clean_price(price_str):
        price_str = str(price_str).replace('.', '').replace(',', '')
        # Nếu có .00 ở cuối, việc xóa dấu chấm sẽ khiến số bị nhân lên 100 lần
        # Dữ liệu của bạn thường có 2 số không ở cuối đại diện cho phần thập phân
        return float(price_str) / 100

    df['Giá bán'] = df['Giá bán'].apply(clean_price)
    
    # 2. Xử lý Tình trạng xe (8,5 -> 8.5)
    df['Tình trạng xe'] = df['Tình trạng xe'].astype(str).str.replace(',', '.').astype(float)
    
    # 3. Xử lý Phụ tùng (đã thay -> 1, chưa thay -> 0)
    df['Phụ tùng'] = df['đã phụ tùng chưa thay'].map({'đã thay': 1, 'chưa thay': 0}).fillna(0)
    
    # 4. Loại bỏ các cột không cần thiết cho huấn luyện
    cols_to_keep = ["Hãng xe", "Dòng xe", "Năm sản xuất", "Số km đã chạy", "Tình trạng xe", "Phụ tùng", "Khu vực bán", "Giá bán"]
    return df[cols_to_keep].dropna()

# ==========================================
# XÂY DỰNG PIPELINE MÔ HÌNH CHUYÊN NGHIỆP
# ==========================================
@st.cache_resource
def build_model(df):
    X = df.drop("Giá bán", axis=1)
    y = df["Giá bán"]
    
    # Phân loại cột
    categorical_features = ["Hãng xe", "Dòng xe", "Khu vực bán"]
    numeric_features = ["Năm sản xuất", "Số km đã chạy", "Tình trạng xe", "Phụ tùng"]
    
    # Tiền xử lý: OneHot cho chữ, Scaler cho số
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Tạo Pipeline (Gồm tiền xử lý và thuật toán Random Forest - mạnh hơn Linear Regression đơn thuần)
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
    ])
    
    # Huấn luyện
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)
    
    # Đánh giá
    y_pred = model_pipeline.predict(X_test)
    score = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return model_pipeline, score, mae

# ==========================================
# THỰC THI APP
# ==========================================
# Ưu tiên đọc file tên motorbike_data.csv theo yêu cầu mới nhất
data_file = "motorbike_data.csv" if os.path.exists("motorbike_data.csv") else "[VQH] Dữ liệu giá bán xe máy - Trang tính1.csv"
df = load_and_preprocess_data(data_file)

if df is not None:
    model_pl, r2, mae = build_model(df)
    
    # --- HEADER ---
    st.title("🏍️ Đồ án: App Dự Đoán Giá Xe Máy Cũ")
    st.info(f"Dữ liệu huấn luyện: {len(df)} mẫu | Độ chính xác: {r2:.2f} | Sai số TB: {mae:,.0f} VNĐ")

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["🎯 Định giá xe", "📊 Phân tích thị trường", "⚙️ Dữ liệu nguồn"])

    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Thông tin xe của bạn")
            brand = st.selectbox("Hãng xe", sorted(df["Hãng xe"].unique()))
            # Lọc dòng xe theo hãng (Tính năng Pro)
            models_by_brand = sorted(df[df["Hãng xe"] == brand]["Dòng xe"].unique())
            model_bike = st.selectbox("Dòng xe", models_by_brand)
            
            year = st.number_input("Năm sản xuất", 2000, 2026, 2022)
            km = st.number_input("Số km đã chạy", 0, 500000, 15000)
            
            condition = st.slider("Tình trạng xe (1: Rất cũ - 10: Như mới)", 1.0, 10.0, 8.5)
            parts = st.radio("Đã thay phụ tùng chưa?", ["chưa thay", "đã thay"], horizontal=True)
            area = st.selectbox("Khu vực bán", sorted(df["Khu vực bán"].unique()))
            
            predict_btn = st.button("🚀 TÍNH GIÁ BÁN NGAY")

        with col2:
            st.subheader("Kết quả dự báo AI")
            if predict_btn:
                # Tạo dataframe cho input để Pipeline xử lý
                input_df = pd.DataFrame([{
                    "Hãng xe": brand,
                    "Dòng xe": model_bike,
                    "Năm sản xuất": year,
                    "Số km đã chạy": km,
                    "Tình trạng xe": condition,
                    "Phụ tùng": 1 if parts == "đã thay" else 0,
                    "Khu vực bán": area
                }])
                
                prediction = model_pl.predict(input_df)[0]
                
                st.success(f"## {prediction:,.0f} VNĐ")
                st.write("---")
                st.markdown(f"""
                **Phân tích của hệ thống:**
                - Giá xe giảm trung bình theo năm: **{year}**
                - Khấu hao qua quãng đường: **{km:,} km**
                - Điểm tình trạng: **{condition}/10**
                """)
                
                # Biểu đồ so sánh nhỏ
                avg_price = df[df["Dòng xe"] == model_bike]["Giá bán"].mean()
                st.write(f"💡 Giá trung bình dòng {model_bike} hiện nay: {avg_price:,.0f}đ")

    with tab2:
        st.subheader("Trực quan hóa dữ liệu (Insights)")
        c1, c2 = st.columns(2)
        
        with c1:
            st.write("**Phân bổ giá theo hãng xe**")
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x="Hãng xe", y="Giá bán", ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
        with c2:
            st.write("**Tương quan: Năm sản xuất vs Giá bán**")
            fig2, ax2 = plt.subplots()
            sns.scatterplot(data=df, x="Năm sản xuất", y="Giá bán", hue="Hãng xe", ax=ax2)
            st.pyplot(fig2)

    with tab3:
        st.subheader("Dữ liệu đã thu thập (n >= 200)")
        st.dataframe(df, use_container_width=True)

else:
    st.error("❌ Không tìm thấy file dữ liệu (motorbike_data.csv). Vui lòng kiểm tra lại!")