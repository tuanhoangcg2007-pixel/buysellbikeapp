import streamlit as st
import pandas as pd
import numpy as np
import os
import time

# Thư viện Đồ họa & Trực quan hóa
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Thư viện Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ==========================================
# CẤU HÌNH TRANG
# ==========================================
st.set_page_config(page_title="AI Motorbike Price Predictor", layout="wide")

# Tùy chỉnh giao diện bằng CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stButton>button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #FF4B4B; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# XỬ LÝ DỮ LIỆU
# ==========================================
@st.cache_data
def load_and_preprocess(file_path):
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path)
    
    # Làm sạch Giá bán: Xử lý các dạng 19.800.000.00 hoặc 30.800.000,00
    def clean_price(val):
        s = str(val).replace('.', '').replace(',', '')
        return float(s) / 100  # Chia 100 vì dữ liệu gốc thường dư 2 số 0 ở cuối

    df['Giá bán'] = df['Giá bán'].apply(clean_price)
    
    # Làm sạch Tình trạng xe: 8,5 -> 8.5
    df['Tình trạng xe'] = df['Tình trạng xe'].astype(str).str.replace(',', '.').astype(float)
    
    # Làm sạch Phụ tùng
    df['Phụ tùng'] = df['đã phụ tùng chưa thay'].map({'đã thay': 1, 'chưa thay': 0}).fillna(0)
    
    return df.dropna(subset=['Giá bán'])

# ==========================================
# HUẤN LUYỆN MÔ HÌNH (PIPELINE)
# ==========================================
@st.cache_resource
def train_model(df):
    features = ["Hãng xe", "Dòng xe", "Năm sản xuất", "Số km đã chạy", "Tình trạng xe", "Phụ tùng", "Khu vực bán"]
    X = df[features]
    y = df["Giá bán"]
    
    cat_features = ["Hãng xe", "Dòng xe", "Khu vực bán"]
    num_features = ["Năm sản xuất", "Số km đã chạy", "Tình trạng xe", "Phụ tùng"]
    
    # Tiền xử lý tự động
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])
    
    # Pipeline chuyên nghiệp
    pipeline = Pipeline([
        ('pre', preprocessor),
        ('reg', RandomForestRegressor(n_estimators=200, random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    return pipeline, r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred)

# --- THỰC THI ---
data_file = "motorbike_data.csv" if os.path.exists("motorbike_data.csv") else "[VQH] Dữ liệu giá bán xe máy - Trang tính1.csv"
df = load_and_preprocess(data_file)

if df is not None:
    model_pl, r2, mae = train_model(df)
    
    st.title("🏍️ AI Motorbike Intelligence - Dự báo & Phân tích")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Định giá xe AI", "📊 Dashboard Phân tích", "📂 Dữ liệu thô"])

    # --- TAB 1: DỰ ĐOÁN GIÁ ---
    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Nhập thông tin xe")
            brand = st.selectbox("Hãng xe", sorted(df["Hãng xe"].unique()))
            models = sorted(df[df["Hãng xe"] == brand]["Dòng xe"].unique())
            model_bike = st.selectbox("Dòng xe", models)
            
            y_col, k_col = st.columns(2)
            year = y_col.number_input("Năm sản xuất", 2000, 2026, 2023)
            km = k_col.number_input("Số km đã đi", 0, 500000, 10000)
            
            cond = st.slider("Tình trạng xe (1-10)", 1.0, 10.0, 8.5)
            part = st.radio("Phụ tùng", ["chưa thay", "đã thay"], horizontal=True)
            area = st.selectbox("Khu vực bán", sorted(df["Khu vực bán"].unique()))
            
            predict_btn = st.button("🚀 TÍNH GIÁ DỰ ĐOÁN")

        with col2:
            st.subheader("Kết quả định giá")
            if predict_btn:
                with st.spinner("Đang tính toán..."):
                    time.sleep(0.5)
                    input_data = pd.DataFrame([{
                        "Hãng xe": brand, "Dòng xe": model_bike, "Năm sản xuất": year,
                        "Số km đã chạy": km, "Tình trạng xe": cond, 
                        "Phụ tùng": 1 if part == "đã thay" else 0, "Khu vực bán": area
                    }])
                    res = model_pl.predict(input_data)[0]
                    st.success(f"## {res:,.0f} VNĐ")
                    st.info(f"Độ tin cậy của mô hình: {r2*100:.1f}%")

    # --- TAB 2: PHÂN TÍCH THỊ TRƯỜNG (BẢN PRO) ---
    with tab2:
        st.header("📊 Báo Cáo Phân Tích Thị Trường")
        
        # Chỉ số tổng quan
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Quy mô mẫu", f"{len(df)} xe")
        m2.metric("Giá TB", f"{df['Giá bán'].mean():,.0f}đ")
        m3.metric("Dòng xe phổ biến", df['Dòng xe'].mode()[0])
        m4.metric("Sai số TB (MAE)", f"{mae:,.0f}đ")

        st.write("##")

        # Biểu đồ 1 & 2
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📌 Phân bổ giá thị trường")
            fig_dist = px.histogram(df, x="Giá bán", color="Hãng xe", marginal="box", 
                                   title="Phân bổ mức giá theo Hãng", nbins=40, color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_dist, use_container_width=True)

        with c2:
            st.subheader("📌 Thị phần theo Giá trị")
            fig_tree = px.treemap(df, path=['Hãng xe', 'Dòng xe'], values='Giá bán',
                                 color='Giá bán', color_continuous_scale='RdBu')
            st.plotly_chart(fig_tree, use_container_width=True)

        # Biểu đồ 3 & 4
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("📉 Khấu hao: Năm SX vs Giá")
            fig_scatter = px.scatter(df, x="Năm sản xuất", y="Giá bán", color="Hãng xe", 
                                    size="Số km đã chạy", hover_data=['Dòng xe'])
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with c4:
            st.subheader("📍 Mặt bằng giá theo Khu vực")
            area_df = df.groupby("Khu vực bán")["Giá bán"].mean().reset_index()
            fig_area = px.bar(area_df, x="Khu vực bán", y="Giá bán", color="Giá bán", color_continuous_scale='Viridis')
            st.plotly_chart(fig_area, use_container_width=True)

    # --- TAB 3: DỮ LIỆU ---
    with tab3:
        st.subheader("Danh sách dữ liệu thu thập (n >= 200)")
        st.dataframe(df, use_container_width=True)

else:
    st.error("⚠️ Không tìm thấy file dữ liệu! Vui lòng kiểm tra file CSV.")
