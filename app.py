import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

# Thư viện Đồ họa
import plotly.express as px

# Thư viện Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ==========================================
# 1. CẤU HÌNH & GIAO DIỆN CSS
# ==========================================
st.set_page_config(page_title="Hệ thống Định giá Xe máy AI", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stButton>button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #FF4B4B; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. XỬ LÝ DỮ LIỆU TỪ FILE CSV
# ==========================================
@st.cache_data
def load_and_preprocess():
    file_path = "motorbike_data.csv"
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip() # Xóa khoảng trắng tên cột
    current_year = datetime.now().year

    # --- Làm sạch Giá bán ---
    def clean_price(val):
        try:
            s = str(val).replace('.', '').replace(',', '').strip()
            # Xử lý trường hợp dư số 0 (ví dụ 19.800.000.00)
            if len(s) > 8: return float(s) / 100
            return float(s)
        except: return np.nan

    df['Giá bán'] = df['Giá bán'].apply(clean_price)
    
    # --- Làm sạch các cột khác ---
    df['Tình trạng xe'] = df['Tình trạng xe'].astype(str).str.replace(',', '.').astype(float)
    df['Phụ tùng'] = df['đã phụ tùng chưa thay'].map({'đã thay': 1, 'chưa thay': 0}).fillna(0)
    df['Tuổi xe'] = current_year - df['Năm sản xuất']
    
    df = df.dropna(subset=['Giá bán', 'Hãng xe', 'Dòng xe'])
    
    # --- Loại bỏ Outliers (Giá ảo) ---
    Q1 = df['Giá bán'].quantile(0.25)
    Q3 = df['Giá bán'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['Giá bán'] >= Q1 - 1.5 * IQR) & (df['Giá bán'] <= Q3 + 1.5 * IQR)]
    
    return df

# ==========================================
# 3. HUẤN LUYỆN MÔ HÌNH AI
# ==========================================
@st.cache_resource
def train_model(df):
    features = ["Hãng xe", "Dòng xe", "Tuổi xe", "Số km đã chạy", "Tình trạng xe", "Phụ tùng", "Khu vực bán"]
    X = df[features]
    y = df["Giá bán"]
    
    cat_features = ["Hãng xe", "Dòng xe", "Khu vực bán"]
    num_features = ["Tuổi xe", "Số km đã chạy", "Tình trạng xe", "Phụ tùng"]
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])
    
    pipeline = Pipeline([
        ('pre', preprocessor),
        ('reg', RandomForestRegressor(n_estimators=200, random_state=42))
    ])
    
    # Train
    pipeline.fit(X, y)
    
    # Tính toán sai số cơ bản
    y_pred = pipeline.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    return pipeline, r2, mae

# --- KHỞI CHẠY HỆ THỐNG ---
df = load_and_preprocess()
current_year = datetime.now().year

if df is not None:
    model_pl, r2, mae = train_model(df)
    
    st.title("🏍️ AI Motorbike Intelligence - Định Giá & Phân Tích")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Dự báo giá xe", "📊 Dashboard Phân tích", "📂 Dữ liệu gốc"])

    # ==========================================
    # TAB 1: DỰ BÁO GIÁ (CHỌN XE THÔNG MINH)
    # ==========================================
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📝 Nhập thông tin xe")
            
            # 1. Chọn Hãng xe (Lấy từ file)
            list_hang = sorted(df["Hãng xe"].unique())
            brand = st.selectbox("Hãng xe", list_hang)
            
            # 2. Chọn Dòng xe (Lọc theo Hãng đã chọn)
            list_dong = sorted(df[df["Hãng xe"] == brand]["Dòng xe"].unique())
            model_bike = st.selectbox("Dòng xe", list_dong)
            
            # 3. Các thông số khác
            y_col, k_col = st.columns(2)
            year = y_col.number_input("Năm sản xuất", 2000, current_year, current_year - 2)
            km = k_col.number_input("Số km đã đi", 0, 500000, 10000, step=1000)
            
            cond = st.slider("Tình trạng ngoại hình (1-10)", 1.0, 10.0, 8.0)
            part = st.radio("Tình trạng máy móc", ["chưa thay", "đã thay"], horizontal=True)
            area = st.selectbox("Khu vực bán", sorted(df["Khu vực bán"].unique()))
            
            predict_btn = st.button("🚀 TÍNH GIÁ DỰ ĐOÁN")

        with col2:
            st.subheader("💰 Kết quả định giá")
            if predict_btn:
                with st.spinner("Đang phân tích dữ liệu thị trường..."):
                    time.sleep(0.4)
                    input_data = pd.DataFrame([{
                        "Hãng xe": brand, "Dòng xe": model_bike, "Tuổi xe": current_year - year,
                        "Số km đã chạy": km, "Tình trạng xe": cond, 
                        "Phụ tùng": 1 if part == "đã thay" else 0, "Khu vực bán": area
                    }])
                    
                    res = model_pl.predict(input_data)[0]
                    
                    st.success(f"### Giá dự báo: {res:,.0f} VNĐ")
                    
                    # Tính giá trung bình của dòng xe này trong file để so sánh
                    avg_model_price = df[df['Dòng xe'] == model_bike]['Giá bán'].mean()
                    diff = res - avg_model_price
                    st.metric("So với trung bình dòng xe", f"{res:,.0f}đ", f"{diff:,.0f}đ")
                    
                    st.info(f"💡 Độ tin cậy: {r2*100:.1f}% | Sai số trung bình: {mae:,.0f}đ")

    # ==========================================
    # TAB 2: DASHBOARD PHÂN TÍCH
    # ==========================================
    with tab2:
        st.header("📊 Insight Thị Trường")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Tổng số mẫu xe", f"{len(df)} xe")
        m2.metric("Giá TB thị trường", f"{df['Giá bán'].mean():,.0f}đ")
        m3.metric("Hãng xe phổ biến", df['Hãng xe'].mode()[0])

        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.histogram(df, x="Giá bán", color="Hãng xe", title="Phân bổ mức giá theo hãng", nbins=30)
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.scatter(df, x="Tuổi xe", y="Giá bán", color="Hãng xe", trendline="lowess", title="Xu hướng khấu hao theo tuổi xe")
            st.plotly_chart(fig2, use_container_width=True)

    # ==========================================
    # TAB 3: DỮ LIỆU
    # ==========================================
    with tab3:
        st.subheader("Chi tiết dữ liệu trong hệ thống")
        st.dataframe(df, use_container_width=True)

else:
    st.error("⚠️ Lỗi: Không tìm thấy file 'motorbike_data.csv'. Vui lòng kiểm tra lại!")
