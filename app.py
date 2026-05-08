import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor # Dùng GradientBoosting cho độ chính xác cao hơn RF
from sklearn.metrics import r2_score

# ==========================================
# 1. GIAO DIỆN (Giữ nguyên phong cách e Tuấn)
# ==========================================
st.set_page_config(page_title="Tìm e Tuấn - Định Giá Xe", page_icon="🏍️", layout="wide")

SĐT_ZALO = "0838088267" 
LINK_ZALO = f"https://zalo.me/0838088267"

st.markdown(f"""
    <style>
    .main {{ background-color: #f4f7f9; }}
    .price-card {{
        background: #1e1e1e; padding: 40px; border-radius: 20px;
        text-align: center; border-left: 10px solid #FF4B4B;
    }}
    .zalo-button {{
        display: inline-block; padding: 15px 25px; background-color: #0068ff;
        color: white !important; text-decoration: none; border-radius: 10px;
        font-weight: bold; text-align: center; width: 100%; margin-top: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. FIX LOGIC XỬ LÝ (QUAN TRỌNG NHẤT)
# ==========================================
@st.cache_data
def load_and_preprocess():
    file_path = "motorbike_data.csv"
    if not os.path.exists(file_path): return None
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    current_year = datetime.now().year

    def clean_price(val):
        try:
            # Xóa mọi ký tự không phải số
            s = ''.join(filter(str.isdigit, str(val)))
            if not s: return np.nan
            num = float(s)
            
            # XỬ LÝ LỖI DƯ SỐ 0: 
            # Nếu số > 500 triệu (thường là do .00 ở cuối), chia cho 100
            # Xe máy cũ bình thường hiếm khi trên 500tr
            if num > 500000000: num = num / 100
            # Nếu số vẫn còn quá lớn (ví dụ 300 triệu mà là Wave), cần lọc bỏ sau
            return num
        except: return np.nan

    df['Giá bán'] = df['Giá bán'].apply(clean_price)
    
    # Xử lý tình trạng xe (8,5 -> 8.5)
    df['Tình trạng xe'] = df['Tình trạng xe'].astype(str).str.replace(',', '.').apply(pd.to_numeric, errors='coerce')
    
    # Xử lý phụ tùng
    df['Phụ tùng'] = df['đã phụ tùng chưa thay'].map({'đã thay': 1, 'chưa thay': 0}).fillna(0)
    
    # Tính tuổi xe
    df['Tuổi xe'] = current_year - df['Năm sản xuất']
    
    # --- BƯỚC LỌC DỮ LIỆU RÁC (OUTLIERS) ---
    # Loại bỏ xe giá < 1 triệu hoặc > 300 triệu (Tránh giá ảo)
    df = df[(df['Giá bán'] > 1000000) & (df['Giá bán'] < 300000000)]
    
    # Loại bỏ các dòng thiếu thông tin quan trọng
    df = df.dropna(subset=['Giá bán', 'Hãng xe', 'Dòng xe', 'Tình trạng xe'])
    
    return df

@st.cache_resource
def train_model(df):
    features = ["Hãng xe", "Dòng xe", "Tuổi xe", "Số km đã chạy", "Tình trạng xe", "Phụ tùng", "Khu vực bán"]
    X = df[features]
    # Dùng Log để AI không bị sốc bởi các xe quá đắt
    y = np.log1p(df["Giá bán"]) 
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ["Tuổi xe", "Số km đã chạy", "Tình trạng xe", "Phụ tùng"]),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ["Hãng xe", "Dòng xe", "Khu vực bán"])
    ])
    
    # GradientBoosting thường dự đoán giá chính xác hơn RandomForest với tập dữ liệu nhỏ
    model = Pipeline([
        ('pre', preprocessor), 
        ('reg', GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=5, random_state=42))
    ])
    
    model.fit(X, y)
    return model, r2_score(y, model.predict(X))

# --- CHẠY APP ---
df = load_and_preprocess()
current_year = datetime.now().year

if df is not None:
    model_pl, r2 = train_model(df)
    
    # SIDEBAR
    with st.sidebar:
        st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>HÃY TÌM E TUẤN</h1>", unsafe_allow_html=True)
        st.success(f"✅ AI Đã Khử Nhiễu Dữ Liệu\nĐộ tin cậy: {r2*100:.1f}%")
        st.write(f"Tổng số xe trong kho dữ liệu: {len(df)}")

    # MAIN CONTENT
    st.markdown("# 🏍️ Muốn Bán xe ư? Hãy tìm e Tuấn")
    
    tab1, tab2 = st.tabs(["🎯 Định Giá Ngay", "📊 Phân Tích Thị Trường"])

    with tab1:
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("### 📝 Nhập thông tin xe")
            brand = st.selectbox("Hãng xe", sorted(df["Hãng xe"].unique()))
            model_bike = st.selectbox("Dòng xe", sorted(df[df["Hãng xe"] == brand]["Dòng xe"].unique()))
            
            c1, c2 = st.columns(2)
            year = c1.number_input("Năm sản xuất", 2000, current_year, current_year-1)
            km = c2.number_input("Số km đã đi", 0, 500000, 10000)
            
            cond = st.slider("Tình trạng thực tế (1-10)", 1.0, 10.0, 8.5)
            part = st.radio("Máy móc / Phụ tùng", ["chưa thay", "đã thay"], horizontal=True)
            area = st.selectbox("Khu vực", sorted(df["Khu vực bán"].unique()))
            
            btn = st.button("🚀 XEM GIÁ TỪ E TUẤN")

        with col2:
            st.markdown("### 💰 Kết quả định giá")
            if btn:
                with st.spinner("E Tuấn đang quét giá thị trường..."):
                    input_data = pd.DataFrame([{
                        "Hãng xe": brand, "Dòng xe": model_bike, "Tuổi xe": current_year - year,
                        "Số km đã chạy": km, "Tình trạng xe": cond, 
                        "Phụ tùng": 1 if part == "đã thay" else 0, "Khu vực bán": area
                    }])
                    # Dự đoán và chuyển ngược từ Log về giá trị thật
                    res_log = model_pl.predict(input_data)[0]
                    res = np.expm1(res_log)
                    
                    st.balloons()
                    st.markdown(f"""
                        <div class="price-card">
                            <h3 style="color: white; margin: 0;">GIÁ THỊ TRƯỜNG ĐỀ XUẤT</h3>
                            <h1 style="color: #FF4B4B; font-size: 50px; margin: 10px 0;">{res:,.0f} VNĐ</h1>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f'<a href="{LINK_ZALO}" class="zalo-button">📱 CHỐT XE QUA ZALO</a>', unsafe_allow_html=True)
    
    with tab2:
        st.subheader("📈 Phân tích giá trị khấu hao (Dữ liệu đã sạch)")
        fig = px.scatter(df, x="Số km đã chạy", y="Giá bán", color="Hãng xe", 
                         hover_data=["Dòng xe", "Năm sản xuất"], 
                         title="Sự ảnh hưởng của ODO đến giá bán")
        st.plotly_chart(fig, use_container_width=True)
