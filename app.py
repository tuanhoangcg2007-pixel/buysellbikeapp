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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ==========================================
# 1. CẤU HÌNH & GIAO DIỆN (UI/UX)
# ==========================================
st.set_page_config(page_title="Tìm e Tuấn - Định Giá Xe", page_icon="🏍️", layout="wide")

# Cấu hình thông tin cá nhân ở đây để dễ thay đổi
SĐT_ZALO = "0901234567"  # <-- TUẤN THAY SỐ ĐIỆN THOẠI CỦA TUẤN VÀO ĐÂY
LINK_ZALO = f"https://zalo.me/{SĐT_ZALO}"

st.markdown(f"""
    <style>
    .main {{ background-color: #f4f7f9; }}
    [data-testid="stSidebar"] {{ background-color: #1e1e1e; }}
    [data-testid="stSidebar"] * {{ color: white !important; }}

    .st-emotion-cache-1r6slb0, .st-emotion-cache-12w0qpk {{
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }}

    /* Nút bấm e Tuấn */
    div.stButton > button:first-child {{
        background: linear-gradient(135deg, #FF4B4B 0%, #8B0000 100%);
        color: white;
        border-radius: 12px;
        border: none;
        height: 55px;
        font-size: 18px;
        font-weight: bold;
    }}

    /* Nút Zalo */
    .zalo-button {{
        display: inline-block;
        padding: 15px 25px;
        background-color: #0068ff;
        color: white !important;
        text-decoration: none;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        width: 100%;
        margin-top: 10px;
        transition: 0.3s;
    }}
    .zalo-button:hover {{
        background-color: #0056d6;
        box-shadow: 0 4px 15px rgba(0,104,255,0.4);
    }}

    .price-card {{
        background: #1e1e1e;
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        border-left: 10px solid #FF4B4B;
    }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. LOGIC XỬ LÝ (LOAD & TRAIN)
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
            s = str(val).replace('.', '').replace(',', '').strip()
            if len(s) > 8: return float(s) / 100
            return float(s)
        except: return np.nan

    df['Giá bán'] = df['Giá bán'].apply(clean_price)
    df['Tình trạng xe'] = df['Tình trạng xe'].astype(str).str.replace(',', '.').astype(float)
    df['Phụ tùng'] = df['đã phụ tùng chưa thay'].map({'đã thay': 1, 'chưa thay': 0}).fillna(0)
    df['Tuổi xe'] = current_year - df['Năm sản xuất']
    df = df.dropna(subset=['Giá bán', 'Hãng xe', 'Dòng xe'])
    return df

@st.cache_resource
def train_model(df):
    features = ["Hãng xe", "Dòng xe", "Tuổi xe", "Số km đã chạy", "Tình trạng xe", "Phụ tùng", "Khu vực bán"]
    X, y = df[features], df["Giá bán"]
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ["Tuổi xe", "Số km đã chạy", "Tình trạng xe", "Phụ tùng"]),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ["Hãng xe", "Dòng xe", "Khu vực bán"])
    ])
    pipeline = Pipeline([('pre', preprocessor), ('reg', RandomForestRegressor(n_estimators=300, random_state=42))])
    pipeline.fit(X, y)
    return pipeline, r2_score(y, pipeline.predict(X))

# --- CHẠY APP ---
df = load_and_preprocess()
current_year = datetime.now().year

if df is not None:
    model_pl, r2 = train_model(df)
    
    # SIDEBAR - THÔNG TIN LIÊN HỆ
    with st.sidebar:
        if os.path.exists("image_550a7b.png"):
            st.image("image_550a7b.png", use_container_width=True)
        
        st.markdown("<h2 style='text-align: center;'>Muốn Bán xe ư?</h2>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>HÃY TÌM E TUẤN</h1>", unsafe_allow_html=True)
        st.divider()
        
        st.markdown(f"📞 **Hotline:** {SĐT_ZALO}")
        st.markdown(f"💬 [Nhắn Zalo ngay]({LINK_ZALO})")
        st.divider()
        st.success(f"✅ AI đang hoạt động\nĐộ tin cậy: {r2*100:.1f}%")

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
                    time.sleep(0.5)
                    input_data = pd.DataFrame([{
                        "Hãng xe": brand, "Dòng xe": model_bike, "Tuổi xe": current_year - year,
                        "Số km đã chạy": km, "Tình trạng xe": cond, 
                        "Phụ tùng": 1 if part == "đã thay" else 0, "Khu vực bán": area
                    }])
                    res = model_pl.predict(input_data)[0]
                    
                    st.balloons()
                    st.markdown(f"""
                        <div class="price-card">
                            <h3 style="color: white; margin: 0;">GIÁ THỊ TRƯỜNG ĐỀ XUẤT</h3>
                            <h1 style="color: #FF4B4B; font-size: 50px; margin: 10px 0;">{res:,.0f} VNĐ</h1>
                            <p style="color: #bbb;">Được tính toán bởi e Tuấn AI</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # NÚT ZALO LIÊN HỆ
                    st.markdown(f"""
                        <a href="{LINK_ZALO}" target="_blank" class="zalo-button">
                            📱 NHẮN TUẤN CHỐT XE NGAY QUA ZALO
                        </a>
                    """, unsafe_allow_html=True)
                    
                    avg = df[df['Dòng xe'] == model_bike]['Giá bán'].mean()
                    st.metric("Chênh lệch so với giá trung bình", f"{res:,.0f}đ", f"{res-avg:,.0f}đ")
            else:
                st.info("Nhập thông tin bên trái rồi nhấn nút để e Tuấn tính giá cho nhé!")
                st.image("https://images.unsplash.com/photo-1558981403-c5f91dbbe480?auto=format&fit=crop&q=80&w=1000")

    with tab2:
        st.subheader("📈 Phân tích giá trị khấu hao")
        fig = px.scatter(df, x="Tuổi xe", y="Giá bán", color="Hãng xe", hover_data=["Dòng xe"], trendline="lowess")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Tuấn ơi, thiếu file 'motorbike_data.csv' rồi!")
