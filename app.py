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
st.set_page_config(page_title="Moto Price AI", page_icon="🏍️", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    [data-testid="stMetricValue"] { font-size: 28px; color: #FF4B4B; font-weight: bold; }
    .st-emotion-cache-1r6slb0, .st-emotion-cache-12w0qpk {
        background-color: white; padding: 20px; border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 20px;
    }
    div.stButton > button:first-child {
        background-color: #FF4B4B; color: white; border-radius: 10px;
        border: none; height: 50px; font-size: 18px; font-weight: bold; transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #D32F2F; box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4); transform: scale(1.02);
    }
    h1 { color: #1E1E1E; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    h3 { color: #424242; }
    </style>
    """, unsafe_allow_html=True)
@st.cache_data
def load_and_preprocess():
    file_path = "motorbike_data.csv"
    if not os.path.exists(file_path): return None
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    current_year = datetime.now().year
    def clean_price(val):
        if pd.isna(val): return np.nan
        try:
            s = str(val).strip()
            # Cắt bỏ hai số 0 ở phần thập phân (vd: 19.800.000.00 hoặc 30.800.000,00)
            if s.endswith('.00') or s.endswith(',00'):
                s = s[:-3]
            # Loại bỏ hoàn toàn dấu chấm và phẩy còn sót lại
            s = s.replace('.', '').replace(',', '')
            return float(s)
        except: 
            return np.nan

    df['Giá bán'] = df['Giá bán'].apply(clean_price)

    for col in ['Hãng xe', 'Dòng xe', 'Khu vực bán']:
        df[col] = df[col].astype(str).str.strip()

    df['Tình trạng xe'] = df['Tình trạng xe'].astype(str).str.replace(',', '.').astype(float)
    
    df['đã phụ tùng chưa thay'] = df['đã phụ tùng chưa thay'].astype(str).str.strip()
    df['Phụ tùng'] = df['đã phụ tùng chưa thay'].apply(lambda x: 1 if 'đã thay' in x.lower() else 0)
    
    df['Tuổi xe'] = df['Năm sản xuất'].apply(lambda x: max(0, current_year - x))
    
    df = df.dropna(subset=['Giá bán', 'Hãng xe', 'Dòng xe'])

    Q1, Q3 = df['Giá bán'].quantile(0.25), df['Giá bán'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['Giá bán'] >= Q1 - 1.5 * IQR) & (df['Giá bán'] <= Q3 + 1.5 * IQR)]
    return df

@st.cache_resource
def train_model(df):
    features = ["Hãng xe", "Dòng xe", "Tuổi xe", "Số km đã chạy", "Tình trạng xe", "Phụ tùng", "Khu vực bán"]
    X, y = df[features], df["Giá bán"]
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ["Tuổi xe", "Số km đã chạy", "Tình trạng xe", "Phụ tùng"]),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ["Hãng xe", "Dòng xe", "Khu vực bán"])
    ])
    pipeline = Pipeline([('pre', preprocessor), ('reg', RandomForestRegressor(n_estimators=200, random_state=42))])
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    return pipeline, r2_score(y, y_pred), mean_absolute_error(y, y_pred)

df = load_and_preprocess()
current_year = datetime.now().year

if df is not None:
    model_pl, r2, mae = train_model(df)
    
    with st.sidebar:
        if os.path.exists("image_550a7b.png"):
            st.image("image_550a7b.png", use_container_width=True)
        else:
            st.title("🏍️ BUÔN BÁN XE MÁY")
        st.markdown("---")
        st.info("**Hướng dẫn:** Chọn thông số xe ở bảng bên phải để nhận định giá chính xác từ AI.")
        st.write(f"📊 **Dữ liệu:** 201 xe")
        st.write(f"🎯 **Độ chính xác:** {r2*100:.1f}%")

    st.title("🏍️ HKTshop-thu mua xe cũ giá cao")
    st.markdown("Hệ thống cửa hàng buôn bán xe cũ uy tín-chất lượng")

    tab1, tab2, tab3 = st.tabs(["🎯 Định giá xe ", "📊 Insight Thị trường", "📂 Dữ liệu gốc"])

    with tab1:
        left_col, right_col = st.columns([1, 1], gap="large")
        
        with left_col:
            st.subheader("📝 Thông tin chi tiết")
            with st.container():
                list_hang = sorted(df["Hãng xe"].unique())
                brand = st.selectbox("Hãng xe", list_hang)
                
                list_dong = sorted(df[df["Hãng xe"] == brand]["Dòng xe"].unique())
                model_bike = st.selectbox("Dòng xe", list_dong)
                
                c1, c2 = st.columns(2)
                year = c1.number_input("Năm sản xuất", 1990, 2030, current_year-2)
                km = c2.number_input("Số km đã đi", 0, 500000, 10000)
                
                cond = st.slider("Tình trạng ngoại hình (1-10)", 1.0, 10.0, 8.5)
                part = st.radio("Tình trạng máy móc", ["chưa thay", "đã thay"], horizontal=True)
                area = st.selectbox("Khu vực bán", sorted(df["Khu vực bán"].unique()))
                
                predict_btn = st.button("🚀 TÍNH GIÁ DỰ ĐOÁN")

        with right_col:
            st.subheader("💰 Kết quả phân tích")
            if predict_btn:
                with st.spinner("Đang tính toán giá trị tối ưu..."):
                    time.sleep(0.6)
                    # Gói dữ liệu để predict
                    input_data = pd.DataFrame([{
                        "Hãng xe": brand, "Dòng xe": model_bike, "Tuổi xe": max(0, current_year - year),
                        "Số km đã chạy": km, "Tình trạng xe": cond, 
                        "Phụ tùng": 1 if part == "đã thay" else 0, "Khu vực bán": area
                    }])
                    res = model_pl.predict(input_data)[0]
                    
                    st.balloons()
                    st.markdown(f"""
                        <div style="background-color: #1E1E1E; padding: 30px; border-radius: 15px; text-align: center;">
                            <h2 style="color: white; margin: 0;">GIÁ DỰ BÁO</h2>
                            <h1 style="color: #FF4B4B; font-size: 45px; margin: 10px 0;">{res:,.0f} VNĐ</h1>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    avg_price = df[df['Dòng xe'] == model_bike]['Giá bán'].mean()
                    st.write("")
                    st.metric(f"So với trung bình dòng {model_bike}", f"{res:,.0f}đ", f"{res-avg_price:,.0f}đ")
            else:
                st.write("Vui lòng điền thông tin và nhấn nút 'Tính giá' để xem kết quả.")
                st.image("https://images.unsplash.com/photo-1558981403-c5f91dbbe480?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", use_container_width=True)

    with tab2:
        st.subheader("📊 Phân tích xu hướng thị trường")
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.box(df, x="Hãng xe", y="Giá bán", color="Hãng xe", title="Phân phối giá theo hãng")
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.scatter(df, x="Tuổi xe", y="Giá bán", color="Hãng xe", size="Số km đã chạy", 
                             hover_data=['Dòng xe'], title="Độ mất giá theo tuổi xe")
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("📂 Bảng dữ liệu chi tiết")
        st.dataframe(df, use_container_width=True)

else:
    st.error("⚠️ Không tìm thấy file 'motorbike_data.csv'. Vui lòng kiểm tra lại!")
