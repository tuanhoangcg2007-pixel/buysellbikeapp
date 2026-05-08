import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import time
from datetime import datetime
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


# ================================
# CONFIG UI
# ================================

st.set_page_config(page_title="Moto Price AI", page_icon="🏍️", layout="wide")


st.markdown("""
<style>

.main {
background-color:#f5f7fb;
}

div.stButton > button:first-child{
background-color:#FF4B4B;
color:white;
border-radius:10px;
height:50px;
font-size:18px;
}

</style>
""", unsafe_allow_html=True)



# ================================
# CLEAN PRICE
# ================================

def clean_price(val):

    try:

        s = str(val).strip()

        s = re.sub(r'[.,]00$', '', s)

        s = s.replace('.', '')
        s = s.replace(',', '')

        return float(s)

    except:

        return np.nan



# ================================
# LOAD DATA
# ================================

@st.cache_data
def load_data():

    file = "motorbike_data.csv"

    if not os.path.exists(file):
        return None

    df = pd.read_csv(file)

    df.columns = df.columns.str.strip()

    df["Giá bán"] = df["Giá bán"].apply(clean_price)

    df["Tình trạng xe"] = df["Tình trạng xe"].astype(str).str.replace(",", ".").astype(float)

    df["Phụ tùng"] = df["đã phụ tùng chưa thay"].map({
        "chưa thay":0,
        "đã thay":1
    }).fillna(0)

    current_year = datetime.now().year

    df["Tuổi xe"] = current_year - df["Năm sản xuất"]

    df = df.dropna(subset=["Giá bán","Hãng xe","Dòng xe"])


    # remove outliers
    Q1 = df["Giá bán"].quantile(0.25)
    Q3 = df["Giá bán"].quantile(0.75)

    IQR = Q3 - Q1

    df = df[(df["Giá bán"] >= Q1 - 1.5*IQR) & (df["Giá bán"] <= Q3 + 1.5*IQR)]

    return df



# ================================
# TRAIN MODEL
# ================================

@st.cache_resource
def train_model(df):

    features = [
        "Hãng xe",
        "Dòng xe",
        "Tuổi xe",
        "Số km đã chạy",
        "Tình trạng xe",
        "Phụ tùng",
        "Khu vực bán"
    ]

    X = df[features]
    y = df["Giá bán"]


    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )


    preprocessor = ColumnTransformer(

        [

            ("num",
             StandardScaler(),
             ["Tuổi xe","Số km đã chạy","Tình trạng xe","Phụ tùng"]
             ),

            ("cat",
             OneHotEncoder(handle_unknown="ignore"),
             ["Hãng xe","Dòng xe","Khu vực bán"]
             )

        ]

    )


    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=20,
        random_state=42
    )


    pipeline = Pipeline(

        [

            ("prep",preprocessor),
            ("model",model)

        ]

    )


    pipeline.fit(X_train,y_train)


    pred = pipeline.predict(X_test)


    r2 = r2_score(y_test,pred)

    mae = mean_absolute_error(y_test,pred)


    return pipeline,r2,mae



# ================================
# MAIN
# ================================

df = load_data()

current_year = datetime.now().year


if df is None:

    st.error("Không tìm thấy file motorbike_data.csv")

    st.stop()


model,r2,mae = train_model(df)



# ================================
# SIDEBAR
# ================================

with st.sidebar:

    st.title("🏍️ Moto AI")

    st.write("Dataset:",len(df))

    st.write("Độ chính xác:",f"{r2*100:.1f}%")

    st.write("Sai số trung bình:",f"{mae:,.0f} VNĐ")



# ================================
# TITLE
# ================================

st.title("🏍️ AI Motorbike Price Predictor")

st.write("Dự đoán giá bán xe máy dựa trên dữ liệu thị trường")



tab1,tab2,tab3 = st.tabs([
"🔮 Dự đoán giá",
"📊 Phân tích thị trường",
"📂 Dataset"
])



# ================================
# TAB 1
# ================================

with tab1:

    col1,col2 = st.columns(2)


    with col1:

        brands = sorted(df["Hãng xe"].unique())

        brand = st.selectbox("Hãng xe",brands)

        models = sorted(df[df["Hãng xe"]==brand]["Dòng xe"].unique())

        model_bike = st.selectbox("Dòng xe",models)

        year = st.number_input("Năm sản xuất",2000,current_year,current_year-3)

        km = st.number_input("Số km đã chạy",0,300000,10000)

        cond = st.slider("Tình trạng xe",1.0,10.0,8.0)

        part = st.radio("Đã thay phụ tùng",["chưa thay","đã thay"])

        area = st.selectbox("Khu vực bán",sorted(df["Khu vực bán"].unique()))

        btn = st.button("Dự đoán giá")


    with col2:

        if btn:

            with st.spinner("AI đang tính..."):

                time.sleep(0.5)

                input_data = pd.DataFrame([{

                    "Hãng xe":brand,
                    "Dòng xe":model_bike,
                    "Tuổi xe":current_year-year,
                    "Số km đã chạy":km,
                    "Tình trạng xe":cond,
                    "Phụ tùng":1 if part=="đã thay" else 0,
                    "Khu vực bán":area

                }])

                price = model.predict(input_data)[0]

                st.success(f"Giá dự đoán: {price:,.0f} VNĐ")



# ================================
# TAB 2
# ================================

with tab2:

    fig1 = px.box(
        df,
        x="Hãng xe",
        y="Giá bán",
        color="Hãng xe",
        title="Phân phối giá theo hãng"
    )

    st.plotly_chart(fig1,use_container_width=True)



    fig2 = px.scatter(
        df,
        x="Tuổi xe",
        y="Giá bán",
        color="Hãng xe",
        size="Số km đã chạy",
        hover_data=["Dòng xe"]
    )

    st.plotly_chart(fig2,use_container_width=True)



# ================================
# TAB 3
# ================================

with tab3:

    st.dataframe(df,use_container_width=True)
