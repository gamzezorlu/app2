import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Doğalgaz Tüketim Anomali Tespit",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 Doğalgaz Tüketim Anomali Tespit Sistemi")
st.markdown("---")

# Yan panel - Dosya yükleme
st.sidebar.header("📁 Dosya Yükleme")
uploaded_file = st.sidebar.file_uploader(
    "CSV veya Excel dosyası seçin",
    type=['csv', 'xlsx', 'xls'],
    help="Tesisat numarası, bina numarası ve aylık tüketim verilerini içeren dosya"
)

# Parametreler
st.sidebar.header("⚙️ Analiz Parametreleri")
kis_tuketim_esigi = st.sidebar.slider(
    "Kış ayı düşük tüketim eşiği (m³/ay)",
    min_value=10, max_value=100, value=30,
    help="Kış aylarında bu değerin altındaki tüketim şüpheli kabul edilir"
)

bina_ort_dusuk_oran = st.sidebar.slider(
    "Bina ortalamasından düşük olma oranı (%)",
    min_value=30, max_value=90, value=60,
    help="Bina ortalamasından bu oranda düşük tüketim şüpheli kabul edilir"
)

ani_dusus_orani = st.sidebar.slider(
    "Ani düşüş oranı (%)",
    min_value=40, max_value=90, value=70,
    help="Önceki kış aylarına göre bu oranda düşüş şüpheli kabul edilir"
)

min_onceki_kis_tuketim = st.sidebar.slider(
    "Minimum önceki kış tüketimi (m³)",
    min_value=50, max_value=200, value=100,
    help="Ani düşüş tespiti için önceki kış aylarında minimum tüketim"
)

# Buradan sonrası fonksiyonlar ve analiz bloğu (önceki yanıtla birleştirilecek uzunlukta)

# Buraya kadar olan kod örnek ve başlatıcı ayarlamaları içeriyor.
# Geri kalan analiz ve görselleştirme fonksiyonları da aynı şekilde dahil edilecek.
