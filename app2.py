import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Doğalgaz Anomali Tespit Sistemi",
    page_icon="🔥",
    layout="wide"
)

st.header("📈 Genel İstatistikler")

# CSS ile görsel iyileştirmeler
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main > div {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #FF6B35, #F7931E);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #FFA726, #FF9800);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .warning-card:hover {
        transform: translateY(-5px);
    }
    
    .success-card {
        background: linear-gradient(135deg, #66BB6A, #4CAF50);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .success-card:hover {
        transform: translateY(-5px);
    }
    
    .info-card {
        background: linear-gradient(135deg, #42A5F5, #2196F3);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-5px);
    }
    
    .sidebar .stSelectbox label {
        color: #FF6B35;
        font-weight: bold;
    }
    
    .sidebar .stSlider label {
        color: #FF6B35;
        font-weight: bold;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #FF6B35, #F7931E);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .analysis-section {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #FF6B35;
    }
    
    .parameter-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class GasLeakDetector:
    def __init__(self):
        self.df = None
        self.suspicious_facilities = []
        
    def load_data(self, uploaded_file):
        """Excel dosyasını yükle ve temizle"""
        try:
            # Farklı dosya formatlarını destekle
            if uploaded_file.name.endswith('.xlsx'):
                self.df = pd.read_excel(uploaded_file, engine='openpyxl')
            elif uploaded_file.name.endswith('.xls'):
                self.df = pd.read_excel(uploaded_file, engine='xlrd')
            else:
                # CSV olarak da deneyelim
                self.df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Dosya yüklendi: {uploaded_file.name}")
            st.info(f"📊 Veri boyutu: {self.df.shape[0]} satır, {self.df.shape[1]} sütun")
            
            return True
        except Exception as e:
            st.error(f"❌ Dosya yüklenirken hata: {str(e)}")
            st.info("💡 Lütfen dosyanızın Excel (.xlsx, .xls) formatında olduğundan emin olun.")
            return False
    
    def preprocess_data(self):
        """Veriyi ön işleme"""
        if self.df is None:
            return False
            
        # Tarih sütunlarını tespit et (2016-2025 arası)
        date_columns = []
        for col in self.df.columns:
            if any(year in str(col) for year in range(2016, 2026)):
                date_columns.append(col)
        
        # Sadece sayısal verileri al
        numeric_columns = ['TN', 'BN'] + date_columns
        self.df = self.df[numeric_columns]
        
        # Eksik verileri 0 ile doldur
        self.df[date_columns] = self.df[date_columns].fillna(0)
        
        # Negatif değerleri 0 yap
        self.df[date_columns] = self.df[date_columns].clip(lower=0)
        
        return True
    
    def detect_anomalies(self, low_threshold=30, neighbor_threshold=60, drop_threshold=70):
        """Anomali tespiti algoritmaları"""
        if self.df is None:
            return []
        
        date_columns = [col for col in self.df.columns if any(year in str(col) for year in range(2016, 2026))]
        suspicious_list = []
        
        for idx, row in self.df.iterrows():
            tn = row['TN']
            bn = row['BN']
            consumption_data = row[date_columns].values
            
            # Anomali testleri
            anomalies = []
            risk_score = 0
            
            # 1. Ani düşüş tespiti
            sudden_drops = self._detect_sudden_drops(consumption_data, drop_threshold)
            if sudden_drops['count'] > 0:
                anomalies.append(f"Ani düşüş: {sudden_drops['count']} kez")
                risk_score += sudden_drops['count'] * 20
            
            # 2. Sıfır tüketim tespiti
            zero_consumption = self._detect_zero_consumption(consumption_data)
            if zero_consumption['count'] > 0:
                anomalies.append(f"Sıfır tüketim: {zero_consumption['count']} ay")
                risk_score += zero_consumption['count'] * 15
            
            # 3. Düşük tüketim tespiti
            low_consumption = self._detect_low_consumption(consumption_data, low_threshold)
            if low_consumption['suspicious']:
                anomalies.append(f"Düşük tüketim: Ortalama {low_consumption['avg_consumption']:.1f}")
                risk_score += 25
            
            # 4. Trend analizi
            trend_anomaly = self._detect_trend_anomaly(consumption_data)
            if trend_anomaly['suspicious']:
                anomalies.append(f"Trend anomalisi: {trend_anomaly['description']}")
                risk_score += 30
            
            # 5. Mevsimsel anomali
            seasonal_anomaly = self._detect_seasonal_anomaly(consumption_data)
            if seasonal_anomaly['suspicious']:
                anomalies.append(f"Mevsimsel anomali: {seasonal_anomaly['description']}")
                risk_score += 20
            
            # 6. Komşu tesisatlarla karşılaştırma
            neighbor_anomaly = self._compare_with_neighbors(bn, tn, consumption_data, neighbor_threshold)
            if neighbor_anomaly['suspicious']:
                anomalies.append(f"Komşu anomalisi: {neighbor_anomaly['description']}")
                risk_score += 35
            
            # Risk seviyesi belirleme
            if risk_score >= 70:
                risk_level = "Yüksek Risk"
            elif risk_score >= 40:
                risk_level = "Orta Risk"
            elif risk_score >= 20:
                risk_level = "Düşük Risk"
            else:
                risk_level = "Normal"
            
            if anomalies:
                suspicious_list.append({
                    'TN': tn,
                    'BN': bn,
                    'Risk_Skoru': risk_score,
                    'Risk_Seviyesi': risk_level,
                    'Anomaliler': '; '.join(anomalies),
                    'Ortalama_Tuketim': np.mean(consumption_data),
                    'Toplam_Tuketim': np.sum(consumption_data),
                    'Son_6_Ay_Ortalama': np.mean(consumption_data[-6:]),
                    'İlk_6_Ay_Ortalama': np.mean(consumption_data[:6])
                })
        
        return suspicious_list
    
    def _detect_sudden_drops(self, data, threshold=70):
        """Ani düşüş tespiti"""
        drops = 0
        for i in range(1, len(data)):
            if data[i-1] > 0 and data[i] < data[i-1] * ((100-threshold)/100):
                drops += 1
        return {'count': drops}
    
    def _detect_zero_consumption(self, data):
        """Sıfır tüketim tespiti"""
        zero_count = np.sum(data == 0)
        return {'count': zero_count}
    
    def _detect_low_consumption(self, data, threshold=30):
        """Düşük tüketim tespiti"""
        non_zero_data = data[data > 0]
        if len(non_zero_data) == 0:
            return {'suspicious': True, 'avg_consumption': 0}
        
        avg_consumption = np.mean(non_zero_data)
        
        # Parametre olarak gelen threshold'u kullan
        if avg_consumption < threshold:
            return {'suspicious': True, 'avg_consumption': avg_consumption}
        
        return {'suspicious': False, 'avg_consumption': avg_consumption}
    
    def _detect_trend_anomaly(self, data):
        """Trend anomalisi tespiti"""
        # Son 24 ayın ortalamasını al
        recent_data = data[-24:]
        if len(recent_data) < 12:
            return {'suspicious': False, 'description': ''}
        
        # Lineer trend hesapla
        x = np.arange(len(recent_data))
        z = np.polyfit(x, recent_data, 1)
        trend_slope = z[0]
        
        # Eğer trend çok negatifse (sürekli azalma) şüpheli
        if trend_slope < -5:
            return {'suspicious': True, 'description': 'Sürekli azalan tüketim trendi'}
        
        return {'suspicious': False, 'description': ''}
    
    def _detect_seasonal_anomaly(self, data):
        """Mevsimsel anomali tespiti"""
        if len(data) < 24:
            return {'suspicious': False, 'description': ''}
        
        # Kış ayları (Aralık, Ocak, Şubat) ve yaz ayları (Haziran, Temmuz, Ağustos)
        # Basit mevsimsel kontrol
        winter_months = []
        summer_months = []
        
        for i in range(len(data)):
            month = (i % 12) + 1
            if month in [12, 1, 2]:  # Kış ayları
                winter_months.append(data[i])
            elif month in [6, 7, 8]:  # Yaz ayları
                summer_months.append(data[i])
        
        if len(winter_months) > 0 and len(summer_months) > 0:
            winter_avg = np.mean(winter_months)
            summer_avg = np.mean(summer_months)
            
            # Kış aylarında tüketim yaz aylarından az ise şüpheli
            if winter_avg < summer_avg * 0.8:
                return {'suspicious': True, 'description': 'Kış aylarında beklenenden düşük tüketim'}
        
        return {'suspicious': False, 'description': ''}
    
    def _compare_with_neighbors(self, bn, tn, data, threshold=60):
        """Komşu tesisatlarla karşılaştırma"""
        # Aynı binadaki diğer tesisatları bul
        neighbors = self.df[self.df['BN'] == bn]
        
        if len(neighbors) <= 1:
            return {'suspicious': False, 'description': ''}
        
        # Kendi verisini çıkar
        neighbors = neighbors[neighbors['TN'] != tn]
        
        if len(neighbors) == 0:
            return {'suspicious': False, 'description': ''}
        
        # Komşuların ortalama tüketimini hesapla
        date_columns = [col for col in self.df.columns if any(year in str(col) for year in range(2016, 2026))]
        neighbor_consumptions = []
        
        for _, neighbor in neighbors.iterrows():
            neighbor_data = neighbor[date_columns].values
            neighbor_avg = np.mean(neighbor_data[neighbor_data > 0])
            if not np.isnan(neighbor_avg):
                neighbor_consumptions.append(neighbor_avg)
        
        if len(neighbor_consumptions) == 0:
            return {'suspicious': False, 'description': ''}
        
        current_avg = np.mean(data[data > 0])
        neighbor_avg = np.mean(neighbor_consumptions)
        
        # Parametre olarak gelen threshold'u kullan
        if current_avg < neighbor_avg * (threshold/100):
            return {'suspicious': True, 'description': f'Komşulardan {((neighbor_avg - current_avg) / neighbor_avg * 100):.0f}% daha az tüketim'}
        
        return {'suspicious': False, 'description': ''}

def main():
    st.markdown('<h1 class="main-header">🔥 Doğalgaz Tüketim Anomali Tespit Sistemi</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #FF6B35, #F7931E); border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">🔍 Kontrol Paneli</h2>
    </div>
    """, unsafe_allow_html=True)
    
    detector = GasLeakDetector()
    
    # Sidebar parametreleri
    st.sidebar.markdown("### 📊 Analiz Parametreleri")
    
    # Parametre kartları
    st.sidebar.markdown("""
    <div class="parameter-card">
        <h4>⚙️ Tespit Eşikleri</h4>
    </div>
    """, unsafe_allow_html=True)
    
    low_consumption_threshold = st.sidebar.slider(
        "Kış ayı düşük tüketim eşiği (m³/ay)", 
        min_value=10, 
        max_value=200, 
        value=30,
        help="Bu değerin altındaki tüketimler şüpheli kabul edilir"
    )
    
    neighbor_ratio_threshold = st.sidebar.slider(
        "Bina ortalamasından düşük olma oranı (%)", 
        min_value=30, 
        max_value=90, 
        value=60,
        help="Komşulardan bu kadar az tüketim şüpheli kabul edilir"
    )
    
    sudden_drop_threshold = st.sidebar.slider(
        "Ani düşüş oranı (%)", 
        min_value=40, 
        max_value=90, 
        value=70,
        help="Bir aydan diğerine bu kadar düşüş şüpheli kabul edilir"
    )
    
    st.sidebar.markdown("### 📋 Tespit Edilen Tarih Sütunları")
    
    # Dosya yükleme
    st.sidebar.markdown("""
    <div class="parameter-card">
        <h4>📂 Dosya Yükleme</h4>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.sidebar.file_uploader(
        "Excel veya CSV dosyasını yükleyin",
        type=['xlsx', 'xls', 'csv'],
        help="Doğalgaz tüketim verilerini içeren dosyayı seçin"
    )
    
    if uploaded_file is not None:
        if detector.load_data(uploaded_file):
            st.sidebar.success("✅ Dosya başarıyla yüklendi!")
            
            # Veri ön işleme
            if detector.preprocess_data():
                st.sidebar.success("✅ Veri işlendi!")
                
                # Veri önizleme
                st.header("👀 Veri Önizleme")
                
                if st.checkbox("Veri önizlemesini göster"):
                    st.write("**İlk 5 satır:**")
                    st.dataframe(detector.df.head())
                    
                    st.write("**Sütun bilgileri:**")
                    col_info = pd.DataFrame({
                        'Sütun': detector.df.columns,
                        'Veri Tipi': detector.df.dtypes,
                        'Null Değer': detector.df.isnull().sum(),
                        'Örnek Değer': [str(detector.df[col].iloc[0]) if len(detector.df) > 0 else 'N/A' for col in detector.df.columns]
                    })
                    st.dataframe(col_info)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="success-card">
                        <h3>{len(detector.df)}</h3>
                        <p>Toplam Tesisat</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    unique_buildings = detector.df['BN'].nunique()
                    st.markdown(f"""
                    <div class="success-card">
                        <h3>{unique_buildings}</h3>
                        <p>Toplam Bina</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Analiz butonu
                if st.sidebar.button("🔍 Anomali Analizi Başlat", type="primary"):
                    with st.spinner("Anomali tespiti yapılıyor..."):
                        suspicious_facilities = detector.detect_anomalies()
                    
                    if suspicious_facilities:
                        suspicious_df = pd.DataFrame(suspicious_facilities)
                        
                        # Sonuç istatistikleri
                        with col3:
                            high_risk_count = len(suspicious_df[suspicious_df['Risk_Seviyesi'] == 'Yüksek Risk'])
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{high_risk_count}</h3>
                                <p>Yüksek Risk</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            medium_risk_count = len(suspicious_df[suspicious_df['Risk_Seviyesi'] == 'Orta Risk'])
                            st.markdown(f"""
                            <div class="warning-card">
                                <h3>{medium_risk_count}</h3>
                                <p>Orta Risk</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Şüpheli tesisatlar tablosu
                        st.header("🚨 Şüpheli Tesisatlar")
                        
                        # Risk seviyesine göre filtreleme
                        risk_filter = st.selectbox(
                            "Risk Seviyesi Filtresi",
                            ["Tümü", "Yüksek Risk", "Orta Risk", "Düşük Risk"]
                        )
                        
                        if risk_filter != "Tümü":
                            filtered_df = suspicious_df[suspicious_df['Risk_Seviyesi'] == risk_filter]
                        else:
                            filtered_df = suspicious_df
                        
                        # Tabloyu göster
                        st.dataframe(
                            filtered_df,
                            use_container_width=True,
                            height=400
                        )
                        
                        # Görselleştirmeler
                        st.header("📊 Görselleştirmeler")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Risk seviyesi dağılımı
                            risk_counts = suspicious_df['Risk_Seviyesi'].value_counts()
                            fig_pie = px.pie(
                                values=risk_counts.values,
                                names=risk_counts.index,
                                title="Risk Seviyesi Dağılımı",
                                color_discrete_map={
                                    'Yüksek Risk': '#ff6b6b',
                                    'Orta Risk': '#ffa726',
                                    'Düşük Risk': '#66bb6a'
                                }
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col2:
                            # Risk skoru dağılımı
                            fig_hist = px.histogram(
                                suspicious_df,
                                x='Risk_Skoru',
                                title="Risk Skoru Dağılımı",
                                nbins=20,
                                color_discrete_sequence=['#1f77b4']
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Bina bazlı analiz
                        st.header("🏢 Bina Bazlı Analiz")
                        
                        building_analysis = suspicious_df.groupby('BN').agg({
                            'TN': 'count',
                            'Risk_Skoru': 'mean'
                        }).rename(columns={'TN': 'Şüpheli_Tesisat_Sayısı', 'Risk_Skoru': 'Ortalama_Risk_Skoru'})
                        
                        building_analysis = building_analysis.sort_values('Ortalama_Risk_Skoru', ascending=False)
                        
                        fig_bar = px.bar(
                            building_analysis.reset_index(),
                            x='BN',
                            y='Ortalama_Risk_Skoru',
                            title="Bina Bazlı Ortalama Risk Skoru",
                            color='Ortalama_Risk_Skoru',
                            color_continuous_scale='Reds'
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                        
                        # Excel raporu indirme
                        st.header("📋 Rapor İndirme")
                        
                        # Rapor oluşturma
                        with st.spinner("Rapor hazırlanıyor..."):
                            # Ana rapor
                            report_data = suspicious_df.copy()
                            
                            # Özet sayfa
                            summary_data = {
                                'Toplam_Tesisat': [len(detector.df)],
                                'Şüpheli_Tesisat': [len(suspicious_df)],
                                'Yüksek_Risk': [high_risk_count],
                                'Orta_Risk': [medium_risk_count],
                                'Düşük_Risk': [len(suspicious_df[suspicious_df['Risk_Seviyesi'] == 'Düşük Risk'])],
                                'Analiz_Tarihi': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                            }
                            summary_df = pd.DataFrame(summary_data)
                            
                            # Excel dosyası oluştur
                            output = io.BytesIO()
                            
                            try:
                                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                    summary_df.to_excel(writer, sheet_name='Özet', index=False)
                                    report_data.to_excel(writer, sheet_name='Şüpheli_Tesisatlar', index=False)
                                    building_analysis.to_excel(writer, sheet_name='Bina_Analizi', index=True)
                                output.seek(0)
                            except ImportError:
                                st.warning("⚠️ Excel çıktısı için openpyxl kütüphanesi gerekli. CSV olarak indiriliyor.")
                                # CSV alternatifi
                                csv_output = report_data.to_csv(index=False)
                                st.download_button(
                                    label="📥 CSV Raporu İndir",
                                    data=csv_output,
                                    file_name=f"dogalgaz_anomali_raporu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                                output = None
                        
                        if output:
                            st.download_button(
                                label="📥 Excel Raporu İndir",
                                data=output.getvalue(),
                                file_name=f"dogalgaz_anomali_raporu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        
                        # Detaylı analiz
                        st.header("🔍 Detaylı Analiz")
                        
                        selected_facility = st.selectbox(
                            "Detayını görmek istediğiniz tesisatı seçin:",
                            options=suspicious_df['TN'].tolist(),
                            format_func=lambda x: f"Tesisat {x}"
                        )
                        
                        if selected_facility:
                            facility_data = suspicious_df[suspicious_df['TN'] == selected_facility].iloc[0]
                            
                            st.subheader(f"Tesisat {selected_facility} - Detaylı Analiz")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Risk Skoru", f"{facility_data['Risk_Skoru']:.0f}")
                            
                            with col2:
                                st.metric("Risk Seviyesi", facility_data['Risk_Seviyesi'])
                            
                            with col3:
                                st.metric("Ortalama Tüketim", f"{facility_data['Ortalama_Tuketim']:.1f} m³")
                            
                            st.write("**Tespit Edilen Anomaliler:**")
                            st.write(facility_data['Anomaliler'])
                            
                            # Tüketim grafiği
                            facility_consumption = detector.df[detector.df['TN'] == selected_facility]
                            if not facility_consumption.empty:
                                date_columns = [col for col in detector.df.columns if any(year in str(col) for year in range(2016, 2026))]
                                consumption_values = facility_consumption[date_columns].values[0]
                                
                                fig_line = px.line(
                                    x=date_columns,
                                    y=consumption_values,
                                    title=f"Tesisat {selected_facility} - Tüketim Grafiği",
                                    labels={'x': 'Tarih', 'y': 'Tüketim (m³)'}
                                )
                                fig_line.update_layout(xaxis_tickangle=-45)
                                st.plotly_chart(fig_line, use_container_width=True)
                    
                    else:
                        st.success("🎉 Herhangi bir şüpheli tesisat tespit edilmedi!")
    
    else:
        st.info("👆 Lütfen sol panelden Excel dosyanızı yükleyin.")
        
        # Örnek veri formatı açıklaması
        st.header("📋 Veri Formatı")
        st.write("""
        **Beklenen Excel dosyası formatı:**
        - **TN**: Tesisat numarası
        - **BN**: Bina numarası  
        - **Tarih Sütunları**: 2016-2025 yılları arasındaki ay/yıl bilgisi içeren sütunlar
        
        **Örnek sütun isimleri:**
        - 2016/1, 2016/2, ..., 2025/6 formatında tarih sütunları
        - Her hücre o ay için doğalgaz tüketim miktarını (m³) içermelidir
        """)

if __name__ == "__main__":
    main()
