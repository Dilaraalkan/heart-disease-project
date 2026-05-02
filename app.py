import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np

# 1. Ayarlar ve Veri Yükleme
st.set_page_config(page_title="Kalp Sağlığı Analizörü", layout="wide")

# Hata yönetimi ile model yükleme
try:
    df = pd.read_csv('data/heart.csv')
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
except Exception as e:
    st.error("Model veya veri dosyaları bulunamadı. Lütfen 'models/' ve 'data/' klasörlerini kontrol edin.")

# Yan Menü Tasarımı
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/822/822118.png", width=100)
st.sidebar.title("Kontrol Paneli")
page = st.sidebar.selectbox("Sayfa Seçiniz:", ["Canlı Tahmin", "Model Analizi & Karar Yapısı"])

# --- SAYFA 1: CANLI TAHMİN ---
if page == "Canlı Tahmin":
    st.title("🩺 Akıllı Kalp Hastalığı Risk Tahmini")
    st.info("Lütfen hastanın klinik verilerini girerek 'Analiz Et' butonuna basınız.")

    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Yaş", 1, 100, 45)
            trestbps = st.number_input("Dinlenme Tansiyonu", 80, 200, 120)
            chol = st.number_input("Kolesterol", 100, 500, 200)
            fbs = st.selectbox("Açlık Kan Şekeri > 120 mg/dl?", [0, 1], format_func=lambda x: "Evet" if x==1 else "Hayır")
            
        with col2:
            sex = st.selectbox("Cinsiyet", [1, 0], format_func=lambda x: "Erkek" if x==1 else "Kadın")
            thalach = st.number_input("Maks. Kalp Hızı", 60, 220, 150)
            oldpeak = st.slider("ST Depresyonu (oldpeak)", 0.0, 6.0, 0.0)
            exang = st.selectbox("Egzersize Bağlı Anjin?", [0, 1], format_func=lambda x: "Evet" if x==1 else "Hayır")
            
        with col3:
            cp = st.selectbox("Göğüs Ağrısı Tipi (0-3)", [0, 1, 2, 3])
            ca = st.selectbox("Damar Sayısı (0-4)", [0, 1, 2, 3, 4])
            thal = st.selectbox("Thalassemia (1-3)", [1, 2, 3])
            slope = st.selectbox("ST Eğimi (0-2)", [0, 1, 2])
            restecg = st.selectbox("EKG Sonucu (0-2)", [0, 1, 2])

    if st.button("Hemen Analiz Et"):
        # 1. Ham veriyi DataFrame olarak oluştur (Sütun isimleri heart.csv ile aynı olmalı)
        input_df = pd.DataFrame([{
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }])
        
        # 2. ÖNEMLİ: Sadece ham veriyi ölçeklendiriyoruz
        # Streamlit arayüzünden gelen (Yaş: 45 vb.) veriler için bu adım şarttır.
        # Bu işlem veriyi '2026-05-01T10-38_export.csv' dosyasındaki formata getirir.
        input_scaled = scaler.transform(input_df)
        
        # 3. Tahmin yapıyoruz
        # prob[0] -> Sağlıklı olma ihtimali, prob[1] -> Hasta olma ihtimali
        prob = model.predict_proba(input_scaled)[0]
        
        # --- Hata Ayıklama Paneli (Sadece kontrol için) ---
        with st.expander("Teknik Detayları Gör"):
            st.write("Arayüzden Gelen Ham Veri:", input_df)
            st.write("Modele Giren Son Hali (Ölçeklenmiş):", input_scaled)
        
        # 4. Sonuçları ekrana basıyoruz
        st.divider()
        if prob[1] > 0.5:
            st.error(f"### ⚠️ Yüksek Risk: %{prob[1]*100:.1f}")
            st.progress(prob[1])
        else:
            st.success(f"### ✅ Düşük Risk: %{prob[0]*100:.1f}")
            st.progress(prob[1])
        
        # Uyarı mesajı
        st.warning(
            "**Yasal Uyarı:** Bu sonuç yalnızca bir makine öğrenmesi modeline dayalı tahmindir. "
            "Tıbbi bir teşhis değildir. Kesin değerlendirme için lütfen bir uzmana başvurunuz."
        )

# --- SAYFA 2: MODEL ANALİZİ ---
elif page == "Model Analizi & Karar Yapısı":
    st.title("Yapay Zeka Nasıl Karar Veriyor?")
    
    st.subheader("Özelliklerin Önem Sıralaması")
    st.write("Modelimiz tahmin yaparken hangi verilere daha çok önem veriyor?")

    # Özellik önemlerini hesapla
    importances = model.feature_importances_
    feature_names = df.drop('target', axis=1).columns
    feature_importance_df = pd.DataFrame({'Özellik': feature_names, 'Önem': importances}).sort_values(by='Önem', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Önem', y='Özellik', data=feature_importance_df, palette='magma', ax=ax)
    plt.title("Model Karar Faktörleri")
    st.pyplot(fig)

    st.write("> **Analiz:** Grafikte en üstte yer alan özellikler, kalp hastalığı tahmininde modelin en çok güvendiği parametrelerdir.")