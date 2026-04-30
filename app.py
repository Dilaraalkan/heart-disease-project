import streamlit as st
import joblib
import numpy as np

# 1. Kaydedilen Modelleri Yükle
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# 2. Sayfa Ayarları ve Başlık
st.set_page_config(page_title="Kalp Hastalığı Risk Analizi", layout="centered")
st.title("🩺 Kalp Hastalığı Risk Tahmin Sistemi")
st.write("Lütfen aşağıdaki sağlık verilerini girerek analiz butonuna basınız.")

# 3. Kullanıcı Giriş Alanları (Sidebar veya Ana Sayfa)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Yaş", min_value=1, max_value=120, value=40)
    sex = st.selectbox("Cinsiyet", options=[0, 1], format_func=lambda x: "Erkek" if x == 1 else "Kadın")
    cp = st.slider("Göğüs Ağrısı Tipi (0-3)", 0, 3, 1)
    trestbps = st.number_input("Dinlenme Tansiyonu", 80, 200, 120)
    chol = st.number_input("Kolesterol", 100, 600, 200)
    fbs = st.selectbox("Açlık Kan Şekeri > 120 mg/dl?", options=[0, 1], format_func=lambda x: "Evet" if x == 1 else "Hayır")

with col2:
    restecg = st.slider("EKG Sonucu (0-2)", 0, 2, 0)
    thalach = st.number_input("Maksimum Kalp Atış Hızı", 60, 220, 150)
    exang = st.selectbox("Egzersize Bağlı Göğüs Ağrısı?", options=[0, 1], format_func=lambda x: "Evet" if x == 1 else "Hayır")
    oldpeak = st.number_input("ST Depresyonu (oldpeak)", 0.0, 6.0, 0.0, step=0.1)
    slope = st.slider("ST Eğimi (0-2)", 0, 2, 1)
    ca = st.slider("Renkli Damar Sayısı (0-4)", 0, 4, 0)
    thal = st.slider("Thalassemia (1-3)", 1, 3, 2)

# 4. Tahmin Butonu ve Sonuç Ekranı
if st.button("ANALİZ ET"):
    # Veriyi hazırla
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    
    # Tahmin
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    
    st.divider()
    
    if prediction[0] == 1:
        st.error(f"### ⚠️ RİSK TESPİT EDİLDİ!")
        st.write(f"Modelin tahmin güveni: **%{probability[0][1]*100:.2f}**")
    else:
        st.success(f"###  DÜŞÜK RİSK")
        st.write(f"Modelin sağlık tahmini güveni: **%{probability[0][0]*100:.2f}**")
        
    st.info("Not: Bu sonuçlar sadece bir makine öğrenmesi modelinin tahmini olup tıbbi tavsiye niteliği taşımaz.")