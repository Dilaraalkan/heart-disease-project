import joblib
import numpy as np

# Modeli ve Ölçekleyiciyi Yükle
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

def tahmin_yap():
    print("\n--- Yeni Hasta Bilgilerini Giriniz ---")
    
    # Kullanıcıdan tek tek veri alma
    try:
        age = float(input("Yaş: "))
        sex = float(input("Cinsiyet (Erkek: 1, Kadın: 0): "))
        cp = float(input("Göğüs Ağrısı Tipi (0-3): "))
        trestbps = float(input("Tansiyon (trestbps): "))
        chol = float(input("Kolesterol (chol): "))
        fbs = float(input("Kan Şekeri > 120 (1/0): "))
        restecg = float(input("EKG Sonucu (0-2): "))
        thalach = float(input("Max Nabız (thalach): "))
        exang = float(input("Egzersize Bağlı Ağrı (1/0): "))
        oldpeak = float(input("ST Depresyonu (oldpeak): "))
        slope = float(input("ST Eğimi (0-2): "))
        ca = float(input("Damar Sayısı (0-4): "))
        thal = float(input("Thalassemia (1-3): "))

        # Veriyi diziye dönüştür ve ölçeklendir
        veri = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        veri_scaled = scaler.transform(veri)

        # Tahmin ve Olasılık
        tahmin = model.predict(veri_scaled)
        olasilik = model.predict_proba(veri_scaled)

        print("\n" + "="*30)
        if tahmin[0] == 1:
            print(f"SONUÇ: Kalp Hastalığı Riski VAR (Olasılık: %{olasilik[0][1]*100:.2f})")
        else:
            print(f"SONUÇ: Kalp Hastalığı Riski YOK (Olasılık: %{olasilik[0][0]*100:.2f})")
        print("="*30)

    except Exception as e:
        print(f"Bir hata oluştu: {e}")

if __name__ == "__main__":
    tahmin_yap()