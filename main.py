import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from src.models import get_models

# 1. Veriyi Yükleme
data_path = 'data/heart.csv'
if not os.path.exists(data_path):
    print(f"Hata: '{data_path}' dosyası bulunamadı!")
else:
    df = pd.read_csv(data_path)

    # 2. Eksik Veri Kontrolü
    print("--- Eksik Veri Durumu ---")
    print(df.isnull().sum())
    print("-" * 30)

    # 3. Özellik (X) ve Hedef (y) Seçimi
    X = df.drop('target', axis=1)
    y = df['target']

    # 4. Veri Ölçeklendirme (Scaling)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Eğitim ve Test Setlerine Ayırma
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 6. Modelleri Eğitme ve Değerlendirme
    models = get_models()
    results = {}

    print("\n--- Model Performansları ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Çapraz doğrulama
        cv_score = cross_val_score(model, X_scaled, y, cv=5).mean()
        
        results[name] = acc
        print(f"{name:20} -> Test Accuracy: {acc:.2f} | CV Score: {cv_score:.2f}")

    # 7. Modeller Klasörünü Hazırlama
    if not os.path.exists('models'):
        os.makedirs('models')

    # 8. En İyi Modeli ve Ölçekleyiciyi Kaydetme (GÜVENLİ YÖNTEM)
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]

    # Modeli kaydet
    joblib.dump(best_model, 'models/best_model.pkl')
    # Ölçekleyiciyi kaydet (Tahmin için bu şart!)
    joblib.dump(scaler, 'models/scaler.pkl')

    print(f"\n[OK] En başarılı model: {best_model_name}")
    print("[OK] 'models/best_model.pkl' ve 'models/scaler.pkl' başarıyla kaydedildi.")

    # 9. Görselleştirme: Confusion Matrix
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', 
                xticklabels=['Sağlıklı', 'Hasta'], 
                yticklabels=['Sağlıklı', 'Hasta'])
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek Değer")
    
    print("\nGrafik penceresini kapattığınızda işlem tamamlanacaktır...")
    plt.show()