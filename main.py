import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
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

    # 4. Veri Ölçeklendirme
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 5. Eğitim ve Test Setlerine Ayırma
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 6. Modelleri Al ve Parametre Optimizasyonu Yap
    models = get_models()
    
    # --- HIZLANDIRILMIŞ OPTİMİZASYON ---
    print("\n[BİLGİ] Random Forest için hızlı parametre taraması başlatılıyor...")
    
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 4],
        'class_weight': ['balanced']
    }
    
    # GridSearchCV yerine RandomizedSearchCV kullanarak en iyi 20 kombinasyonu rastgele seçiyoruz (Çok daha hızlıdır)
    # verbose=1 ekledik, böylece terminalde ilerlemeyi göreceksin.
    random_rf = RandomizedSearchCV(
        RandomForestClassifier(random_state=42), 
        param_distributions=rf_params, 
        n_iter=20, 
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1, 
        verbose=1,
        random_state=42
    )
    
    random_rf.fit(X_train, y_train)
    
    models['Random Forest'] = random_rf.best_estimator_
    print(f"[OK] RF En İyi Parametreler: {random_rf.best_params_}")

    results = {}

    print("\n--- Model Performansları ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        
        # CV Score'u tüm veri setinde değil sadece eğitim setinde hesaplamak daha hızlı ve doğrudur
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        cv_mean = cv_scores.mean()
        
        results[name] = test_acc
        
        print(f"{name:20}")
        print(f"   -> Train Acc: {train_acc:.2f}")
        print(f"   -> Test Acc:  {test_acc:.2f}")
        print(f"   -> CV Score:  {cv_mean:.2f}")
        
        if train_acc - test_acc > 0.15:
            print("   ⚠️ UYARI: Overfitting (Aşırı Öğrenme) tespit edildi!")
        print("-" * 30)

    # 7. Kaydetme İşlemleri
    if not os.path.exists('models'):
        os.makedirs('models')

    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]

    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    print(f"\n[OK] En başarılı model: {best_model_name}")
    print("[OK] Modeller kaydedildi.")

    # 8. Görselleştirme
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', 
                xticklabels=['Sağlıklı', 'Hasta'], 
                yticklabels=['Sağlıklı', 'Hasta'])
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek Değer")
    plt.show()