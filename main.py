# main.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import warnings

# Terminaldeki FutureWarning ve UserWarning mesajlarını gizleyerek temiz bir çıktı almak için:
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.models import get_models

# 1. Veriyi Yükleme
data_path = 'data/heart.csv'
if not os.path.exists(data_path):
    print(f"Hata: '{data_path}' dosyası bulunamadı!")
else:
    df = pd.read_csv(data_path)

    # 2. ÖNLEM 1: Yinelenen (Duplicate) Satırları Temizleme
    print("--- Veri Seti Ön Kontrolleri ---")
    print(f"Temizlik Öncesi Toplam Satır: {len(df)}")
    df = df.drop_duplicates()
    print(f"Temizlik Sonrası Eşsiz Satır: {len(df)}")
    
    # Eksik Veri Kontrolü
    print(f"Eksik Veri Sayısı: {df.isnull().sum().sum()}")
    print("-" * 60)

    # 3. Özellik (X) ve Hedef (y) Seçimi
    X = df.drop('target', axis=1)
    y = df['target']

    # 4. ÖNLEM 2: Önce Veriyi Bölüyoruz (Data Leakage Önlemi)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 5. ÖNLEM 3: Doğru Veri Ölçeklendirme
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # 6. Tüm Modeller İçin Hiperparametre Havuzlarının Tanımlanması (ADİL YARIŞ)
    # Lojistik regresyon uyarısını engellemek için penalty kalıbı güncellendi.
    param_grids = {
        "Logistic Regression": {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0]
        },
        "KNN": {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        "SVM": {
            'C': [0.1, 1.0, 10.0, 100.0],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'kernel': ['linear', 'rbf']
        },
        "Random Forest": {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5, 10],
            'class_weight': ['balanced', None]
        }
    }

    base_models = get_models()
    optimized_models = {}
    results_f1 = {}

    print("\n[BİLGİ] TÜM MODELLER için hızlı parametre optimizasyonu başlatılıyor (Metrik: F1-Score)...")
    print("-" * 60)
    
    for name, model in base_models.items():
        print(f"-> {name} optimize ediliyor ve eğitiliyor...")
        
        # Logistic regression parametre sayısı 5 adet olduğu için n_iter uyarısı vermemesi adına dinamik n_iter ayarı:
        current_n_iter = min(15, np.prod([len(v) for v in param_grids[name].values()]))
        
        search = RandomizedSearchCV(
            estimator=model, 
            param_distributions=param_grids[name], 
            n_iter=int(current_n_iter), 
            cv=5, 
            scoring='f1', 
            n_jobs=-1, 
            random_state=42,
            verbose=0
        )
        
        search.fit(X_train, y_train)
        
        # DÜZELTME: Parantezler kaldırıldı ve sonuna alt çizgi (_) eklendi.
        best_model_instance = search.best_estimator_
        optimized_models[name] = best_model_instance
        
        # Performans Hesaplamaları
        y_train_pred = best_model_instance.predict(X_train)
        y_test_pred = best_model_instance.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        
        cv_scores = cross_val_score(best_model_instance, X_train, y_train, cv=5, scoring='f1')
        cv_mean = cv_scores.mean()
        
        results_f1[name] = test_f1
        
        print(f"   [OK] Seçilen En İyi Parametreler: {search.best_params_}")
        print(f"   -> Train Accuracy:  {train_acc:.2f}")
        print(f"   -> Test Accuracy:   {test_acc:.2f}")
        print(f"   -> Test Precision:  {test_precision:.2f}")
        print(f"   -> Test Recall (⚠️): {test_recall:.2f} (Hastaları Teşhis Etme Gücü)")
        print(f"   -> Test F1-Score:   {test_f1:.2f}")
        print(f"   -> CV F1-Score Mean:{cv_mean:.2f}")
        
        if train_acc - test_acc > 0.15:
            print("   ⚠️ UYARI: Overfitting (Aşırı Öğrenme) riski tespit edildi!")
        print("-" * 60)

    # 7. En Başarılı Modeli Seçme ve Kaydetme İşlemleri
    if not os.path.exists('models'):
        os.makedirs('models')

    best_model_name = max(results_f1, key=results_f1.get)
    best_model = optimized_models[best_model_name]

    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    print(f"\n[OK] Klinik Değerlendirme Sonucunda En Başarılı Seçilen Model: {best_model_name}")
    print("[OK] En iyi model ve scaler 'models/' klasörüne başarıyla kaydedildi.")

    # 8. Hata Matrisi (Confusion Matrix) Görselleştirme
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', 
                xticklabels=['Sağlıklı', 'Hasta'], 
                yticklabels=['Sağlıklı', 'Hasta'])
    plt.title(f"Confusion Matrix - {best_model_name} (F1 Optimized)")
    plt.xlabel("Tahmin Edilen Durum")
    plt.ylabel("Gerçek Klinik Durum")
    plt.show()