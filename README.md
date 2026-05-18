# Kalp Hastalığı Risk Tahmin Projesi 

Bu proje; klinik parametreler üzerinden kalp hastalığı riskini yüksek kararlılıkla öngören, **veri sızıntısı (data leakage) içermeyen** uçtan uca entegre bir Klinik Karar Destek Sistemidir.

## Özellikler
* **Yenilik:** Ham veri setindeki mükerrer (duplicate) satırlar temizlenmiş ve hileli yüksek skorların (data leakage) önüne geçilmiştir.
* **Algoritmalar:** KNN, SVM, Random Forest, Logistic Regression.
* **Akıllı Seçim:** Tüm modeller **F1-Score** odaklı optimize edilir (`RandomizedSearchCV`) ve sistem en başarılı modeli otomatik seçer (`best_model.pkl`).
* **Açıklanabilir Yapay Zeka (XAI):** Seçilen modele göre öznitelik önem dereceleri (`feature_importances_` veya `coef_`) arayüzde grafiksel olarak açıklanır (Kara kutu engeli).
* **Arayüzler:** Hem Streamlit ile geliştirilmiş 2 sayfalık modern bir web arayüzü (`app.py`) hem de interaktif terminal tahmin modülü (`predict.py`) mevcuttur.

## Kurulum
1. `pip install -r requirements.txt`
2. `python3 main.py` *(Modelleri eğitir ve en iyisini kaydeder)*
3. `streamlit run app.py` *(Web arayüzünü başlatır)*