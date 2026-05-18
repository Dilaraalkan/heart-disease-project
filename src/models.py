# src/models.py
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def get_models():
    """
    Kullanılacak makine öğrenmesi modellerini ham halleriyle döndürür.
    Tüm modeller main.py içindeki RandomizedSearchCV ile adilce optimize edilecektir.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }
    return models