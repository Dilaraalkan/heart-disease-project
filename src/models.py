from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def get_models():
    """
    Kullanılacak makine öğrenmesi modellerini sözlük yapısında döndürür.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, kernel='linear'),
        "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=100, min_samples_leaf=4)
    }
    return models