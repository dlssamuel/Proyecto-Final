import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class AlcoholLogisticModel:
    def __init__(self, random_state=42, **kwargs):
        """
        Inicializa el modelo de regresión logística.
        
        Parámetros:
        - random_state: semilla para reproducibilidad.
        - kwargs: otros hiperparámetros para LogisticRegression.
        """
        self.model = LogisticRegression(random_state=random_state, max_iter=1000, **kwargs)

    def fit(self, X_train, y_train):
        """Entrena el modelo con los datos de entrenamiento."""
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Genera predicciones (0 o 1) sobre el conjunto de test."""
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Evalúa el modelo en el conjunto de test y devuelve métricas.
        """
        y_pred = self.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }
        return metrics

    def save_model(self, filepath="alcohol_model.pkl"):
        """Guarda el modelo entrenado en un archivo .pkl."""
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, filepath="alcohol_model.pkl"):
        """Carga un modelo previamente guardado."""
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)