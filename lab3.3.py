import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# Данные для подключения
host = "povt-cluster.tstu.tver.ru"
port = 5432
user = "mpi"
password = "135a1"
database = "db_housing"

def connect_to_db(host, port, user, password, database):
    """Подключение к базе данных PostgreSQL."""
    engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{database}")
    print("Подключение успешно")
    return engine

def load_data(engine, table_name):
    """Загрузка данных из таблицы PostgreSQL."""
    return pd.read_sql(f"SELECT * FROM {table_name} LIMIT 5000", engine)

def exploratory_data_analysis(df):
    """Разведочный анализ данных."""
    print(f"Размер датафрейма: {df.shape[0]} строк, {df.shape[1]} столбцов")
    print(f"Размер датафрейма в оперативной памяти: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    print("C) Статистики для числовых переменных:")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_stats = df[numeric_cols].describe(percentiles=[0.25, 0.75]).T[['min', '50%', 'mean', 'max', '25%', '75%']]
    numeric_stats.rename(columns={'50%': 'median'}, inplace=True)
    print(numeric_stats)

    print("D) Мода для категориальных переменных:")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        mode_value = df[col].mode().dropna().values[0]
        mode_count = df[col].value_counts().max()
        print(f"   {col}: мода = '{mode_value}', встречается {mode_count} раз")

def preprocess_data(df, target_column):
    """Предобработка данных."""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Заполнение пропусков медианными значениями
    imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # Заполнение пропущенных категориальных данных (модой)
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    # Кодирование категориальных переменных (Frequency Encoding)
    for col in categorical_cols:
        df[col] = df[col].map(df[col].value_counts(normalize=True))

    # Бинаризация целевой переменной
    if df[target_column].nunique() > 2:
        threshold = df[target_column].median()
        df[target_column] = (df[target_column] > threshold).astype(int)

    # Разделение данных
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Масштабирование данных (StandardScaler)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

class CustomKNN:
    """Реализация K-Nearest Neighbors (KNN) с нуля."""
    def __init__(self, k=5):
        self.k = k

    def fit(self, X_train, y_train):
        """Запоминаем обучающие данные."""
        self.X_train = X_train
        self.y_train = y_train.to_numpy()  # Приводим к numpy для корректной индексации

    def predict(self, X_test):
        """Предсказываем класс для каждого объекта в X_test."""
        predictions = [self._predict_one(x) for x in X_test]
        return np.array(predictions)

    def _predict_one(self, x):
        """Предсказание класса для одного объекта x."""
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))  # Евклидово расстояние
        k_indices = np.argsort(distances)[:self.k]  # Индексы k ближайших соседей
        k_nearest_labels = self.y_train[k_indices]  # Их классы
        most_common = np.bincount(k_nearest_labels).argmax()  # Выбор наиболее частого класса
        return most_common


def fused_metric(accuracy, precision, recall, f1):
    """Объединённая метрика для оценки моделей."""
    return (accuracy + precision + recall + f1) / 4

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Обучает и оценивает модели KNN (самописный), Logistic Regression и SVM."""
    models = {
        "KNN (Custom)": CustomKNN(k=5),  # Заменяем библиотечный KNN на свой
        "Logistic Regression": LogisticRegression(max_iter=500),
        "SVM": SVC()
    }
    
    results = []
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=1)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        fused = fused_metric(accuracy, precision, recall, f1)
        execution_time = time.time() - start_time

        results.append((name, accuracy, precision, recall, f1, fused, execution_time))

    # Вывод результатов
    for model, acc, prec, rec, f1, fused, exec_time in results:
        print(f"{model} Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, Fused: {fused:.4f}, Время выполнения: {exec_time:.2f} сек")

    # Выбор лучшей модели по fused метрике
    best_model = max(results, key=lambda x: x[5])
    print(f"\nЛучшая модель: {best_model[0]}, fused метрика: {best_model[5]:.4f}")

def main():
    engine = connect_to_db(host, port, user, password, database)
    table_name = "public.\"Nashville_Housing\""
    df = load_data(engine, table_name)
    exploratory_data_analysis(df)
    target_column = "saleprice"
    X_train, X_test, y_train, y_test = preprocess_data(df, target_column)
    train_and_evaluate_models(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
