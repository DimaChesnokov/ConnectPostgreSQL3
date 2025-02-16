Описание
Приложение предназначено для анализа и классификации данных с использованием Python. В ходе работы реализованы три алгоритма:

K-Nearest Neighbors (KNN) – реализован вручную без использования sklearn.
Logistic Regression – линейный классификатор.
Support Vector Machine (SVM) – метод опорных векторов.

Программа выполняет следующие задачи:
Загрузка данных из базы PostgreSQL.
Проведение разведочного анализа данных (EDA).
Обработка пропущенных значений и кодирование категориальных признаков.
Масштабирование данных с использованием StandardScaler.
Обучение и тестирование классификационных моделей.
Оценка качества моделей по Accuracy, Precision, Recall, F1-score, Fused Metric.
Выбор наилучшей модели на основе метрик.

Инструкция по запуску:
1. Установка необходимых библиотек
   pip install pandas numpy seaborn matplotlib sqlalchemy scikit-learn psycopg2
2. Настройка подключения к базе данных
  -host = "povt-cluster.tstu.tver.ru"
  -port = 5432
  -user = "mpi"
  -password = "135a1"
  -atabase = "NameBD"
3.Запуск программы
python main.py

