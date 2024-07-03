# Используйте официальный образ Apache Airflow как базовый
FROM apache/airflow:latest

USER airflow

# Установка необходимых библиотек
RUN pip install --no-cache-dir \
    scikit-learn \
    joblib \
    pandas \
    sqlalchemy \
    pendulum \
    psycopg2-binary \
	kaggle
	

USER root
	
# Создание директории .kaggle и копирование файла kaggle.json
RUN mkdir -p /home/airflow/.kaggle && \
    chown -R airflow: /home/airflow/.kaggle

COPY .kaggle/kaggle.json /home/airflow/.kaggle/kaggle.json

# Изменение прав доступа и владельца файла kaggle.json
RUN chmod 600 /home/airflow/.kaggle/kaggle.json && \
    chown airflow: /home/airflow/.kaggle/kaggle.json
	
USER airflow

