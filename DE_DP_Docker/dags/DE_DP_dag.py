# Импорт библиотек и модулей, необходимых для выполнения DAG
from datetime import timedelta
from airflow.decorators import dag, task
import pendulum
from text_classification_module import (
    getting_dataset_by_api,
    unzip_and_replace_datasets,
    transforming_datasets,
    prepare_dfs_to_labeling,
    rule_based_labeling,
    merging_labeled_dfs,
    teaching_and_saving_model,
    testing_model,
    accuracy_scoring,
    create_postgres_database,
    write_dataframe_to_postgres
)

# Kaggle API для доступа к датасетам на Kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

# urllib для работы с URL
from urllib.parse import urljoin

# sklearn для машинного обучения:
from sklearn.model_selection import train_test_split  # Разделение данных на обучающую и тестовую выборки
from sklearn.feature_extraction.text import TfidfVectorizer  # Векторизация текста с использованием TF-IDF
from sklearn.linear_model import LogisticRegression  # Логистическая регрессия для классификации
from sklearn.utils import shuffle  # Перемешивание данных
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Метрики для оценки модели

# joblib для сохранения и загрузки модели
from joblib import dump, load

# datetime для работы с датами и временем
from datetime import datetime, timedelta

# Работа с файлами и директориями
import zipfile
import os

# pandas для работы с данными в табличном виде
import pandas as pd

# sqlalchemy для работы с базами данных через SQL выражения
from sqlalchemy import create_engine

# Airflow для организации и управления рабочими процессами:
from airflow.models import Variable  # Работа с переменными Airflow
from airflow.providers.postgres.hooks.postgres import PostgresHook # Хук Airflow для работы с PostgreSQL

# psycopg2 и sqlalchemy для работы с PostgreSQL:
import psycopg2  # Библиотека для работы с PostgreSQL
from psycopg2 import sql  # Модуль для безопасного формирования SQL запросов

# Определяем директорию для сохранения данных
DATA_DIR = "/opt/airflow/data"

# Определение DAG файла с описанием всех задач и их связей:
@dag(
    'Medical_text_classification',
    default_args={
        'owner': 'AllenovNS',       # Владелец DAG
        'depends_on_past': False,   # Не зависит от предыдущих запусков
        'start_date': pendulum.datetime(2024, 4, 25, tz='UTC'),     # Дата начала первого запуска
        'retries': 1,               # Количество попыток повторения при неудаче
        'retry_delay': timedelta(minutes=4),    # Задержка между попытками
    },
    description='A DAG to process and classification datasets contain medical abstracts and store data in PostgreSQL',
    schedule_interval=None,         # Интервал запуска
    catchup=False,                  # Не запускать пропущенные запуски
    tags=['DE_Diploma_project'],    # Теги для удобства поиска и группировки DAG
)
def my_text_classification_dag():

    # Загружает данные по API и сохраняет их в указанную директорию (DATA_DIR).
    @task
    def getting_dataset_by_api_task():
        getting_dataset_by_api(path=DATA_DIR)
    
    # Распаковывает архив с данными и заменяет имеющиеся датасеты в директории.
    @task
    def unzip_and_replace_datasets_task():
        unzip_and_replace_datasets(zip_path=f"{DATA_DIR}/archive.zip", extract_to=DATA_DIR)

    # Преобразует данные из бинарных файлов в CSV формат для дальнейшей обработки.
    @task
    def transforming_datasets_task():
        return transforming_datasets(
            train_path=f"{DATA_DIR}/train.dat",
            test_path=f"{DATA_DIR}/test.dat",
            test_csv_path=f"{DATA_DIR}/ma_test.csv",
            train_csv_path=f"{DATA_DIR}/ma_train.csv"
        )

    # Разделяет преобразованные данные на тренировочный и тестовый наборы.
    @task(multiple_outputs=True)
    def split_dataframes(df1_df2):
        df_train, df_test = df1_df2
        return {'dftn': df_train, 'dfts': df_test}

    # Разделяет тренировочный набор данных для ручной и автоматической разметки.
    @task
    def prepare_dfs_to_labeling_task(df_train: str):
        return prepare_dfs_to_labeling(df_train)

    # Выполняет автоматическую разметку данных на основе заданных правил.
    @task
    def rule_based_labeling_task(df_prep: str):
        return rule_based_labeling(df_prep)

    # Объединяет данные, размеченные вручную и автоматически.
    @task
    def merging_labeled_dfs_task(df_rbl: str):
        return merging_labeled_dfs(df_rbl)

    # Обучает модель на размеченных данных и сохраняет её вместе с векторизатором.
    @task
    def teaching_and_saving_model_task(df_merged: str):
        return teaching_and_saving_model(df_merged)

    # Применяет обученную модель к тестовому набору данных.
    @task
    def testing_model_task(df_test: str):
        return testing_model(df_test)

    # Оценивает точность модели на тренировочных данных.
    @task
    def train_accuracy_scoring_task(df_trained: str):
        accuracy_scoring(df_trained)

    # Оценивает точность модели на тестовых данных.
    @task
    def test_accuracy_scoring_task(df_tested: str):
        accuracy_scoring(df_tested)

    # Создаёт базу данных PostgreSQL для хранения результатов.
    @task
    def create_postgresdb_task():
        create_postgres_database(postgres_conn_id='diploma_project_conn', database_name='de_diploma_project')

    # Записывает тренировочные данные с предсказаниями в базу данных PostgreSQL.
    @task
    def write_train_task(df_trained: str):
        write_dataframe_to_postgres('train_df_with_predictions', df_trained, 'diploma_project_conn', 'de_diploma_project')

    # Записывает тестовые данные с предсказаниями в базу данных PostgreSQL.
    @task
    def write_test_task(df_tested: str):
        write_dataframe_to_postgres('test_df_with_predictions', df_tested, 'diploma_project_conn', 'de_diploma_project')

    # Определение зависимостей и последовательности выполнения задач для DAG:
    create_db = create_postgresdb_task()
    get_ds = getting_dataset_by_api_task()
    unzip = unzip_and_replace_datasets_task()
    df1_df2 = transforming_datasets_task()
    split_dfs = split_dataframes(df1_df2=df1_df2)
    df_prep = prepare_dfs_to_labeling_task(df_train=split_dfs['dftn'])
    df_rbl = rule_based_labeling_task(df_prep=df_prep)
    df_merged = merging_labeled_dfs_task(df_rbl=df_rbl)
    ds_trained = teaching_and_saving_model_task(df_merged=df_merged)
    ds_tested = testing_model_task(df_test=split_dfs['dfts'])
    train_scoring = train_accuracy_scoring_task(df_trained=ds_trained)
    test_scoring = test_accuracy_scoring_task(df_tested=ds_tested)
    write_train = write_train_task(df_trained=ds_trained)
    write_test = write_test_task(df_tested=ds_tested)

    create_db >> get_ds >> unzip >> df1_df2
    split_dfs['dftn'] >> df_prep >> df_rbl >> df_merged >> ds_trained >> train_scoring >> write_train
    split_dfs['dfts'] >> ds_tested >> test_scoring >> write_test

# вызов функции my_text_classification_dag(), которая декорирована как 
# DAG (Directed Acyclic Graph) с помощью декоратора @dag. 
# Этот вызов создаёт экземпляр DAG, который затем присваивается переменной DE_Diploma_dag.
DE_Diploma_dag = my_text_classification_dag()
