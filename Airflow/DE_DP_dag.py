
# Импорт библиотек

from urllib.parse import urljoin
from sklearn.model_selection import train_test_split    
from sklearn.feature_extraction.text import TfidfVectorizer    
from sklearn.linear_model import LogisticRegression                 
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump, load
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from airflow.models import Variable
from airflow.hooks.base_hook import BaseHook
from airflow.decorators import dag, task
from datetime import datetime, timedelta
import time
import zipfile
import os
import pandas as pd
import pendulum
import MySQLdb
from text_classification_pkg.text_classification_module import getting_dataset_by_api, unzip_and_replace_datasets,\
        transforming_datasets, prepare_dfs_to_labeling, rule_for_labeling,\
        rule_based_labeling, merging_labeled_dfs, teaching_and_saving_model,\
        testing_model, accuracy_scoring, create_database, write_dataframe_to_mysql

# Определение DAG с использованием декоратора @dag
@dag(
    'Medical_text_classification',
    default_args={
        'owner': 'AllenovNS',
        'depends_on_past': False,
        'start_date': pendulum.datetime(2024, 4, 25, tz='UTC'),
        'retries': 1,
        'retry_delay': timedelta(minutes=4),
    },
    description='A DAG to process and classification datasets contain\
        medical abstracts and store data in MySQL',
    schedule_interval=None,
    catchup=False,
    tags=['DE_Diploma_project'],
)

def my_text_classification_dag():
    # Использование декоратора @task для определения задачи

    @task
    def getting_dataset_by_api_task():
        getting_dataset_by_api()

    @task
    def unzip_and_replace_datasets_task():
        unzip_and_replace_datasets()

    @task
    def transforming_datasets_task():
        return transforming_datasets()
    
    @task(multiple_outputs=True)
    def split_dataframes(df1_df2):
        df_train, df_test = df1_df2
        return {'dftn': df_train, 'dfts': df_test}

    @task
    def prepare_dfs_to_labeling_task(df_train: str):
        return prepare_dfs_to_labeling(df_train)

    @task
    def rule_based_labeling_task(df_prep: str):
        return rule_based_labeling(df_prep)

    @task
    def merging_labeled_dfs_task(df_rbl: str):
        return merging_labeled_dfs(df_rbl)

    @task
    def teaching_and_saving_model_task(df_merged: str):
        return teaching_and_saving_model(df_merged)

    @task
    def testing_model_task(df_test: str):
        return testing_model(df_test)

    @task
    def train_accuracy_scoring_task(df_trained: str):
        accuracy_scoring(df_trained)

    @task
    def test_accuracy_scoring_task(df_tested: str):
        accuracy_scoring(df_tested)

    @task
    def create_db_task():
        create_database('airflow_db', 'DE_DP_text_classification')

    @task
    def write_train_task(df_trained: str):
        write_dataframe_to_mysql('train_df_with_predictions', df_trained, 'mysql_conn_id')

    @task
    def write_test_task(df_tested: pd.DataFrame):
        write_dataframe_to_mysql('test_df_with_predictions', df_tested, 'mysql_conn_id')

    
    # Установка зависимостей
    create_db = create_db_task()
    get_ds = getting_dataset_by_api_task()
    unzip = unzip_and_replace_datasets_task()
    
    # Зависимости для задач, связанных с обработкой и анализом данных
    df1_df2 = transforming_datasets_task()
    split_dfs = split_dataframes(df1_df2=df1_df2)
    df_prep = prepare_dfs_to_labeling_task(df_train=split_dfs['dftn'])
    df_rbl = rule_based_labeling_task(df_prep=df_prep)
    df_merged = merging_labeled_dfs_task(df_rbl=df_rbl)
    df_trained = teaching_and_saving_model_task(df_merged=df_merged)
    df_tested = testing_model_task(df_test=split_dfs['dfts'])
    train_scoring = train_accuracy_scoring_task(df_trained)
    test_scoring = test_accuracy_scoring_task(df_tested)
    write_train = write_train_task(df_trained)
    write_test = write_test_task(df_tested)

    # Установка порядка выполнения задач
    create_db >> get_ds >> unzip >> df1_df2
    df1_df2['dftn'] >> df_prep >> df_rbl >> df_merged >> df_trained >> train_scoring >> write_train
    df1_df2['dfts'] >> df_tested >> test_scoring >> write_test
    
# Создание экземпляра DAG
dag_instance = my_text_classification_dag()