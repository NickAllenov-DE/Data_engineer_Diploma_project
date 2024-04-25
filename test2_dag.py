
# Импорт библиотек

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from urllib.parse import urljoin
from sklearn.model_selection import train_test_split    
from sklearn.feature_extraction.text import TfidfVectorizer    
from sklearn.linear_model import LogisticRegression                 
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump, load
from datetime import datetime, timedelta

import time
import zipfile
import os
import pandas as pd
from sqlalchemy import create_engine
from airflow.models import Variable
from text_classification_module import *
from airflow.hooks.base_hook import BaseHook
from airflow.decorators import dag, task
from datetime import datetime, timedelta
import pendulum
import MySQLdb




# Определение DAG с использованием декоратора @dag
@dag(
    'Medical_text_classification',
    default_args={
        'owner': 'AllenovNS',
        'depends_on_past': False,
        'start_date': pendulum.datetime(2024, 4, 25, tz='UTC'),
        'retries': 1,
        'retry_delay': timedelta(minutes=2),
    },
    description='A DAG to process and classification datasets contain medical abstracts and store data in MySQL',
    schedule_interval=None,
    catchup=False,
    tags=['DE_Diploma_project'],
)

def my_text_classification_dag():
    # Использование декоратора @task для определения задачи
    @task
    def create_db_task():
        create_database('DE_DP_text_classification')

    @task
    def write_train_task():
        write_dataframe_to_mysql('train_df_with_predictions', 'ma_train_with_predictions.csv')

    @task
    def write_test_task():
        write_dataframe_to_mysql('test_df_with_predictions', 'ma_test_with_predictions.csv')


    # Определение других задач с использованием @task
    # ...

    # Установка зависимостей
    create_db = create_db_task()
    write_train = write_train_task()
    write_test = write_test_task()

    create_db >> [write_train, write_test]

# Создание экземпляра DAG
dag_instance = my_text_classification_dag()