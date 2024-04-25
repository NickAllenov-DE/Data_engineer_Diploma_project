
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
from joblib import dump, load
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import zipfile
import os
import pandas as pd
from text_classification_module import *

default_args = {
    'owner': 'AllenovNS',
    'depends_on_past': False,
    'start_date': datetime(2023, 4, 25),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

dag = DAG(
    'GB_DE_Diploma_Project_pipeline',
    default_args=default_args,
    description='A DAG to process and classification datasets contain\
        medical abstracts and store data in MySQL',
    schedule_interval=None,
    catchup=False
)

# Определение задач для Airflow
task_getting_datasets = PythonOperator(
    task_id='getting_datasets',
    python_callable=getting_datasets,
    dag=dag,
)

task_unzip_and_replace_datasets = PythonOperator(
    task_id='unzip_and_replace_datasets',
    python_callable=unzip_and_replace_datasets,
    dag=dag,
)

task_transforming_datasets = PythonOperator(
    task_id='transforming_datasets',
    python_callable=transforming_datasets,
    dag=dag,
)

task_merging_labeled_dfs = PythonOperator(
    task_id='merging_labeled_dfs',
    python_callable=merging_labeled_dfs,
    dag=dag,
)

task_teaching_and_saving_model = PythonOperator(
    task_id='teaching_and_saving_model',
    python_callable=teaching_and_saving_model,
    dag=dag,
)

task_testing_model = PythonOperator(
    task_id='testing_model',
    python_callable=testing_model,
    dag=dag,
)

task_accuracy_scoring = PythonOperator(
    task_id='accuracy_scoring',
    python_callable=accuracy_scoring,
    dag=dag,
)

task_write_to_mysql = PythonOperator(
    task_id='write_to_mysql',
    python_callable=write_to_mysql,
    dag=dag,
)

# Определение порядка выполнения задач
task_getting_datasets >> task_unzip_and_replace_datasets >> task_transforming_datasets
task_transforming_datasets >> [task_merging_labeled_dfs, task_write_to_mysql]
task_merging_labeled_dfs >> task_teaching_and_saving_model >> task_accuracy_scoring
task_write_to_mysql >> task_testing_model >> task_accuracy_scoring