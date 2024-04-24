
# Импорт библиотек
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from urllib.parse import urljoin
from sklearn.model_selection import train_test_split                # разделение данных на обучающую и тестовую части
from sklearn.feature_extraction.text import TfidfVectorizer         # преобразование текста в вектор
from sklearn.linear_model import LogisticRegression                 # использование модели логистической регрессии
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
import text_clas_pkg

# Определение аргументов по умолчанию
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 4, 24),
    'email': ['my_email@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Определение DAG
dag = DAG(
    'my_model_training_pipeline',
    default_args=default_args,
    description='A simple pipeline for model training',
    schedule_interval=timedelta(days=1),
)

# Определение задач
t1 = PythonOperator(
    task_id='getting_datasets',
    python_callable=text_clas_pkg.getting_datasets,
    dag=dag,
)

t2 = PythonOperator(
    task_id='unzip_and_replace_dataset',
    python_callable=text_clas_pkg.unzip_and_replace_dataset,
    op_kwargs={'zip_path': "C:\\Users\\Allen\\Downloads\\archive.zip", 
               'extract_to': "D:\\GeekBrains\\Data_engineer_Diploma_project"},
    dag=dag,
)

t3 = PythonOperator(
    task_id='transforming_datasets',
    python_callable=text_clas_pkg.transforming_datasets,
    op_kwargs={'test_path': "D:\\GeekBrains\\Data_engineer_Diploma_project\\test.dat", 
               'train_path': "D:\\GeekBrains\\Data_engineer_Diploma_project\\train.dat", 
               'test_csv_path': "D:\\GeekBrains\\Data_engineer_Diploma_project\\ma_test.csv", 
               'train_csv_path': "D:\\GeekBrains\\Data_engineer_Diploma_project\\ma_train.csv"},
    dag=dag,
)

t4 = PythonOperator(
    task_id='prepare_dfs_to_labeling',
    python_callable=text_clas_pkg.prepare_dfs_to_labeling,
    op_kwargs={'df_train': "{{ ti.xcom_pull(task_ids='transforming_datasets') }}", 
               'manual_label_csv': 'manual_label_sample.csv', 
               'rule_based_csv': 'rule_based_sample.csv', 
               'train_size': 0.01},
    dag=dag,
)

t5 = PythonOperator(
    task_id='rule_based_labeling',
    python_callable=text_clas_pkg.rule_based_labeling,
    op_kwargs={'df_rbs': "{{ ti.xcom_pull(task_ids='prepare_dfs_to_labeling') }}"},
    dag=dag,
)

t6 = PythonOperator(
    task_id='merging_labeled_dfs',
    python_callable=text_clas_pkg.merging_labeled_dfs,
    op_kwargs={'df_rule': "{{ ti.xcom_pull(task_ids='rule_based_labeling') }}"},
    dag=dag,
)

t7 = PythonOperator(
    task_id='teaching_and_saving_model',
    python_callable=text_clas_pkg.teaching_and_saving_model,
    op_kwargs={'train_df': "{{ ti.xcom_pull(task_ids='merging_labeled_dfs') }}"},
    dag=dag,
)

t8 = PythonOperator(
    task_id='testing_model',
    python_callable=text_clas_pkg.testing_model,
    op_kwargs={'path_to_csv': "D:\\GeekBrains\\Data_engineer_Diploma_project\\ma_test.csv"},
    dag=dag,
)

t9 = PythonOperator(
    task_id='accuracy_scoring',
    python_callable=text_clas_pkg.accuracy_scoring,
    op_kwargs={'df_for_evaluation': "{{ ti.xcom_pull(task_ids='testing_model') }}"},
    dag=dag,
)

# Определение последовательности выполнения задач
t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7 >> t8 >> t9