import MySQLdb
from airflow.operators.python_operator import PythonOperator
from airflow.hooks.base_hook import BaseHook
from airflow.decorators import dag, task
from datetime import datetime, timedelta
import pendulum

# Определение DAG с использованием декоратора @dag
@dag(
    'GB_DE_Diploma_Project_pipeline',
    default_args={
        'owner': 'AllenovNS',
        'depends_on_past': False,
        'start_date': pendulum.datetime(2024, 4, 25, tz='UTC'),
        'retries': 1,
        'retry_delay': timedelta(minutes=1),
    },
    description='A DAG to process and classification datasets contain medical abstracts and store data in MySQL',
    schedule_interval=None,
    catchup=False,
    tags=['example'],
)

def my_text_classification_dag():
    # Использование декоратора @task для определения задачи
    @task
    def create_database(mysql_conn_id: str, database_name: str):
        # Подключение к MySQL
        import MySQLdb
        # Получение параметров подключения из Airflow
        connection_params = BaseHook.get_connection(mysql_conn_id) 
        connection = MySQLdb.connect(
            user=connection_params.login,
            passwd=connection_params.password,
            host=connection_params.host
        )
        cursor = connection.cursor()
        # Создание базы данных
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name};")
        # Закрытие соединения
        cursor.close()
        connection.close()
        print(f"Database {database_name} created successfully.")

    # Определение других задач с использованием @task
    # ...

    # Вызов задач
    create_database('airflow_db', 'DE_DP_text_classification')
    # Остальные задачи могут быть добавлены здесь


@dag(
    'write_df_to_mysql_dag',
    default_args={
        'owner': 'AllenovNS',
        'start_date': pendulum.datetime(2024, 4, 25, tz='UTC'),
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description='Write Pandas DataFrames to MySQL',
    schedule_interval=None,
    catchup=False
)
def write_train_df_to_mysql():
    write_dataframe_to_mysql('train_table', '/path/to/ma_train_with_predictions.csv', 'mysql_conn_id')

@task
def write_test_df_to_mysql():
    write_dataframe_to_mysql('test_table', '/path/to/ma_test_with_predictions.csv', 'mysql_conn_id')

train_df_to_mysql = write_train_df_to_mysql()
test_df_to_mysql = write_test_df_to_mysql()

# Создание экземпляра DAG
dag_instance = my_text_classification_dag()
