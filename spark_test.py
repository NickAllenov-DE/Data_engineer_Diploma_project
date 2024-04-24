from pyspark.sql import SparkSession

def create_and_fill_database(spark: SparkSession, train_df_path: str, test_df_path: str, database_name: str, train_table_name: str, test_table_name: str):
    # Создание базы данных
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {database_name}")
    spark.sql(f"USE {database_name}")

    # Загрузка датафреймов из файлов CSV
    train_df = spark.read.csv(train_df_path, header=True, inferSchema=True)
    test_df = spark.read.csv(test_df_path, header=True, inferSchema=True)

    # Запись датафреймов в таблицы базы данных
    train_df.write.saveAsTable(train_table_name, mode='overwrite')
    test_df.write.saveAsTable(test_table_name, mode='overwrite')


from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

# Определение задачи для создания базы данных и заполнения её датафреймами
create_db = SparkSubmitOperator(
    task_id='create_and_fill_database',
    application='/path/to/your/spark_application.py',  # Путь к вашему Spark приложению
    name='create_and_fill_database',
    conn_id='spark_default',  # Идентификатор соединения Spark, настроенный в Airflow
    verbose=False,
    conf={'spark.yarn.queue': 'root.default'},
    application_args=[
        "{{ ti.xcom_pull(task_ids='teaching_and_saving_model') }}",
        "{{ ti.xcom_pull(task_ids='testing_model') }}",
        'my_database',
        'train_table',
        'test_table'
    ],
    dag=dag,
)

# Определение последовательности выполнения задач
t7 >> create_db
t8 >> create_db