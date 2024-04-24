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






# import org.apache.spark.sql.SparkSession
# import org.apache.spark.sql.SaveMode

# object MySQLSparkExample {
#   def main(args: Array[String]): Unit = {
#     val spark = SparkSession.builder()
#       .appName("MySQLSparkExample")
#       .getOrCreate()

#     // Параметры подключения к базе данных MySQL
#     val jdbcUrl = "jdbc:mysql://localhost:3306/"
#     val dbProperties = new java.util.Properties()
#     dbProperties.setProperty("user", "yourUsername")
#     dbProperties.setProperty("password", "yourPassword")
#     dbProperties.setProperty("driver", "com.mysql.jdbc.Driver")

#     // Создание базы данных (если она еще не создана)
#     val connection = java.sql.DriverManager.getConnection(jdbcUrl, "yourUsername", "yourPassword")
#     try {
#       val statement = connection.createStatement()
#       statement.executeUpdate("CREATE DATABASE IF NOT EXISTS myDatabase")
#     } finally {
#       connection.close()
#     }

#     // Загрузка датасетов в Spark
#     val dataset1 = spark.read.json("path_to_your_dataset1.json")
#     val dataset2 = spark.read.json("path_to_your_dataset2.json")

#     // Запись датасетов в таблицы MySQL
#     dataset1.write
#       .mode(SaveMode.Overwrite)
#       .jdbc(jdbcUrl + "myDatabase", "dataset1Table", dbProperties)

#     dataset2.write
#       .mode(SaveMode.Overwrite)
#       .jdbc(jdbcUrl + "myDatabase", "dataset2Table", dbProperties)

#     spark.stop()
#   }
# }