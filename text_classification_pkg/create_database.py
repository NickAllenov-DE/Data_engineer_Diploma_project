# Создание базы данных
def create_database(mysql_conn_id: str, database_name: str):
    # Подключение к MySQL
    import MySQLdb
    from airflow.hooks.base_hook import BaseHook
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