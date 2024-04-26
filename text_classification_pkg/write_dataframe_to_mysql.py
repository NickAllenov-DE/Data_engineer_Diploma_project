from sqlalchemy import create_engine
from airflow.models import Variable
from airflow.hooks.base_hook import BaseHook
import pandas as pd

def write_dataframe_to_mysql(table_name: str, df_path: str, mysql_conn_id: str):
    # Получение параметров подключения из Airflow
    connection_params = BaseHook.get_connection(mysql_conn_id)
    conn_str = f"mysql+mysqldb://{connection_params.login}:{connection_params.password}" \
               f"@{connection_params.host}/{connection_params.schema}"
    engine = create_engine(conn_str)

    df = pd.read_csv(df_path)

    # Запись датафрейма в базу данных MySQL
    df.to_sql(name=table_name, con=engine, if_exists='replace', index=False)

    print(f"Dataframe is written to MySQL table {table_name} successfully.")
