# superset_config.py
import os

# Убедитесь, что SECRET_KEY генерируется с помощью надежного метода, например:
# openssl rand -base64 42
SECRET_KEY = os.environ.get('SUPERSET_SECRET_KEY', 'my_superset_secret_key')

# Установите URI базы данных для Superset
SQLALCHEMY_DATABASE_URI = os.environ.get('SUPERSET_DB_URI')