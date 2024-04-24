# Импорт библиотек
def import_libraries() -> None:
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
    from datetime import datetime, timedelta
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import time
    import zipfile
    import os
    import pandas as pd
