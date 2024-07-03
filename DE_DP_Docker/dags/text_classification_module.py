# Импорт библиотек

# Kaggle API для доступа к датасетам на Kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

# urllib для работы с URL
from urllib.parse import urljoin

# sklearn для машинного обучения:
from sklearn.model_selection import train_test_split  # Разделение данных на обучающую и тестовую выборки
from sklearn.feature_extraction.text import TfidfVectorizer  # Векторизация текста с использованием TF-IDF
from sklearn.linear_model import LogisticRegression  # Логистическая регрессия для классификации
from sklearn.utils import shuffle  # Перемешивание данных
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Метрики для оценки модели

# joblib для сохранения и загрузки модели
from joblib import dump, load

# datetime для работы с датами и временем
from datetime import datetime, timedelta

# Работа с файлами и директориями
import zipfile
import os

# pandas для работы с данными в табличном виде
import pandas as pd

# sqlalchemy для работы с базами данных через SQL выражения
from sqlalchemy import create_engine

# Airflow для организации и управления рабочими процессами:
from airflow.models import Variable  # Работа с переменными Airflow
from airflow.providers.postgres.hooks.postgres import PostgresHook  # Базовый класс для всех хуков в Airflow

# psycopg2 и sqlalchemy для работы с PostgreSQL:
import psycopg2  # Библиотека для работы с PostgreSQL
from psycopg2 import sql  # Модуль для безопасного формирования SQL запросов


def getting_dataset_by_api(ds_name: str = 'chaitanyakck/medical-text', path: str = os.getcwd()):
    """
    Downloads a dataset from Kaggle using the Kaggle API and unpacks it to the specified directory.
    Arguments:
    ds_name (str): The name of the dataset on Kaggle. The default is 'chaitanyakck/medical-text'.
    path (str): The path to the directory where the dataset files will be uploaded. By default, the current working directory.
    Return value: None
    """

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(ds_name, path, unzip=True)
    

def unzip_and_replace_datasets(zip_path: str ="archive.zip", extract_to: str = os.getcwd()) -> None:
    '''
    The function unzips the downloaded archive to the working directory.
    
    Parameters:
    zip_path (str): The path to the ZIP archive to be unzipped. By default "archive.zip ".
    extract_to (str): The path to the directory where the contents of the archive will be extracted. By default, the current working directory.

    Return value:
    the function returns nothing.
    '''

    # Проверка существования файла
    if not os.path.exists(zip_path):
        # Если файл не существует, выводим сообщение об ошибке и завершаем выполнение функции
        print(f"The file {zip_path} does not exist.")
        return

    # Разархивирование архива
    try:
        # Открытие zip-файла для чтения
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Извлечение всех файлов из архива в указанную директорию
            zip_ref.extractall(extract_to)
        # Сообщение об успешном извлечении архива
        print(f"Archive extracted to {extract_to}")
    except zipfile.BadZipFile:
        # Обработка ошибки поврежденного архива
        print("The file is a bad zip file and cannot be extracted.")
    except Exception as e:
        # Обработка любых других ошибок, возникших при извлечении архива
        print(f"An error occurred: {e}")
 

def transforming_datasets(train_path: str = "train.dat", test_path: str = "test.dat",
                          test_csv_path: str = "ma_test.csv", train_csv_path: str = "ma_train.csv") -> tuple[str, str]:
    '''The function opens downloaded files, generates datasets adapted to processing based on them, and saves new datasets in .csv format'''

    # В архиве датасеты (тренировочный и тестовый) содержатся в формате .dat
    # поэтому, нам нужно их переформатировать в датасеты, пригодные и удобные для дальнейшего использования
    
    # Функция использует pd.read_fwf для чтения файлов .dat и преобразует их в датафреймы df_train и df_test. 
    # Параметр sep='\t' указывает, что данные в файле разделены табуляцией, а параметр header=None указывает, 
    # что в файле нет заголовков столбцов. 
    df_train = pd.read_fwf(train_path, sep='\t', header=None)
    df_test = pd.read_fwf(test_path, sep='\t', header=None)
    
    # Датафрейм df_test имеет атипичную ненормализованную структуру - всего 101 столбец, все аннотации содержатся в первом столбце, 
    # остальные колонки пустые, поэтому нам нужно создать датафрейм только из первой колонки.
    # Датафрейм df_train имеет схожую структуру - 101 столбец, первая колонка - классы заболеваний,
    # все аннотации содержатся во втором столбце, остальные колонки пустые.
    # Для нашей дальнейшей работы метки классов нам не требуются (потому что они неправильные),
    # поэтому нам нужно создать датафрейм из второго столбца.

    # Датафрейм df_train содержит 101 столбец, где вторая колонка (с индексом 1) содержит аннотации. 
    # Функция выбирает эту колонку и переименовывает ее в abstracts. 
    df_ma_train = df_train.iloc[:, [1]].rename(columns={1: 'abstracts'})

    # Датафрейм df_test также содержит 101 столбец, но аннотации находятся в первой колонке (с индексом 0). 
    # Функция выбирает эту колонку и переименовывает ее в abstracts. 
    df_ma_test = df_test.iloc[:, [0]].rename(columns={0: 'abstracts'})

    # Преобразованные датафреймы df_ma_train и df_ma_test сохраняются в формате .csv по указанным путям. 
    # Параметры index=False и header=['abstracts'] указывают, что индексы строк не будут сохраняться, 
    # а столбец будет иметь заголовок abstracts. 
    df_ma_train.to_csv(train_csv_path, index=False, header=['abstracts'])
    df_ma_test.to_csv(test_csv_path, index=False, header=['abstracts'])    

    # Функция возвращает кортеж из двух строк, содержащих пути к сохраненным файлам .csv для 
    # тренировочного и тестового датасетов. 
    return train_csv_path, test_csv_path


def prepare_dfs_to_labeling(path_to_ds_csv: str, manual_label_csv: str = 'manual_label_sample.csv', 
                        rule_based_csv: str = 'rule_based_sample.csv', train_size: float = 0.01) -> str:
    """
    Divides the dataframe into two parts for manual markup and for automatic rule-based markup.
    Returns dataframe for automatic rule-based markup.
    
    Parameters:
    path_to_ds_csv (str): Path to the CSV file containing the dataset.
    manual_label_csv (str): Path to the CSV file for manual labeling.
    rule_based_csv (str): Path to the CSV file for rule-based labeling.
    train_size (float): Proportion of the dataframe to be used for manual labeling.
    
    Returns:
    str: Path to the CSV file for rule-based labeling.
    """
    
    # Чтение датафрейма из CSV файла
    df_train = pd.read_csv(path_to_ds_csv)

    # Разделение датафрейма на две части - для ручной разметки и для разметки на основе правил
    manual_label_sample, rule_based_sample = train_test_split(df_train, train_size=train_size, random_state=42)

    # Сохранение датафреймов в файлы .csv для дальнейшей обработки
    manual_label_sample.to_csv(manual_label_csv, index=False)
    rule_based_sample.to_csv(rule_based_csv, index=False)

    return rule_based_csv


def rule_for_labeling(text: str) -> int:
    '''The function defines a rule for assigning a label to the text and performs markup'''
    
    # Определяем списки с ключевыми значениями по каждой из четырех категорий 
    neoplasms_list = [
        'neoplas', 'tumor', 'cancer', 'lymphom', 'blastoma', 'malign', 'benign', 'melanom', 'leukemi', 'metasta', 'carcinom', 'sarcoma', 'glioma',
        'adenoma', 'chemotherapy', 'radiotherapy', 'oncology', 'carcinogenesis', 'mutagen', 'angiogenesis', 'radiation', 'immunotherapy', 'biopsy',
        'brachytherapy', 'metastasis', 'prognosis', 'biological therapy', 'carcinoma', 'myeloma', 'genomics', 'immunology', 'cell stress',
        'oncogene', 'tumorigenesis', 'cytology', 'histology', 'oncologist', 'neoplasm', 'oncogenic', 'tumor suppressor genes', 'malignancy',
        'cancerous', 'non-cancerous', 'solid tumor', 'liquid tumor', 'tumor marker', 'oncogenesis', 'tumor microenvironment', 'carcinogenesis', 
        'adenocarcinoma', 'squamous cell carcinoma'
    ]

    digestive_list = [
        'digestive', 'esophag', 'stomach', 'gastr', 'liver', 'cirrhosis', 'hepati', 'pancrea', 'intestin', 'sigmo', 'recto', 'rectu', 'cholecyst', 
        'gallbladder', 'portal pressure', 'portal hypertension', 'appendic', 'ulcer', 'bowel', 'dyspepsia', 'colitis', 'enteritis', 'gastroenteritis', 
        'endoscopy', 'colonoscopy', 'peptic', 'gastrointestinal', 'abdominal', 'dysphagia', 'diverticulitis', 'irritable bowel syndrome', 
        'inflammatory bowel disease', 'gastroesophageal reflux', 'celiac disease', 'crohn\'s disease', 'ulcerative colitis',
        'gastroscopy', 'biliary', 'esophageal', 'gastritis', 'hepatic', 'lactose intolerance', 'gastroenterologist', 'digestion', 'absorption', 
        'malabsorption', 'intestinal flora', 'microbiota', 'probiotics', 'prebiotics', 'dietary fiber', 'nutrition'
    ]

    neuro_list = [
        'neuro', 'nerv', 'reflex', 'brain', 'cerebr', 'white matter', 'subcort', 'plegi', 'intrathec', 'medulla', 'mening', 'epilepsy', 
        'multiple sclerosis', 'parkinson\'s disease', 'alzheimer\'s disease', 'seizure', 'paresthesia', 'dementia', 'encephalopathy', 
        'neuropathy', 'neurodegeneration', 'stroke', 'cerebral', 'spinal cord', 'neurotransmitter', 'synapse', 'neuralgia', 'neurology', 
        'neurosurgery', 'neurooncology', 'neurovascular', 'autonomic nervous system', 'central nervous system', 'peripheral nervous system', 
        'brain injury', 'concussion', 'traumatic brain injury', 'spinal injury', 'neurological disorder', 'neurodevelopmental disorders',
        'neurodegenerative disorders', 'neuroinflammation', 'neuroimaging', 'neuroscience', 'neurophysiology', 'neurotransmission', 
        'neuroplasticity', 'neurogenesis', 'neuroendocrinology', 'neuropsychology', 'neurotoxicity', 'neuromodulation', 'neuroprotection', 
        'neuropathology'
    ]

    cardio_list = [
        'cardi', 'heart', 'vascul', 'embolism', 'stroke', 'reperfus', 'thromboly', 'ischemi', 'hypercholesterolemia', 'hyperten', 'blood pressure', 
        'valv', 'ventric', 'aneurysm', 'coronar', 'arter', 'aort', 'electrocardiogra', 'arrhythm', 'clot', 'mitral', 'endocard', 'hypertension', 
        'myocardial', 'infarction', 'cardiover', 'fibrillat', 'bypass', 'pericarditis', 'cardiomyopathy', 'hypotension', 'angiography', 'stenting', 
        'cardiac catheterization', 'vascular', 'echocardiogram', 'cardiogenic', 'angioplasty', 'cardiac arrest', 'heart failure', 
        'cardiac rehabilitation', 'electrophysiology', 'heart valve disease', 'cardiopulmonary', 'cardiothoracic surgery', 'vascular surgery', 
        'cardiovascular disease', 'cardiovascular health', 'cardiovascular risk', 'cardiovascular system', 'cardioprotection', 'cardiovascular imaging', 
        'cardiovascular physiology', 'cardiovascular pharmacology', 'cardiovascular intervention', 'cardiovascular diagnostics', 'cardiovascular genetics'
    ]

    # Приведем текст аннотаций к нижнему регистру
    row = text.lower()
    
    # В используемом датасете используется следующая маркировка:
    # neoplasms = 1
    # digestive system diseases = 2
    # nervous system diseases = 3
    # cardiovascular diseases = 4
    # general pathological conditions = 5

    # Создаём словарь в котором ключи - категории заболеваний, а значения - количество ключевых значений в тексте по каждой категории
    res_dict = {
        '1': 0,
        '2': 0,
        '3': 0,
        '4': 0
    }
    # Рассчитываем количество ключевых значений в тексте и заполняем словарь
    for p in neoplasms_list:
        res_dict['1'] += row.count(p)
    for d in digestive_list:
        res_dict['2'] += row.count(d)
    for n in neuro_list:
        res_dict['3'] += row.count(n)
    for c in cardio_list:
        res_dict['4'] += row.count(c)
    
    # Рассчитываем наиболее часто встречаемую категорию в тексте и её отношение ко всем выявленным значения по всем категориям.
    # Для отнесения текста к определенной категории его доля должна превышать условно взятое значение - 0,3.
    # Если не превышает, то текст будет отнесён к категории 'general pathological conditions' и ему будет присвоена метка - 5
    most_frequent = max(res_dict.values())
    divisor = sum(res_dict.values())
    if divisor > 0 and (most_frequent / divisor) > 0.3:
        for key, value in res_dict.items(): 
            if value == most_frequent:
                return int(key)
    else:
        return int(5)
    

def rule_based_labeling(df_rbs_path: str, df_rbs_csv: str = 'df_rule_labeled.csv') -> str:
    '''
    The function performs the markup of the dataframe - we add a column to the dataframe, 
    in which there will be labels based on a rule defined by us.
    
    Parameters:
    df_rbs_path (str): Путь к входному CSV файлу, содержащему датафрейм для разметки.
    df_rbs_csv (str): Путь к выходному CSV файлу, содержащему размеченный датафрейм.
    
    Returns:
    str: Путь к выходному CSV файлу.
    '''

    # Функция принимает путь к CSV файлу, содержащему датафрейм для разметки, и считывает 
    # его с помощью pd.read_csv(df_rbs_path). Этот датафрейм должен содержать столбец с 
    # текстами (аннотациями), которые нужно разметить.
    df_rbs = pd.read_csv(df_rbs_path)

    # Для каждого текста в столбце abstracts применяется функция rule_for_labeling, 
    # которая присваивает тексту метку на основе содержания. 
    # Новая метка добавляется в столбец labeled_condition_mark.
    df_rbs['labeled_condition_mark'] = df_rbs['abstracts'].apply(rule_for_labeling)

    # Размеченный датафрейм сохраняется в новый CSV файл с помощью 
    # df_rbs.to_csv(df_rbs_csv, index=False). Название выходного файла передаётся 
    # через параметр df_rbs_csv, который по умолчанию имеет значение 'df_rule_labeled.csv'.
    df_rbs.to_csv(df_rbs_csv, index=False)

    # Функция возвращает путь к сохранённому CSV файлу, содержащему размеченный датафрейм.
    return df_rbs_csv


# Ручную разметку выборки выдержек из медицинских статей в размере 144 шт (0,01 от всего датасета) я провел в Label Studio,
# с использованием маркировки цифровыми значениями. 
# Результат разметки сохранен в текущую директорию с именем ls_manual_labeled.csv.

# Объединение датасетов, если есть датасет, размеченный вручную, 
# и приведение их к виду который будет использоваться для обучения модели

def merging_labeled_dfs(df_rule_path: str, merge_df_csv: str = 'df_merged.csv') -> str:
    '''The function combines the date frames obtained as a result of automatic 
    rule-based markup and manual markup and brings the combined dataframe to the 
    form in which it will be used to train the model'''

    df_rule = pd.read_csv(df_rule_path)

    # Проверка наличия датасета размеченного вручную в текущей директории
    # Поскольку процесс будет выполняться автоматизированно, этап ручной разметки может быть исключен из процесса,
    # либо выполняться не при каждом запуске процесса
    dset_name = 'ls_manual_labeled.csv'  
    dset_exists = os.path.exists(dset_name)

    if dset_exists:
        df_manual = pd.read_csv('ls_manual_labeled.csv')
        # Для объединения датасетов приведем датасет созданный Label Studio к соответствующему виду:
        df_manual.drop(['annotation_id', 'annotator', 'created_at', 'id', 'lead_time', 'updated_at'], axis=1, inplace=True)
        df_manual.rename(columns={'sentiment': 'labeled_condition_mark'}, inplace=True)
        # Теперь объединим датасеты:
        df_merged = pd.concat([df_rule, df_manual])
    # Если датасета размеченного вручную в текущей директории нет,
    # то итоговым датафреймом будет датафрейм, размеченный на основе правила
    else:
        df_merged = df_rule

    # Сохраним результирующий датасет
    df_merged.to_csv(merge_df_csv, index=False)

    return merge_df_csv


def teaching_and_saving_model(train_df_path: str, trained_df_csv: str = 'ma_train_with_predictions.csv') -> str:
    '''The function trains a machine learning model on a marked-up dataset, saves the model and 
    a vectorizer for further use, and returns a dataframe with the markup'''

    # Датасет, содержащий размеченные тексты, загружается из указанного CSV файла.
    train_df = pd.read_csv(train_df_path)

    # Для начала, перемешаем датасет. Датасет перемешивается для случайного 
    # распределения данных, что помогает улучшить обучение модели.
    train_df = shuffle(train_df)

    # Разделение данных на тексты и метки:
    # Тексты (X) и метки (Y) извлекаются из соответствующих колонок датафрейма.
    X = train_df['abstracts']
    Y = train_df['labeled_condition_mark']

    # Создаем векторизатор и преобразуем тексты в векторы. 
    # Используется TfidfVectorizer для преобразования текстов в векторы признаков. 
    # Этот шаг необходим для того, чтобы модель машинного обучения могла 
    # обрабатывать текстовые данные.
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    # Обучаем модель.
    # Модель логистической регрессии обучается на векторизованных данных. 
    # max_iter=15000 указывает максимальное количество итераций для оптимизации модели.
    model = LogisticRegression(max_iter=15000)
    model.fit(X_vectorized, Y)

    # Модель делает предсказания меток на тех же данных, на которых она была обучена.
    Y_predicted = model.predict(X_vectorized)

    # Добавление колонки с предсказанными значениями в датафрейм.
    train_df['predicted_mark'] = Y_predicted

    # Сохранение датасета с предсказанными значениями
    train_df.to_csv(trained_df_csv, index=False)

    # Сохранение модели и векторизатора
    dump(model, 'model_ma_trained.joblib')
    dump(vectorizer, 'vectorizer_ma_trained.joblib')

    return trained_df_csv


def testing_model(path_to_ds_csv: str, tested_df_csv: str = 'ma_test_with_predictions.csv') -> str:
    '''The function loads a trained machine learning model and applies it to an untagged dataframe'''

    # Обученная модель машинного обучения и векторизатор загружаются из сохранённых файлов.
    # Загрузка модели.
    model = load('model_ma_trained.joblib')

    # Загрузка векторизатора
    vectorizer = load('vectorizer_ma_trained.joblib')

    # Новый тестовый датасет загружается из указанного CSV файла.
    df_test = pd.read_csv(path_to_ds_csv)

    # Преобразование текстовых данных нового датафрейма в векторный формат
    X_new = vectorizer.transform(df_test['abstracts'])
    
    # Используем модель для предсказания меток новых данных
    Y_new_predicted = model.predict(X_new)

    # разметка тестового датасета на основе правил для послдующей
    # оценки эффективности модели
    df_test = rule_based_labeling(df_test)

    # Добавление колонки с предсказанными значениями в датафрейм
    df_test['predicted_mark'] = Y_new_predicted
   
    # Сохранение датасета с размеченными и предсказанными значениями
    df_test.to_csv('ma_test_with_predictions.csv', index=False)

    return tested_df_csv



def accuracy_scoring(df_for_evaluation_path: str):
    '''The function evaluates the effectiveness of the machine learning model 
    and saves results into .txt files'''
    
    # Датафрейм, содержащий истинные метки и предсказанные значения, 
    # загружается из CSV файла.
    df_for_evaluation = pd.read_csv(df_for_evaluation_path)

    # Извлекаются истинные метки (true_labels) и предсказанные 
    # значения (predicted_labels) из датафрейма.
    true_labels = df_for_evaluation['labeled_condition_mark']
    predicted_labels = df_for_evaluation['predicted_mark']
    
    # Вычисление точности модели как доли правильно предсказанных меток.
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Генерируется подробный отчёт о классификации, включающий метрики, 
    # такие как точность, полнота и F1-оценка для каждой категории.
    report = classification_report(true_labels, predicted_labels)

    # Построение матрицы ошибок, показывающей количество правильных 
    # и ошибочных предсказаний для каждой категории.
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Получение текущей даты и времени
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # Сохранение точности с датой и временем
    with open('accuracy.txt', 'a') as f:
        f.write(f"Accuracy: {accuracy} (Дата и время: {formatted_time})\n")

    # Отчёт о классификации сохраняется в файл classification_report.txt, включая дату и время.
    with open('classification_report.txt', 'a') as f:
        f.write(f"Отчет о классификации (Дата и время: {formatted_time}):\n{report}\n")

    # Матрица ошибок сохраняется в файл confusion_matrix.txt, включая дату и время.
    with open('confusion_matrix.txt', 'a') as f:
        f.write(f"\nМатрица ошибок (Дата и время: {formatted_time}):\n")
        for line in conf_matrix:
            f.write(' '.join(str(x) for x in line) + '\n')



def create_postgres_database(postgres_conn_id: str, database_name: str) -> None:
    '''
    This function is designed to create a database in PostgreSQL, if it does not already exist. 
    It uses a connection whose parameters are taken from Apache Air flow. 
    The function uses PostgresHook to get the database connection parameters 
    and psycopg2 to execute SQL commands.

    Returns: nothing
    '''
    
    # Используется PostgresHook из Airflow для получения параметров подключения к PostgreSQL.
    hook = PostgresHook(postgres_conn_id=postgres_conn_id)
    connection_params = hook.get_connection()
    
    # Создается подключение к базе данных postgres. 
    # Включается режим autocommit для выполнения команд без явного вызова commit(). 
    # Создается курсор для выполнения SQL-запросов.
    connection = psycopg2.connect(
        user=connection_params.login,
        password=connection_params.password,
        host=connection_params.host,
        port=connection_params.port,
        dbname='postgres'
    )
    connection.autocommit = True
    cursor = connection.cursor()
    
    # Выполняется запрос для проверки наличия базы данных с 
    # именем database_name в системе PostgreSQL. 
    # Если запрос вернул результат, значит база данных существует, иначе — нет.
    cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database_name,))
    exists = cursor.fetchone()

    # Если база данных не существует (exists равен None), выполняется команда CREATE DATABASE. 
    # Выводится сообщение о создании базы данных или о её существовании.
    if not exists:
        cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))
        print(f"Database {database_name} created successfully.")
    else:
        print(f"Database {database_name} already exists.")
    
    # Закрывается курсор и соединение для освобождения ресурсов.
    cursor.close()
    connection.close()


def write_dataframe_to_postgres(table_name: str, df_path: str, postgres_conn_id: str, database_name: str) -> None:
    '''
    The function is designed to write the contents of the final dataframe stored in a CSV file 
    to a PostgreSQL database table. It uses Airflow to get connection parameters and 
    the pandas library to process data.

    Returns: nothing
    '''

    # Блок try-except используется для обработки потенциальных ошибок во время выполнения функции. 
    # В случае возникновения исключения, выводится сообщение об ошибке.
    try:

        # Используется PostgresHook из Airflow для получения параметров подключения к PostgreSQL.
        hook = PostgresHook(postgres_conn_id=postgres_conn_id)
        connection_params = hook.get_connection(postgres_conn_id)

        # Создается строка подключения для SQLAlchemy, используя параметры соединения.
        conn_str = f"postgresql+psycopg2://{connection_params.login}:{connection_params.password}" \
                   f"@{connection_params.host}:{connection_params.port}/{database_name}"
        
        # create_engine(conn_str) создает объект engine, который используется для управления соединением с базой данных.
        engine = create_engine(conn_str)

        # Чтение данных из указанного CSV-файла с помощью pandas.
        df = pd.read_csv(df_path)

        # Использование контекстного менеджера для управления соединением. 
        # Это обеспечивает автоматическое закрытие соединения после завершения операций.
        with engine.begin() as conn:
            df.to_sql(name=table_name, con=conn, if_exists='replace', index=False, method='multi')

        # Сообщение об успешной записи данных.
        print(f"Dataframe is written to PostgreSQL table {table_name} successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == "__main__":
    
    getting_dataset_by_api()
    unzip_and_replace_datasets()

    df_train, df_test = transforming_datasets()

    df_prep = prepare_dfs_to_labeling(df_train)

    df_rbl = rule_based_labeling(df_prep)
    
    df_merged = merging_labeled_dfs(df_rbl)

    df_train_with_predictions = teaching_and_saving_model(df_merged)
    accuracy_scoring(df_train_with_predictions)

    df_test_with_predictions = testing_model(df_test)
    accuracy_scoring(df_test_with_predictions)

    create_postgres_database('airflow_dp_db', 'DE_DP_text_classification')

    write_dataframe_to_postgres(df_train_with_predictions)
    write_dataframe_to_postgres(df_test_with_predictions)

