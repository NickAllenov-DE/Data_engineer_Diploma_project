
def getting_dataset_by_api(ds_name: str = 'chaitanyakck/medical-text', path: str = os.getcwd()):
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(ds_name, path, unzip=True)


def getting_datasets() -> None:
    '''The result of executing this function is a dataset downloaded into the directory "Downloads"'''

    # Инициализируем WebDriver:
    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 YaBrowser/24.1.0.0 Safari/537.36'
    chrome_options = Options()
    chrome_options.add_argument(f'user-agent={USER_AGENT}')
    driver = webdriver.Chrome()

    main_url = 'https://www.kaggle.com'
    sign_in_url = 'https://www.kaggle.com/account/login'
    # Датасет находится на странице: https://www.kaggle.com/datasets/chaitanyakck/medical-text/data
    dataset_url = "https://www.kaggle.com/datasets/chaitanyakck/medical-text/data"

    try:
        # Перейдём на страницу входа в аккаунт Kaggle для авторизированного скачивания датасета
        driver.get(sign_in_url)

        # Нажмём на кнопку "Sign in with Google" для авторизации (подразумевая, что аккаунт уже зарегистрирован)
        google_sign_in_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//button[contains(., "Sign in with Google")]'))
            )        
        google_sign_in_button.click()

        # Добавим ожидание с целью снижения нагрузки на сервер.
        time.sleep(3)

        try:
            # Проверим, что мы вошли в аккаунт и можем скачать датасет под своим аккаунтом.
            if driver.find_element((By.XPATH, '//h1[contains(., "Welcome")]')):

                # Теперь мы вошли в систему и можем переходить к дальнейшим действиям.
                # Откроем веб-страницу с датасетом:
                driver.get(dataset_url)

                # Найдём кнопку "Download":
                download_button = driver.find_element(By.XPATH, '//button[contains(., "file_download")]')
                # Если кнопка найдена, нажмём на неё
                if download_button:
                    download_button.click()
                # В противном случае воспользуемся альтернативным способом получения датасета - 
                # непосредственным ереходом по ссылке загрузки датасета
                else:
                    # найдём элемент, в котором содержится относительная ссылка
                    href_element = driver.find_element(By.XPATH, '//div[@class="sc-emfenM sc-fnpAPw cvuSKw gzjyQr"]/a')
                    # извлечём относительную ссылку
                    rel_link = href_element.get_attribute('href')
                    # составим абсолютный путь на скачивание архива
                    ds_download_link = urljoin(main_url, rel_link)
                    # перейдём по прямой ссылке загрузки
                    driver.get(ds_download_link)

                # Можно использовать аналогичный код:
                # В этом варианте кода заменён time.sleep() на явные ожидания WebDriverWait(), которые, возможно, являются более надежными.
                
                # try:
                #     download_button = WebDriverWait(driver, 10).until(
                #         EC.element_to_be_clickable((By.XPATH, '//button[contains(., "file_download")]'))
                #     )
                #     download_button.click()
                # except Exception as e:
                #     href_element = WebDriverWait(driver, 10).until(
                #         EC.presence_of_element_located((By.XPATH, '//div[@class="sc-emfenM sc-fnpAPw cvuSKw gzjyQr"]/a'))
                #     )
                #     rel_link = href_element.get_attribute('href')
                #     ds_download_link = urljoin(main_url, rel_link)
                #     driver.get(ds_download_link)

                # WebDriverWait(driver, 10).until(
                #     EC.presence_of_element_located((By.CLASS_NAME, "download-modal"))
                # )

                # Также, можно добавить контекстный менеджер with для инициализации и автоматического закрытия WebDriver после завершения работы функции.

        except Exception as e:
            print(f'Произошла ошибка в процессе поиска элемента на странице - {e}')
        # Подождём, пока загрузится файл:
        # Используем ожидание появления элемента с определенным классом, указывающим на загрузку
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "download-modal"))
        )
    except Exception as E:
        print(f'Произошла ошибка в процессе авторизации - {E}')
    # Закроем браузер в любом случае
    finally:
        driver.quit()


def unzip_and_replace_datasets(zip_path: str ="C:\\Users\\Allen\\Downloads\\archive.zip", 
                              extract_to: str = os.getcwd()) -> None:
    '''The function unzips the downloaded archive into the working directory'''

    # Проверка существования файла
    if not os.path.exists(zip_path):
        print(f"The file {zip_path} does not exist.")
        return

    # Разархивирование архива
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Archive extracted to {extract_to}")
    except zipfile.BadZipFile:
        print("The file is a bad zip file and cannot be extracted.")
    except Exception as e:
        print(f"An error occurred: {e}")
 

def transforming_datasets(test_path: str = "test.dat", train_path: str = "train.dat", 
                          test_csv_path: str = "ma_test.csv", train_csv_path: str = "ma_train.csv") -> pd.DataFrame:
    '''The function opens downloaded files, generates datasets adapted to processing based on them, and saves new datasets in .csv format'''

    # В архиве датасеты (тренировочный и тестовый) содержатся в формате .dat
    # поэтому, нам нужно их переформатировать в датасеты, пригодные и удобные для дальнейшего использования
    # Чтение файла .dat

    df_test = pd.read_fwf(test_path, sep='\t', header=None)
    df_train = pd.read_fwf(train_path, sep='\t', header=None)

    # Датафрейм df_test имеет атипичную ненормализованную структуру - всего 101 столбец, все аннотации содержатся в первом столбце, 
    # остальные колонки пустые, поэтому нам нужно создать датафрейм только из первой колонки.
    # Датафрейм df_train имеет схожую структуру - 101 столбец, первая колонка - классы заболеваний,
    # все аннотации содержатся во втором столбце, остальные колонки пустые.
    # Для нашей дальнейшей работы метки классов нам не требуются,
    # поэтому нам нужно создать датафрейм из второго столбца.

    # Трансформируем df_test в датасет формата .csv:
    # Выбор только первого столбца
    df_ma_test = df_test.iloc[:, [0]].rename(columns={0: 'abstracts'})

    # Запись данных первого столбца в файл .csv с заголовком
    df_ma_test.to_csv(test_csv_path, index=False, header=['abstracts'])

    # Теперь преобразуем df_train:
    # Выбор второго столбца
    df_ma_train = df_train.iloc[:, [1]].rename(columns={1: 'abstracts'})

    # Запись данных столбцов в файл .csv с заголовком
    df_ma_train.to_csv(train_csv_path, index=False, header=['abstracts'])


def prepare_dfs_to_labeling(path_to_ds_csv: str, manual_label_csv: str = 'manual_label_sample.csv', 
                        rule_based_csv: str = 'rule_based_sample.csv', train_size: float = 0.01):
    """
    Divides the dataframe into two parts for manual markup and for automatic rule-based markup.
    Returns dataframe for automatic rule-based markup.
    
    Parameters:
    df_train: pd.DataFrame: Датафрейм получаемый из функции 'transforming_datasets'.
    manual_label_csv (str): Путь к файлу CSV для ручной разметки.
    rule_based_csv (str): Путь к файлу CSV для разметки на основе правил.
    train_size (float): Доля датафрейма для ручной разметки.
    """
    
    df_train = pd.read_csv(path_to_ds_csv)

    # Разделение датафрейма на две части - для ручной разметки и для разметки на основе правил
    manual_label_sample, rule_based_sample = train_test_split(df_train, train_size=train_size, random_state=42)

    # Сохранение датафреймов в файлы .csv для дальнейшей обработки
    manual_label_sample.to_csv(manual_label_csv, index=False)
    rule_based_sample.to_csv(rule_based_csv, index=False)

    return rule_based_sample


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
    

def rule_based_labeling(df_rbs: pd.DataFrame):
    '''The function performs the markup of the dataframe - we add a column to the dataframe, 
    in which there will be labels based on a rule defined by us'''

    df_rbs['labeled_condition_mark'] = df_rbs['abstracts'].apply(rule_for_labeling)
    return df_rbs


# Ручную разметку выборки выдержек из медицинских статей в размере 144 шт (0,01 от всего датасета) я провел в Label Studio,
# с использованием маркировки цифровыми значениями. 
# Результат разметки сохранен в текущую директорию с именем ls_manual_labeled.csv.

# Объединение датасетов, если есть датасет, размеченный вручную, 
# и приведение их к виду который будет использоваться для обучения модели
def merging_labeled_dfs(df_rule: pd.DataFrame) -> pd.DataFrame:
    '''The function combines the date frames obtained as a result of automatic 
    rule-based markup and manual markup and brings the combined dataframe to the 
    form in which it will be used to train the model'''

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
    df_merged.to_csv('merged_dataset.csv', index=False)

    return df_merged


def teaching_and_saving_model(train_df: pd.DataFrame):
    '''The function trains a machine learning model on a marked-up dataset, saves the model and 
    a vectorizer for further use, and returns a dataframe with the markup'''

    # Для начала, перемешаем датасет.
    train_df = shuffle(train_df)

    X = train_df['abstracts']
    Y = train_df['labeled_condition_mark']

    # Создаем векторизатор и преобразуем тексты в векторы
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    # Обучаем модель
    model = LogisticRegression(max_iter=15000)
    model.fit(X_vectorized, Y)

    # Делаем предсказания на всех данных
    Y_predicted = model.predict(X_vectorized)

    # Добавление колонки с предсказанными значениями в датафрейм
    train_df['predicted_mark'] = Y_predicted

    # Сохранение датасета с предсказанными значениями
    train_df.to_csv('ma_train_with_predictions.csv', index=False)

    # Сохранение модели и векторизатора
    dump(model, 'model_ma_trained.joblib')
    dump(vectorizer, 'vectorizer_ma_trained.joblib')

    return train_df


def testing_model(path_to_ds_csv: str) -> pd.DataFrame:
    '''The function loads a trained machine learning model and applies it to an untagged dataframe'''

    # Загрузка модели
    model = load('model_ma_trained.joblib')
    # Загрузка векторизатора
    vectorizer = load('vectorizer_ma_trained.joblib')

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

    return df_test



def accuracy_scoring(df_for_evaluation: pd.DataFrame):
    '''The function evaluates the effectiveness of the machine learning model 
    and saves results into .txt files'''
    
    true_labels = df_for_evaluation['labeled_condition_mark']
    predicted_labels = df_for_evaluation['predicted_mark']
    
    # Вычисление точности
    accuracy = accuracy_score(true_labels, predicted_labels)
    # Вычисление других метрик
    report = classification_report(true_labels, predicted_labels)
    # Построение матрицы ошибок
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Получение текущей даты и времени
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # Сохранение точности с датой и временем
    with open('accuracy.txt', 'a') as f:
        f.write(f"Accuracy: {accuracy} (Дата и время: {formatted_time})\n")

    # Сохранение отчета о классификации с датой и временем
    with open('classification_report.txt', 'a') as f:
        f.write(f"Отчет о классификации (Дата и время: {formatted_time}):\n{report}\n")

    # Сохранение матрицы ошибок с датой и временем
    with open('confusion_matrix.txt', 'a') as f:
        f.write(f"\nМатрица ошибок (Дата и время: {formatted_time}):\n")
        for line in conf_matrix:
            f.write(' '.join(str(x) for x in line) + '\n')


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


def write_dataframe_to_mysql(table_name: str, df: pd.DataFrame, mysql_conn_id: str):
    # Получение параметров подключения из Airflow
    connection_params = BaseHook.get_connection(mysql_conn_id)
    conn_str = f"mysql+mysqldb://{connection_params.login}:{connection_params.password}" \
               f"@{connection_params.host}/{connection_params.schema}"
    engine = create_engine(conn_str)

    # Запись датафрейма в базу данных MySQL
    df.to_sql(name=table_name, con=engine, if_exists='replace', index=False)

    print(f"Dataframe is written to MySQL table {table_name} successfully.")



if __name__ == "__main__":
     
# Импорт библиотек

    from kaggle.api.kaggle_api_extended import KaggleApi
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
    from sqlalchemy import create_engine
    
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

    create_database('airflow_db', 'DE_DP_text_classification')

    write_dataframe_to_mysql(df_train_with_predictions)
    write_dataframe_to_mysql(df_test_with_predictions)

