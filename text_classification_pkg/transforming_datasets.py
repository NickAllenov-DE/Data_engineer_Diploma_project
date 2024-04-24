# Преобразование датасетов 
def transforming_datasets(test_path: str = "D:\\GeekBrains\\Data_engineer_Diploma_project\\test.dat", 
                          train_path: str = "D:\\GeekBrains\\Data_engineer_Diploma_project\\train.dat", 
                          test_csv_path: str = "D:\\GeekBrains\\Data_engineer_Diploma_project\\ma_test.csv", 
                          train_csv_path: str = "D:\\GeekBrains\\Data_engineer_Diploma_project\\ma_train.csv") -> (pd.DataFrame, pd.DataFrame):
    
    '''The function opens downloaded files, generates datasets adapted to processing based on them, and saves new datasets in .csv format'''

    # Импорт библиотек
    import pandas as pd

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
    df_ma_test = df_test.iloc[:, 0]

    # Запись данных первого столбца в файл .csv с заголовком
    df_ma_test.to_csv(test_csv_path, index=False, header=['abstracts'])

    # Теперь преобразуем df_train:
    # Выбор второй колонки
    df_ma_train = df_train.iloc[:, 1]

    # Запись данных столбца в файл .csv с заголовками
    df_ma_train.to_csv(train_csv_path, index=False, header=['abstracts'])

    return df_ma_train, df_ma_test


if __name__ == "__main__":
    test_dat_path = "D:\\GeekBrains\\Data_engineer_Diploma_project\\test.dat"
    train_dat_path = "D:\\GeekBrains\\Data_engineer_Diploma_project\\train.dat"
    test_csv_path = "D:\\GeekBrains\\Data_engineer_Diploma_project\\ma_test.csv"
    train_csv_path = "D:\\GeekBrains\\Data_engineer_Diploma_project\\ma_train.csv"
    transforming_datasets(test_dat_path, train_dat_path, test_csv_path, train_csv_path)
    