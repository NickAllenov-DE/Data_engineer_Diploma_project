# Преобразование датасетов 
def transforming_datasets(test_path: str = "D:\\GeekBrains\\Data_engineer_Diploma_project\\test.dat", 
                          train_path: str = "D:\\GeekBrains\\Data_engineer_Diploma_project\\train.dat", 
                          test_csv_path: str = "D:\\GeekBrains\\Data_engineer_Diploma_project\\ma_test.csv", 
                          train_csv_path: str = "D:\\GeekBrains\\Data_engineer_Diploma_project\\ma_train.csv") -> None:
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
    # все аннотации содержатся во втором столбце, остальные колонки пустые, 
    # поэтому нам нужно создать датафрейм из первых двух столбцов.

    # Трансформируем df_test в датасет формата .csv:
    # Выбор только первого столбца
    first = df_test.iloc[:, 0]

    # Запись данных первого столбца в файл .csv с заголовком
    first.to_csv(test_csv_path, index=False, header=['abstracts'])

    # Теперь преобразуем df_train:
    # Выбор первых двух сколонок
    first_two = df_train.iloc[:, :2]

    # Запись данных столбцов в файл .csv с заголовками
    first_two.to_csv(train_csv_path, index=False, header=['labels', 'abstracts'])


if __name__ == "__main__":
    test_dat_path = "D:\\GeekBrains\\Data_engineer_Diploma_project\\test.dat"
    train_dat_path = "D:\\GeekBrains\\Data_engineer_Diploma_project\\train.dat"
    test_csv_path = "D:\\GeekBrains\\Data_engineer_Diploma_project\\ma_test.csv"
    train_csv_path = "D:\\GeekBrains\\Data_engineer_Diploma_project\\ma_train.csv"
    transforming_datasets(test_dat_path, train_dat_path, test_csv_path, train_csv_path)