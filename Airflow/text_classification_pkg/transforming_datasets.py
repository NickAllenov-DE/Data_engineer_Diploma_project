
# Преобразование датасетов 
def transforming_datasets(train_path: str = "train.dat", test_path: str = "test.dat",
                          test_csv_path: str = "ma_test.csv", train_csv_path: str = "ma_train.csv") -> tuple[str, str]:
    '''The function opens downloaded files, generates datasets adapted to processing based on them, and saves new datasets in .csv format'''

    # В архиве датасеты (тренировочный и тестовый) содержатся в формате .dat
    # поэтому, нам нужно их переформатировать в датасеты, пригодные и удобные для дальнейшего использования
    # Чтение файла .dat

    df_train = pd.read_fwf(train_path, sep='\t', header=None)
    df_test = pd.read_fwf(test_path, sep='\t', header=None)
    
    # Датафрейм df_test имеет атипичную ненормализованную структуру - всего 101 столбец, все аннотации содержатся в первом столбце, 
    # остальные колонки пустые, поэтому нам нужно создать датафрейм только из первой колонки.
    # Датафрейм df_train имеет схожую структуру - 101 столбец, первая колонка - классы заболеваний,
    # все аннотации содержатся во втором столбце, остальные колонки пустые.
    # Для нашей дальнейшей работы метки классов нам не требуются (потому что они неправильные),
    # поэтому нам нужно создать датафрейм из второго столбца.

    # Преобразуем df_train:
    # Выбор второго столбца
    df_ma_train = df_train.iloc[:, [1]].rename(columns={1: 'abstracts'})
    # Трансформируем df_test в датасет формата .csv:
    # Выбор только первого столбца
    df_ma_test = df_test.iloc[:, [0]].rename(columns={0: 'abstracts'})
    # Запись датафреймов в файл .csv с заголовком
    df_ma_train.to_csv(train_csv_path, index=False, header=['abstracts'])
    df_ma_test.to_csv(test_csv_path, index=False, header=['abstracts'])    

    return train_csv_path, test_csv_path


def prepare_dfs_to_labeling(path_to_ds_csv: str, manual_label_csv: str = 'manual_label_sample.csv', 
                        rule_based_csv: str = 'rule_based_sample.csv', train_size: float = 0.01) -> str:
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

    return rule_based_csv



if __name__ == "__main__":
    test_dat_path = "test.dat"
    train_dat_path = "train.dat"
    test_csv_path = "ma_test.csv"
    train_csv_path = "ma_train.csv"
    transforming_datasets(test_dat_path, train_dat_path, test_csv_path, train_csv_path)
    