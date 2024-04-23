def prepare_to_labeling(input_csv: str = 'ma_train.csv', manual_label_csv: str = 'manual_label_sample.csv', 
                        rule_based_csv: str = 'rule_based_sample.csv', train_size: float = 0.01):
    """
    Divides the dataframe into two parts for manual markup and for automatic rule-based markup.
    
    Parameters:
    input_csv (str): Путь к исходному файлу CSV.
    manual_label_csv (str): Путь к файлу CSV для ручной разметки.
    rule_based_csv (str): Путь к файлу CSV для разметки на основе правил.
    train_size (float): Доля датафрейма для ручной разметки.
    """
    
    # Импорт библиотек
    import pandas as pd
    from sklearn.model_selection import train_test_split  

    # Создание датафрейма Pandas из файла .csv
    df_train = pd.read_csv(input_csv, engine='python', encoding='utf-8', on_bad_lines='skip', encoding_errors='ignore')
    # Разделение датафрейма на две части - для ручной разметки и для разметки на основе правил
    manual_label_sample, rule_based_sample = train_test_split(df_train, train_size=train_size, random_state=42)

    # Сохранение датафреймов в файлы .csv для дальнейшей обработки
    manual_label_sample.to_csv(manual_label_csv, index=False)
    rule_based_sample.to_csv(rule_based_csv, index=False)