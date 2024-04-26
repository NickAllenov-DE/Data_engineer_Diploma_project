
# Готовим датафреймы к разметке
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