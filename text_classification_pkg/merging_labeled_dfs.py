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