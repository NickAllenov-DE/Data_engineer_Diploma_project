# Проводим разметку датасета на основе правила
def rule_based_labeling() -> pd.DataFrame:
    '''The function performs layout of the dataset based on the rule and returns the marked-up dataframe'''

    # Импорт библиотеки
    import pandas as pd
    from text_clas_pkg import rule_for_labeling

    # Создание датафрейма Pandas
    df_rule = pd.read_csv('rule_based_sample.csv')

    # Разметка датафрейма - добавляем в датафрейм колонку в которой 
    # будут метки на онове определенного нами правила
    df_rule['labeled condition name'] = df_rule['medical_abstract'].apply(rule_for_labeling)

    return df_rule