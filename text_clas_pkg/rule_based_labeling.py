# Проводим разметку датасета на основе правила
def rule_based_labeling(df_rbs: pd.DataFrame) -> pd.DataFrame:
    '''The function performs the markup of the dataframe - we add a column to the dataframe, 
    in which there will be labels based on a rule defined by us'''

    # Импорт библиотеки и модуля
    import pandas as pd
    from text_clas_pkg import rule_for_labeling

    # Разметка датафрейма - добавляем в датафрейм колонку, в которой 
    # будут метки на онове определенного нами правила
    df_rbs['labeled_condition_mark'] = df_rbs['abstracts'].apply(rule_for_labeling)

    return df_rbs