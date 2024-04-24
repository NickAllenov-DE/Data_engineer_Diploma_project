# Проводим разметку датасета на основе правила
def rule_based_labeling(df_rbs: pd.DataFrame) -> pd.DataFrame:

    # Импорт библиотеки и модуля
    import pandas as pd
    from text_clas_pkg import rule_for_labeling

    # Разметка датафрейма - добавляем в датафрейм колонку, в которой 
    # будут метки на онове определенного нами правила
    df_rbs['labeled_condition_mark'] = df_rbs['abstracts'].apply(rule_for_labeling)

    return df_rbs