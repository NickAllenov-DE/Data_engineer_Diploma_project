
# Проводим разметку датасета на основе правила
def rule_based_labeling(df_rbs_path: str, df_rbs_csv: str = 'df_rule_labeled.csv') -> str:
    '''The function performs the markup of the dataframe - we add a column to the dataframe, 
    in which there will be labels based on a rule defined by us'''

    df_rbs = pd.read_csv(df_rbs_path)
    df_rbs['labeled_condition_mark'] = df_rbs['abstracts'].apply(rule_for_labeling)
    df_rbs.to_csv(df_rbs_csv, index=False)

    return df_rbs_csv