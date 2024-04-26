
def testing_model(path_to_ds_csv: str) -> pd.DataFrame:
    '''The function loads a trained machine learning model and applies it to an untagged dataframe'''

    # Загрузка модели
    model = load('model_ma_trained.joblib')
    # Загрузка векторизатора
    vectorizer = load('vectorizer_ma_trained.joblib')

    df_test = pd.read_csv(path_to_ds_csv)

    # Преобразование текстовых данных нового датафрейма в векторный формат
    X_new = vectorizer.transform(df_test['abstracts'])
    # Используем модель для предсказания меток новых данных
    Y_new_predicted = model.predict(X_new)

    # разметка тестового датасета на основе правил для послдующей
    # оценки эффективности модели
    df_test = rule_based_labeling(df_test)

    # Добавление колонки с предсказанными значениями в датафрейм
    df_test['predicted_mark'] = Y_new_predicted
   
    # Сохранение датасета с размеченными и предсказанными значениями
    df_test.to_csv('ma_test_with_predictions.csv', index=False)

    return df_test

