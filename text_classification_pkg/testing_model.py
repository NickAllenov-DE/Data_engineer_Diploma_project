def testing_model(path_to_csv: str = "D:\\GeekBrains\\Data_engineer_Diploma_project\\ma_test.csv"):
    '''The function loads a trained machine learning model and applies it to a conditionally untagged dataframe'''

    # Импорт библиотек
    import pandas as pd
    from joblib import load
    from sklearn.utils import shuffle
    from text_classification_pkg import rule_based_labeling

    # Загрузка модели
    model = load('model_ma_trained.joblib')
    # Загрузка векторизатора
    vectorizer = load('vectorizer_ma_trained.joblib')

    # Загрузка тестового датасета
    test_df = pd.read_csv(path_to_csv)
    # разметка тестового датасета на основе правил
    test_df = rule_based_labeling(test_df)

    # Применим векторизатор к новым текстовым данным
    new_texts_vectorized = vectorizer.transform(test_df['abstracts'])
    # Используем модель для предсказания меток новых данных
    new_predictions = model.predict(new_texts_vectorized)
    # Добавление колонки с предсказанными значениями в датафрейм
    test_df['predicted_mark'] = new_predictions
   
    # Сохранение датасета с размеченными и предсказанными значениями
    test_df.to_csv('ma_test_with_predictions.csv', index=False)

    return test_df