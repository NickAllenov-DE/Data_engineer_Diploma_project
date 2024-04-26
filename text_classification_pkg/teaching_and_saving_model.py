# Переходим к обучению модели.
# На объединенном размеченном датасете:

def teaching_and_saving_model(train_df: pd.DataFrame):
    '''The function trains a machine learning model on a marked-up dataset, saves the model and 
    a vectorizer for further use, and returns a dataframe with the markup'''

    # Импорт библиотек
    from sklearn.model_selection import train_test_split                # разделение данных на обучающую и тестовую части
    from sklearn.feature_extraction.text import TfidfVectorizer         # преобразование текста в вектор
    from sklearn.linear_model import LogisticRegression                 # использование модели логистической регрессии
    from sklearn.utils import shuffle
    from joblib import dump

    # Для начала, ещё раз перемешаем датасет.
    train_df = shuffle(train_df)

    X = train_df['abstracts']
    Y = train_df['labeled_condition_mark']

    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X_vectorized, Y, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=15000)
    model.fit(X_train, Y_train)

    Y_test_predicted = model.predict(X_test)
    # Добавление колонки с предсказанными значениями в датафрейм
    train_df['predicted_mark'] = Y_test_predicted
    # Сохранение датасета с предсказанными значениями
    train_df.to_csv('ma_train_with_predictions.csv', index=False)

    # Сохранение модели
    dump(model, 'model_ma_trained.joblib')
    # Сохранение векторизатора
    dump(vectorizer, 'vectorizer_ma_trained.joblib')

    return train_df