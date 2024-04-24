# Переходим к обучению модели.
# На объединенном размеченном датасете:
def teaching_and_saving_model(train_df: pd.DataFrame):

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

    # Сохранение модели
    dump(model, 'model_ma_trained.joblib')
    # Сохранение векторизатора
    dump(vectorizer, 'vectorizer_ma_trained.joblib')

    return Y_test, Y_test_predicted
