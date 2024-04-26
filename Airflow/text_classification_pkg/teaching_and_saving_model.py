# Переходим к обучению модели.
# На объединенном размеченном датасете:

def teaching_and_saving_model(train_df_path: str, trained_df_csv: str = 'ma_train_with_predictions.csv') -> str:
    '''The function trains a machine learning model on a marked-up dataset, saves the model and 
    a vectorizer for further use, and returns a dataframe with the markup'''

    train_df = pd.read_csv(train_df_path)

    # Для начала, перемешаем датасет.
    train_df = shuffle(train_df)

    X = train_df['abstracts']
    Y = train_df['labeled_condition_mark']

    # Создаем векторизатор и преобразуем тексты в векторы
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    # Обучаем модель
    model = LogisticRegression(max_iter=15000)
    model.fit(X_vectorized, Y)

    # Делаем предсказания на всех данных
    Y_predicted = model.predict(X_vectorized)

    # Добавление колонки с предсказанными значениями в датафрейм
    train_df['predicted_mark'] = Y_predicted

    # Сохранение датасета с предсказанными значениями
    train_df.to_csv(trained_df_csv, index=False)

    # Сохранение модели и векторизатора
    dump(model, 'model_ma_trained.joblib')
    dump(vectorizer, 'vectorizer_ma_trained.joblib')

    return trained_df_csv