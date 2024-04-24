# Оценим эффективность модели.
def accuracy_scoring(df_for_evaluation: pd.DataFrame):
    '''The function evaluates the effectiveness of the machine learning model 
    and saves results into .txt files'''
    
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from datetime import datetime

    true_labels = df_for_evaluation['labeled_condition_mark']
    predicted_labels = df_for_evaluation['predicted_mark']
    
    # Вычисление точности
    accuracy = accuracy_score(true_labels, predicted_labels)
    # Вычисление других метрик
    report = classification_report(true_labels, predicted_labels)
    # Построение матрицы ошибок
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # # Если необходимо отобразить результаты, раскоментировать:
    # print(accuracy)
    # print(report)
    # print(conf_matrix)

    # Получение текущей даты и времени
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # Сохранение точности с датой и временем
    with open('accuracy.txt', 'a') as f:
        f.write(f"Accuracy: {accuracy} (Дата и время: {formatted_time})\n")

    # Сохранение отчета о классификации с датой и временем
    with open('classification_report.txt', 'a') as f:
        f.write(f"Отчет о классификации (Дата и время: {formatted_time}):\n{report}\n")

    # Сохранение матрицы ошибок с датой и временем
    with open('confusion_matrix.txt', 'a') as f:
        f.write(f"Матрица ошибок (Дата и время: {formatted_time}):\n")
        for line in conf_matrix:
            f.write(' '.join(str(x) for x in line) + '\n')