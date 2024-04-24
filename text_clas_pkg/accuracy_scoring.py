# Оценим эффективность моделей.
def accuracy_scoring(Y_test, Y_test_predicted):
    
    # оценка производительности модели
    from sklearn.metrics import accuracy_score  

    accuracy = accuracy_score(Y_test, Y_test_predicted)
    return accuracy