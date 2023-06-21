from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os
from termcolor import colored
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

os.system('color')

@ignore_warnings(category=ConvergenceWarning)
def compare_models(model1, model2, X_test, y_test):
    """
    Compare two classification models using precision, recall, F1 score and accuracy metrics.

    Args:
    model1 (object): trained classification model
    model2 (object): trained classification model
    X_test (pandas DataFrame): input data to test the model on
    y_test (pandas Series): true labels of the test data

    Returns:
    True if F1 score of model1 is better
    False if F1 score of model2 is better
    """
    # Predict labels using both models
    # If either model fails, it will continue with a 'working' model.
    try:
        y_pred1 = model1.predict(X_test)
    except ValueError:
        return False
    try:
        y_pred2 = model2.predict(X_test)
    except ValueError:
        return True

    # Calculate evaluation metrics for the first model
    precision1 = precision_score(y_test, y_pred1, average='macro', zero_division=1)
    recall1 = recall_score(y_test, y_pred1, average='macro')
    f1_score1 = f1_score(y_test, y_pred1, average='macro')
    accuracy1 = accuracy_score(y_test, y_pred1)

    # Calculate evaluation metrics for the second model
    precision2 = precision_score(y_test, y_pred2, average='macro', zero_division=1)
    recall2 = recall_score(y_test, y_pred2, average='macro')
    f1_score2 = f1_score(y_test, y_pred2, average='macro')
    accuracy2 = accuracy_score(y_test, y_pred2)

    # Print the evaluation metrics for both models
    print("")
    print("Generated model evaluation metrics:")
    print(f"Precision: {precision1:.5f}")
    print(f"Recall: {recall1:.5f}")
    print(f"F1-score: {f1_score1:.5f}")
    print(f"Accuracy: {accuracy1:.5f}")
    print("")
    print("Saved model evaluation metrics:")
    print(f"Precision: {precision2:.5f}")
    print(f"Recall: {recall2:.5f}")
    print(f"F1-score: {f1_score2:.5f}")
    print(f"Accuracy: {accuracy2:.5f}")
    
    if f1_score1 > f1_score2:
        print(colored("Replacing the model!", "green"))
        return True
    else:
        print(colored("Not replacing the model.", "red"))
        return False