import torch
import torch.nn.functional as F
import os
from termcolor import colored
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

os.system('color')

@ignore_warnings(category=ConvergenceWarning)
def compare_models(model1, model2, X_test, y_test):
    """
    Compare two classification models using precision, recall, F1 score, and accuracy metrics.

    Args:
    model1 (nn.Module): trained classification model
    model2 (nn.Module): trained classification model
    X_test (numpy.ndarray): input data to test the model on
    y_test (numpy.ndarray): true labels of the test data

    Returns:
    True if F1 score of model1 is better
    False if F1 score of model2 is better
    """
    model1.eval()
    model2.eval()

    # Predict labels using both models
    with torch.no_grad():
        y_pred1 = model1(X_test)
        y_pred1 = torch.argmax(F.softmax(y_pred1, dim=1), dim=1)
        y_pred2 = model2(X_test)
        y_pred2 = torch.argmax(F.softmax(y_pred2, dim=1), dim=1)

    # Convert tensors to numpy arrays
    y_pred1 = y_pred1.cpu().numpy()
    y_pred2 = y_pred2.cpu().numpy()

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



