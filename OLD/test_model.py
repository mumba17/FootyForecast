import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from compare_models import compare_models
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import random

model = joblib.load('FootyForecast_NEW.joblib')

# Load the dataset into a pandas DataFrame
filename = 'dataset_footyf_v1.csv'  # Replace with the actual filename
dataset = pd.read_csv(filename)

@ignore_warnings(category=UserWarning)
def test_model():
    Total_correct = 0
    for i in range(101):
        random_index = random.randint(918, len(dataset) - 1)
        random_line = dataset.iloc[random_index]

        # Remove the last value (label) from the random line
        input_data = random_line.iloc[:-1].values

        # Print the random line and the label
        label = random_line.iloc[-1]
        print("Label:", int(label))

        # Reshape the input data to match the model's input shape
        input_data = input_data.reshape(1, -1)
        
        # Make predictions using the model
        predictions = model.predict(input_data)
        print("Predictions:", int(predictions))
        
        if int(label) == int(predictions):
            Total_correct += 1

    print(f'Total correct out of {i}: {Total_correct}')

test_model()