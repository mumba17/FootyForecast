import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random

# Load the dataset into a pandas DataFrame
filename = 'dataset_footyf_v1.csv'  # Replace with the actual filename
df = pd.read_csv(filename)

class FootballDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out

def test_model():
    Total_correct = 0
    for i in range(101):
        random_index = random.randint(0, len(df) - 1)
        random_line = df.iloc[random_index]

        # Remove the last value (label) from the random line
        input_data = random_line.iloc[:-1].values

        # Print the random line and the label
        label = random_line.iloc[-1]
        print("Label:", int(label))

        # Reshape the input data to match the model's input shape
        input_data = input_data.reshape(1, -1)

        # Load the trained model
        input_size = input_data.shape[1]
        num_classes = 3  # Replace with the actual number of classes
        model = LogisticRegressionModel(input_size, num_classes)

        # Load the trained model weights
        model.load_state_dict(torch.load('FootyForecast.pt'))

        # Set the model to evaluation mode
        model.eval()

        # Convert input data to torch tensor
        input_tensor = torch.from_numpy(input_data).float()

        print("\n")
        # Make predictions using the model
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = nn.functional.softmax(output, dim=1)
            _, predicted_class = torch.max(probabilities, 1)

        predicted_label = predicted_class.item()
        print("Probability:", probabilities)
        print("Predictions:", predicted_label)
        if int(label) == predicted_label:
            Total_correct += 1

    print(f'Total correct out of {i}: {Total_correct}')

test_model()
