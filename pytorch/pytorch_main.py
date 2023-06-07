import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

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

df = pd.read_csv('dataset_eredivisie_v4.csv', on_bad_lines='skip')

@ignore_warnings(category=ConvergenceWarning)
def generate_model(compare=True):
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2)
    X_train = X_train.dropna().values
    y_train = y_train.dropna().values

    # Define the logistic regression model
    input_size = X_train.shape[1]
    num_classes = len(pd.unique(y_train))
    model = LogisticRegressionModel(input_size, num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Prepare the training data
    train_dataset = FootballDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    if compare:
        loaded_model = torch.load('FootyForecast_NEW.pt')
        if compare_models(model, loaded_model, X_test, y_test):
            torch.save(model, 'FootyForecast_NEW.pt')

while True:
    generate_model()
