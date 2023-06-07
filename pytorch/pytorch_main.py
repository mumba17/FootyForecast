import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from pytorch_compare import compare_models
import torch.cuda

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

df = pd.read_csv('dataset_footyf_v1.csv', on_bad_lines='skip')
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")

@ignore_warnings(category=ConvergenceWarning)
def generate_model(compare=True):
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2)
    X_train = X_train.dropna().values
    y_train = y_train.dropna().values
    X_test = X_test.dropna().values  # Convert X_test to a numpy array
    y_test = y_test.dropna().values  # Convert y_test to a numpy array
    
    cuda_id = torch.cuda.current_device()

    # Define the logistic regression model
    input_size = X_train.shape[1]
    num_classes = len(pd.unique(y_train))
    model = LogisticRegressionModel(input_size, num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.002)

    # Prepare the training data
    train_dataset = FootballDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Train the model
    num_epochs = 20
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    if compare:
        try:
            loaded_model_state_dict = torch.load('FootyForecast.pt', map_location=torch.device('cuda'))
            loaded_model = LogisticRegressionModel(input_size, num_classes)
            loaded_model.load_state_dict(loaded_model_state_dict)
            if compare_models(model, loaded_model, X_test, y_test):
                torch.save(model.state_dict(), 'FootyForecast.pt')
        except FileNotFoundError:
            torch.save(model.state_dict(), 'FootyForecast.pt')
    else:
        torch.save(model.state_dict(), 'FootyForecast.pt')

while True:
    generate_model()
