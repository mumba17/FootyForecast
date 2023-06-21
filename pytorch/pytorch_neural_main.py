import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from pytorch_compare import compare_models
import torch.nn.functional as F
import torch.cuda
import random
import numpy as np

class FootballDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

class NeuralNetworkModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetworkModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 56)
        self.fc2 = nn.Linear(56, 188)
        self.fc3 = nn.Linear(188, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out

df = pd.read_csv('footy_dataset.csv', on_bad_lines='skip')
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("Running on CPU")

def generate_model(compare=True, stopped=False):
    global model
    data = df.dropna()
    
    #random_lr = random.uniform(0.001,0.01)
    random_lr = 0.01
    
    print(f'Setting starting learning rate to: {random_lr}')

# Split the non-null data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data.iloc[:, :-1].values,
        data.iloc[:, -1].values,
        test_size=0.25
    )
    
    y_test = np.array(y_test)

    # Define the neural network model
    input_size = X_train.shape[1]
    num_classes = len(pd.unique(y_train))

    model = NeuralNetworkModel(input_size, num_classes)
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=random_lr)

    # Define the early stopping parameters
    best_loss = float('inf')
    early_stop_counter = 0

    # Prepare the training data
    train_dataset = FootballDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=802, shuffle=True)
    
    # Prepare the validation data
    val_dataset = FootballDataset(X_test, y_test)
    val_loader = DataLoader(val_dataset, batch_size=802, shuffle=False)

    # Define the learning rate scheduler
    patience = 5
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience)

    # Train the model
    num_epochs = 250
    for epoch in range(num_epochs):
        # Train for one epoch
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Calculate the validation loss
        with torch.no_grad():
            model.eval()
            val_loss = 0
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()
            val_loss /= len(val_loader)

        # Update the learning rate based on the validation loss
        scheduler.step(val_loss)

        # Check if validation loss has improved
        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_counter = 0
            # Save the model checkpoint if desired
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience and not stopped:
                stopped = True
                print(f"Early stopping at epoch {epoch}")

    X_test_tensor = torch.from_numpy(X_test).float().to(device)

    if compare:
        try:
            loaded_model_state_dict = torch.load('FootyForecast_neural.pt', map_location=device)
            loaded_model = NeuralNetworkModel(input_size, num_classes)
            loaded_model.load_state_dict(loaded_model_state_dict)
            loaded_model = loaded_model.to(device)
            
            if compare_models(model, loaded_model, X_test_tensor, y_test):
                torch.save(model.state_dict(), 'FootyForecast_neural.pt')
        except FileNotFoundError:
            torch.save(model.state_dict(), 'FootyForecast_neural.pt')
    else:
        torch.save(model.state_dict(), 'FootyForecast_neural.pt')

First = True

def test_model(First):
    Total_correct = 0
    for i in range(101):
        random_index = random.randint(0, len(df) - 1)
        random_line = df.iloc[random_index]

        # Remove the last value (label) from the random line
        input_data = random_line.iloc[:-1].values

        # Print the random line and the label
        label = random_line.iloc[-1]
        # print("Label:", int(label))

        # Reshape the input data to match the model's input shape
        input_data = input_data.reshape(1, -1)

        # Load the trained model weights
        model.load_state_dict(torch.load('FootyForecast_neural.pt'))

        # Set the model to evaluation mode
        model.eval()

        # Convert input data to torch tensor
        input_tensor = torch.from_numpy(input_data).float()
        device = torch.device("cuda:0")
        input_tensor = input_tensor.to(device)
        # print("")
        # Make predictions using the model
        with torch.no_grad():
            output = model(input_tensor)
            model.to(device)
            probabilities = nn.functional.softmax(output, dim=1)
            _, predicted_class = torch.max(probabilities, 1)

        predicted_label = predicted_class.item()
        # print("Probability:", probabilities)
        # print("Predictions:", predicted_label)
        if int(label) == predicted_label:
            Total_correct += 1

    print(f'Total correct out of {i}: {Total_correct}')
    highscore = 0
    if Total_correct > 60 and First:
        First = False
        for _ in range(20):
            highscore += test_model(First)
            
        with open('highscore.txt', 'r') as file:
            previous_highscore = int(file.read())
            
        if highscore > previous_highscore:
            torch.save(model.state_dict(), 'FootyForecast_neural_highscore.pt')
            with open('highscore.txt', 'w') as file:
                file.write(str(highscore))
                print("REPLACED HIGHSCORE MODEL!")
            
    return Total_correct

while True:
    stopped = False
    generate_model()
    test_model(First)
