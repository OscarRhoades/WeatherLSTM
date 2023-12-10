import preprocessing
import loader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix

# Simple feed forward neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size - 4)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size -4, hidden_size - 8)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size - 8, output_size)
        self.sigmoid = nn.Sigmoid()  # To normalize between 0 and 1

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x










# Training loop
def train(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data  # Assuming inputs and labels are in the loader
            labels = labels.view(-1, 1) # data from loader needs to be reshaped
            optimizer.zero_grad()
            outputs = model(inputs.float())  # inference
            loss = criterion(outputs, labels.float())  # calculate loss
            loss.backward()  # Backpropagate values
            optimizer.step()  # Update weights

            
            running_loss += loss.item()
            if i % 10 == 9:  
                print(f"Epoch [{epoch + 1}/{epochs}], "
                      f"Step [{i + 1}/{len(train_loader)}], "
                      f"Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

# Testing Loop


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []

   
    
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            labels = labels.view(-1, 1) 
            outputs = model(inputs.float())
            predicted = (outputs > 0.5).float()  
            
            total += labels.size(0)
            correct += (predicted == labels.float()).sum().item()

            predicted_labels.extend(predicted.squeeze().tolist())
            true_labels.extend(labels.squeeze().tolist())
    
    accuracy = correct / total
    print(f"Accuracy on test data: {accuracy * 100:.2f}%")
    

    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(confusion_mat)


    return accuracy



input_size = 22
hidden_size = 16
output_size = 1
num_samples = 100

preprocessed_data = preprocessing.preprocess_synthetic('weatherAUS.csv')
train_loader, test_loader = loader.split_data(preprocessed_data) 



def main():

    # Initalized model, criterion and optimizer
    model = SimpleNN(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Adam optimizer with learning rate 0.001

    
    train(model, train_loader, criterion, optimizer, epochs=10)

   
    test(model, test_loader)


main()

