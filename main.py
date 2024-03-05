import loader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import loader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve

# LSTM neural network
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.7):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)  # feed forward layer
        self.sigmoid = nn.Sigmoid()  # To normalize between 0 and 1
        self.batch_norm = nn.BatchNorm1d(hidden_size)  # normalization

    def forward(self, x):
        lstm_out, _ = self.lstm(x.view(len(x), 1, -1))
        lstm_out = lstm_out.view(len(x), -1)
        lstm_out_normalized = self.batch_norm(lstm_out)
        output = self.fc(lstm_out_normalized)  # normalization
        output = self.sigmoid(output)  # put the values between zero and one.

        return output


# Training loop
def train(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data  # Assuming inputs and labels are in the loader
            labels = labels.view(-1, 1)
            optimizer.zero_grad()
            outputs = model(inputs.float())  # inference

            loss = criterion(outputs, labels.float())  # calculate loss
            loss.backward()  # Backpropagate values
            optimizer.step()  # Updates weights and biases

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
    threshold_labels = []

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            labels = labels.view(-1, 1)  # reshapes values
            outputs = model(inputs.float())
            predicted = (outputs > threshold).float()  # applies the threshold

            total += labels.size(0)
            correct += (predicted == labels.float()).sum().item()

            # values to be graphed to find the optimal threshold
            predicted_labels.extend(outputs.squeeze().tolist())
            # predictions with predefined threshold
            threshold_labels.extend(predicted.squeeze().tolist())
            true_labels.extend(labels.squeeze().tolist())

    precision, recall, thresholds = precision_recall_curve(
        true_labels, predicted_labels)
    # Adding small epsilon to avoid division by zero
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)


    # precision recall curve
    plt.figure(figsize=(8, 6))
    plt.scatter(thresholds, precision[:-1], label='Precision')
    plt.scatter(thresholds, recall[:-1], label='Recall')
    plt.scatter(thresholds, f1_scores[:-1],
                color='green', lw=2, label='F1-score')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    accuracy = correct / total
    print(f"Accuracy on test data: {accuracy * 100:.2f}%")

    # Show the confusion matrix
    confusion_mat = confusion_matrix(true_labels, threshold_labels)
    print("Confusion Matrix:")
    print(confusion_mat)

    # Find the maximum of the F1 score parabola
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f"Optimal Threshold: {best_threshold}")

    return accuracy


input_size = 22
# I am using the heuristic of the hidden layer being twice theinput size + 1
hidden_size = 46 
output_size = 1
dropout = 0.80
threshold = 0.335

data = pd.read_csv("synthetic_data.csv")
train_loader, test_loader = loader.split_data(data)


def main():

    model = LSTMNet(input_size, hidden_size, output_size, dropout=dropout)
    criterion = nn.BCELoss()  # binary cross entropy loss
    # Adam optimizer with learning rate 0.0005
    # This was found through hyperparameter training
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    train(model, train_loader, criterion, optimizer, epochs=10)
    test(model, test_loader)


main()
