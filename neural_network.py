import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Define Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[512, 256], output_size=10, dropout_rate=0.2):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x


# Load MNIST Dataset
def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# Train Function
def train_model(model, train_loader, criterion, optimizer, device, epochs=5):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")


# Evaluate Function
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


# Main
if __name__ == "__main__":
    # Device config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model & Hyperparameters
    model = NeuralNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load data
    train_loader, test_loader = get_data_loaders()

    # Train and evaluate
    train_model(model, train_loader, criterion, optimizer, device)
    evaluate_model(model, test_loader, device)

    # Save model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth")
