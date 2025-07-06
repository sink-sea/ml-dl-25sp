import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import *
# from utils.metrics import accuracy, average_loss
from utils.args import get_args
from model.net import *
import matplotlib.pyplot as plt

def train(model, train_loader, criterion, optimizer, device, epochs, log_interval=10, plot_accuracy=False):
    """
    Train the model
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): DataLoader for the training dataset
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters
        device (str): Device to use for training ('cuda' or 'cpu')
        epochs (int): Number of epochs to train the model
        log_interval (int): Interval for logging training progress
        plot_accuracy (bool): Whether to plot accuracy during training
    Returns:
        float: Average loss for the epoch
    """
    model.to(device)
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    if plot_accuracy:
        accuracies = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        avg_loss = running_loss / len(train_loader)
        total_loss += avg_loss
        accuracy_value = 100. * correct / total

        if plot_accuracy:
            accuracies.append(accuracy_value)
        if (epoch + 1) % log_interval == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy_value:.2f}%')
    if plot_accuracy:
        plt.plot(range(1, epochs + 1), accuracies, label='Training Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Training Accuracy over Epochs')
        plt.grid()
        plt.xticks(range(1, epochs + 1, epochs // 10))
        plt.legend()
        plt.show()


def test(model, test_loader, criterion, device):
    """
    Test the model
    Args:
        model (nn.Module): The model to test
        test_loader (DataLoader): DataLoader for the test dataset
        criterion (nn.Module): Loss function
        device (str): Device to use for testing ('cuda' or 'cpu')
    Returns:
        float: Average loss for the test dataset
        float: Accuracy of the model on the test dataset
    """
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    avg_loss = total_loss / len(test_loader)
    accuracy_value = 100. * correct / total
    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy_value:.2f}%')
    return avg_loss, accuracy_value

def save_model(model, path):
    """
    Save the trained model to a file
    Args:
        model (nn.Module): The model to save
        path (str): Path to save the model
    """
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

def load_model(model, path, device):
    """
    Load a model from a file
    Args:
        model (nn.Module): The model to load the state into
        path (str): Path to the model file
        device (str): Device to load the model on ('cuda' or 'cpu')
    Returns:
        nn.Module: The model with loaded state
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f'Model loaded from {path}')
    return model
