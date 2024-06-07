import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from scipy.special import softmax
from thefuzz import fuzz
import dataset_embedding
import pandas as pd



dataset = pd.read_pickle('transformed_dataset.pkl')
top_k = 10
split_index = int(len(dataset) * 0.8)

##### WEIGHTED COSINE DISTANCE MODEL #####
# class WeightedCosineDistance():
#     def __init__(self, num_features: int=384):

# # Loss function
# def weighted_cosine_distance(u: np.ndarray, v: np.ndarray, W: np.ndarray) -> float:
#     return 1 - (np.exp(W) @ u) @ (np.exp(W) @ v) / (np.linalg.norm(np.exp(W) @ u) * np.linalg.norm(np.exp(W) @ v))

# # Initialize W
# W = np.random.rand(384, 384)

# # Calculate the probabilities of each pair of embeddings
# def Probs(dataset: list, W: np.ndarray=W) -> np.ndarray:
#     distances = []
#     for i in range(len(dataset)):
#         distances.append(weighted_cosine_distance(dataset[i][0], dataset[i][1], W))
#     return softmax(distances)

# # Calculate the gradient of the loss function
# def gradient(dataset: list, weights: np.ndarray=W) -> np.ndarray:
#     probs = Probs(dataset, weights)
#     grad = np.zeros(384)
#     for i in range(len(dataset)):
#         grad += (probs[i] - 1) * (dataset[i][0] - dataset[i][1])
#     return grad

# # Update the weights using gradient descent (weights must be positive)
# def update_weights(dataset: list=dataset, weights: np.ndarray=W, lr: float=0.01) -> np.ndarray:
#     grad = gradient(dataset, weights)
#     weights -= lr * grad
#     return weights

# # Train the model
# def train_model(dataset: list=dataset, weights: np.ndarray=W, split_index: int=split_index, num_epochs: int=10, lr: float=0.01) -> np.ndarray:
#     for epoch in range(num_epochs):
#         weights = update_weights(dataset[:split_index], weights, lr)
#         print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {np.linalg.norm(gradient(dataset[:split_index], weights))}')
#         print(f'Epoch {epoch+1}/{num_epochs}, Test Loss: {np.linalg.norm(gradient(dataset[split_index:], weights))}')
#     return weights

# weights = train_model(dataset, W, split_index=split_index, num_epochs=40, lr=0.00001)

# def top_k_accuracy_weighted(dataset: list=dataset, weights: np.ndarray=weights, top_k: int=top_k, split_index: int=split_index) -> float:
#     correct = 0
#     for i in range(len(dataset[split_index:])):
#         cosines = [weighted_cosine_distance(dataset[i+split_index][0], dataset[j+split_index][1], weights) for j in range(len(dataset[split_index:]))] # Guess among test points
#         # cosines = [weighted_cosine(dataset[i+split_index][0], dataset[j][1], weights) for j in range(len(dataset))] # Guess among all data points
        
#         # Sort the cosine distances and get the top-k indices with the smallest distances
#         top_k_indices = np.argsort(cosines)[:top_k]
#         if i in top_k_indices:
#             correct += 1
#     return correct / len(dataset[split_index:])

# print(top_k_accuracy_weighted(dataset, weights, 20, split_index))





##### WEIGHTED COSINE DISTANCE PYTORCH MODEL #####

# Define the device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

# Helper class to convert the dataset to torch tensors
class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset: list=dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return torch.Tensor(self.dataset[idx][0], device=device), torch.Tensor(self.dataset[idx][1], device=device)

# Helper class for the loss function
class WeightedCosineDistanceLoss(nn.Module):
    def __init__(self):
        super(WeightedCosineDistanceLoss, self).__init__()

    def forward(self) -> torch.Tensor:
        return 1 - (torch.matmul(torch.exp(self.W), u) @ torch.matmul(torch.exp(self.W), v)) / (torch.norm(torch.matmul(torch.exp(self.W), u)) * torch.norm(torch.matmul(torch.exp(self.W), v)))

# Helper class for the optimizer
class ProbOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001):
        super(ProbOptimizer, self).__init__(params, defaults=dict(lr=lr))

    # def Probs(self) -> torch.Tensor:
    #     distances = []
    #     for i in range(len(self.dataset)):
    #         distances.append(self.forward(self.dataset[i][0], self.dataset[i][1]))
    #     return torch.softmax(torch.Tensor(distances), dim=0)
    
    # def gradient(self) -> torch.Tensor:
    #     probs = self.Probs()
    #     grad = torch.zeros(384)
    #     for i in range(len(self.dataset)):
    #         grad += (probs[i] - 1) * (self.dataset[i][0] - self.dataset[i][1])
    #     return grad

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                p.data.add_(-group['lr'], grad)

# Helper function to split the dataset into training and validation sets and create the dataloaders
def split_dataset(dataset: list=dataset, batch_size: int=32, train_size: float=0.8):
    train_size = int(len(dataset) * train_size)
    train_dataset = Dataset(dataset[:train_size])
    val_dataset = Dataset(dataset[train_size:])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Define the training loop
def train_model(model, criterion, optimizer, train_loader, num_epochs: int=10, lr: float=0.01) -> None:
    model.to(device)
    # best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # loss = criterion(output, torch.ones(u.shape[0], device=device))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if (i) % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Step {i}/{len(train_loader)}, Loss: {loss.item()}')
            
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss / len(train_loader)}')

# Define the evaluation loop
def evaluate_model(model, criterion, val_loader) -> None:
    model.to(device)
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

        print(f'Validation Loss: {val_loss / len(val_loader)}')

# Define the model
class WeightedCosineDistanceModel(nn.Module):
    def __init__(self, dataset: list=dataset, num_features: int=384, num_epochs: int=10, lr: float=0.01):
        self.W = nn.Parameter(torch.rand(num_features, num_features))
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.lr = lr

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return 1 - (torch.matmul(torch.exp(self.W), u) @ torch.matmul(torch.exp(self.W), v)) / (torch.norm(torch.matmul(torch.exp(self.W), u)) * torch.norm(torch.matmul(torch.exp(self.W), v)))
    


model = WeightedCosineDistanceModel(dataset, num_features=384, num_epochs=10, lr=0.01)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# criterion = nn.CosineSimilarity()
criterion = WeightedCosineDistanceLoss()

train_loader, val_loader = split_dataset(dataset, batch_size=32, train_size=0.8)

train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10, lr=0.01)
evaluate_model(model, criterion, val_loader)
