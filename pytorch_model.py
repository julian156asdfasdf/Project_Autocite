import time
import random
import torch
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from scipy.special import softmax

import dataset_embedding

dataset = np.array(pd.read_pickle('transformed_dataset.pkl'))

# Set the seed
# SEED = 3
SEED = 2
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

##### TRIPLET LOSS PYTORCH MODEL #####

# Define the device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

# Define variables
num_features = 384
batch_size = 384
num_epochs = 10
lr = 1e-3
train_size = int(len(dataset) * 0.8)
top_k = 20

# Define the dataset
class arXivDataset(Dataset):
    def __init__(self, dataset, train=True):
        self.is_train = train
        self.dataset = dataset

        # if self.is_train:
        self.contexts = dataset[:, 0]
        self.articles = dataset[:, 1]
        self.indexes = np.arange(len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        anchor = self.contexts[idx]

        if self.is_train:
            positive_list = np.unique(np.where(self.articles == self.articles[idx])[0])
            negative_list = np.array([idx for idx in self.indexes if idx not in positive_list])

            # negative_list = self.indexes[self.articles != self.articles[idx]]

            positive_idx = np.random.choice(positive_list)
            negative_idx = np.random.choice(negative_list)

            positive = self.articles[positive_idx]
            negative = self.articles[negative_idx]
            # positive_idx = np.random.choice(self.indexes[self.indexes != idx])
            # negative_idx = np.random.choice(self.indexes)

            # positive = self.articles[positive_idx]
            # negative = self.articles[negative_idx]

            return anchor, positive, negative
        
        return anchor
    
train_set = arXivDataset(dataset[:train_size],
                          train=True)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True) # drop_last=True

test_set = arXivDataset(dataset[train_size:], 
                         train=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False) # Batch size must be 1 for top-k accuracy

# Define the loss function
class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, W: torch.Tensor, margin: torch.Tensor) -> torch.Tensor:
        A = torch.diag(torch.exp(W))
        
        D_pos = (anchor - positive).T @ A @ (anchor - positive)
        D_neg = (anchor - negative).T @ A @ (anchor - negative)

        losses = torch.relu(D_pos - D_neg + margin)

        return torch.mean(losses)

# Define the model
class TripletModel(nn.Module):
    def __init__(self, num_features, device=device):
        super(TripletModel, self).__init__()
        # self.fc = nn.Linear(num_features, 128)
        # self.W = nn.Parameter(torch.randn(num_features, num_features)) # Skal måske være np.eye(num_features)
        self.device = device
        self.W = nn.Parameter(torch.randn(num_features))
        self.alpha = nn.Parameter(torch.randn(1))
        self.A = torch.diag(torch.exp(self.W))
        self.A = self.A.to(self.device)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        D_pos = (anchor - positive).T @ self.A @ (anchor - positive)
        D_neg = (anchor - negative).T @ self.A @ (anchor - negative)

        losses = torch.relu(D_pos - D_neg + self.alpha)

        return torch.mean(losses)

# Create instances of the model, loss function and optimizer
model = TripletModel(num_features).to(device)
model = torch.jit.script(model) # Using TorchScript for performance

# criterion = TripletLoss().to(device)
criterion = torch.jit.script(TripletLoss()).to(device)
# PyTorch also has a built-in TripletMarginLoss, but haven't tested whether it's compatible with the weight matrix approach
# criterion = nn.TripletMarginLoss(margin=1.0, p=2)

optimizer = optim.SGD(model.parameters(), lr=lr)

# Training loop
def train_model(model: nn.Module, 
                criterion: nn.Module, 
                optimizer: optim.Optimizer, 
                train_loader: DataLoader, 
                num_epochs: int, 
                eval_every: int,
                plot_loss: bool=False) -> None:
    """
    Train the model using the triplet loss function.

    Arguments:
        model: A PyTorch model.
        criterion: A PyTorch loss function.
        optimizer: A PyTorch optimizer.
        train_loader: A PyTorch DataLoader.
        num_epochs: An integer specifying the number of epochs.
        eval_every: An integer specifying the number of epochs between evaluations.
        plot_loss: A boolean specifying whether to plot the training loss.

    Returns:
        None

    """

    model.train()
    running_train_loss = np.array([])

    for epoch in range(num_epochs):
        for i, (anchor, positive, negative) in enumerate(tqdm(train_loader, desc='Training', leave=False)):
            if anchor.size(0) != model.W.size(0):
                continue
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            # params = [(n, p) for n, p in model.named_parameters()]
            # params[0][1], params[1][1]

            # Forward pass
            outputs = model(anchor, positive, negative)
            loss = criterion(anchor, positive, negative, model.W, model.alpha)

            # Backward pass and optimization
            if loss.item() > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_train_loss = np.append(running_train_loss, loss.item())

        if (epoch+1) % eval_every == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {running_train_loss.mean()}')

    if plot_loss:
        plt.plot(running_train_loss)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()

# Evaluation loop
def compute_topk_accuracy(model: nn.Module, 
                          criterion: nn.Module,
                          dataset: np.ndarray,
                          test_loader: DataLoader, 
                          k: int) -> float:
    """
    Compute the top-k accuracy of the model on the test set.

    Arguments:
        model: A PyTorch model.
        criterion: A PyTorch loss function.
        dataset: A NumPy array containing the dataset.
        test_loader: A PyTorch DataLoader.
        k: An integer specifying the top-k accuracy.

    Returns:
        A float representing the top-k accuracy.
    
    """

    model.eval()
    topk_accuracy = 0
    running_test_loss = np.array([])

    with torch.no_grad():
        for i, anchor in enumerate(tqdm(test_loader, desc='Testing', leave=False)):
            if anchor.size(1) != model.W.size(0):
                continue
            anchor = anchor.to(device)
            distances = np.array([])

            for j, article in enumerate(dataset[:, 1]):
                article = torch.from_numpy(article).to(device)

                # Compute distances to all possible articles
                A = torch.diag(torch.exp(model.W))
                A = A.to(model.device)
                D = (anchor - article) @ A @ (anchor - article).T
                distances = np.append(distances, D.item())

            # Get the top k articles
            topk = np.argsort(distances)[:k]

            if i in topk:
                topk_accuracy += 1
                
    return topk_accuracy / len(test_loader)

    # with torch.no_grad():
    #     for i, (anchor, positive, negative) in enumerate(tqdm(test_loader, desc='Testing', leave=False)):
    #         if anchor.size(0) != test_loader.batch_size:
    #             continue
    #         anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

    #         for j in range(anchor.size(0)):
    #             anchor_j = anchor[j]
    #             positive_j = positive[j]
    #             negative_j = negative[j]

    #             # Compute the distances
    #             D_pos = (anchor_j - positive_j).T @ torch.exp(model.W) @ (anchor_j - positive_j)
    #             D_neg = (anchor_j - negative_j).T @ torch.exp(model.W) @ (anchor_j - negative_j)

    #             if D_pos < D_neg:
    #                 topk_accuracy += 1

            # # Forward pass
            # outputs = model(anchor, positive, negative)
            # loss = criterion(anchor, positive, negative, model.W, model.alpha)

            
                

# compute_topk_accuracy(model, criterion, dataset, test_loader, top_k)
# Train the model
train_model(model, criterion, optimizer, train_loader, num_epochs=50, eval_every=1, plot_loss=False)

# Evaluate the model
accuracy = compute_topk_accuracy(model, criterion, dataset, test_loader, top_k)
print(f'Top-{top_k} Accuracy: {accuracy}')

# SEED = 1
# Epoch 50/50, Training Loss: 0.07044965393096209
# Top-20 Accuracy: 0.0070140280561122245

# Save the model
# torch.save(model.state_dict(), 'triplet_model.pth')
# torch.save({"model_state_dict": model.state_dict(),
#             "optimzier_state_dict": optimizer.state_dict()
#            }, "triplet_model.pth")

# Load the model
# model = TripletModel(num_features).to(device)
# model.load_state_dict(torch.load('triplet_model.pth'))
# model.eval()

# if __name__ == '__main__':
#     # Train the model
#     train_model(model, criterion, optimizer, train_loader, num_epochs, eval_every=1)

#     # Evaluate the model
#     # evaluate_model(model, criterion, test_loader)