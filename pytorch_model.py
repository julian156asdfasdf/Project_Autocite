import time
import random
import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from typing import Any, Callable

##### TRIPLET LOSS PYTORCH MODEL #####

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
        target_article = self.articles[idx]

        if self.is_train:
            positive_list = np.unique(np.where(self.articles == self.articles[idx])[0])
            negative_list = np.array([idx for idx in self.indexes if idx not in positive_list])

            positive_idx = np.random.choice(positive_list)
            negative_idx = np.random.choice(negative_list)

            positive = self.articles[positive_idx]
            negative = self.articles[negative_idx]

            return anchor, positive, negative
        
        return anchor, target_article

# Define distance functions
class Distance:
    """
    The possible distance functions to train
    """
    def weighted_squared_euclidean(anchor: torch.Tensor, abstract: torch.Tensor, A: torch.Tensor):
        dist = torch.diag((anchor-abstract) @ A @ (anchor-abstract).T)    
        return dist
    
    def weighted_euclidean(anchor: torch.Tensor, abstract: torch.Tensor, A: torch.Tensor):
        dist = torch.sqrt(torch.diag((anchor-abstract) @ A @ (anchor-abstract).T))   
        return dist
    
    def weighted_manhatten(anchor: torch.Tensor, abstract: torch.Tensor, A: torch.Tensor):
        return torch.sum(torch.abs(anchor - abstract) @ A, dim=1)

# Define the loss function
class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, W: torch.Tensor, margin: torch.Tensor, d_func: Callable[..., Any]) -> torch.Tensor:
        A = torch.diag(torch.exp(W))
        
        # D_pos = torch.empty(64)
        # D_neg = torch.empty(64)

        # for i in range(anchor.size(0)):
        #     D_pos[i] = (anchor[i] - positive[i]) @ A @ (anchor[i] - positive[i])
        #     D_neg[i] = (anchor[i] - negative[i]) @ A @ (anchor[i] - negative[i])

        # D_pos = torch.diag((anchor - positive) @ A @ (anchor - positive).T)
        # D_neg = torch.diag((anchor - negative) @ A @ (anchor - negative).T)

        D_pos = d_func(anchor, positive, A)
        D_neg = d_func(anchor, negative, A)

        losses = torch.relu(D_pos - D_neg + margin)

        return torch.mean(losses)

# Define the model
class TripletModel(nn.Module):
    def __init__(self, num_features, alpha, d_func, device=torch.device('cpu')):
        super(TripletModel, self).__init__()
        self.device = device
        self.W = nn.Parameter(torch.zeros(num_features)) # torch.randn(num_features)
        self.alpha = torch.tensor(alpha)
        self.d_func = d_func

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        A = torch.diag(torch.exp(self.W))
        A = A.to(self.device)

        # D_pos = torch.empty(64)
        # D_neg = torch.empty(64)

        # for i in range(anchor.size(0)):
        #     D_pos[i] = (anchor[i] - positive[i]) @ A @ (anchor[i] - positive[i])
        #     D_neg[i] = (anchor[i] - negative[i]) @ A @ (anchor[i] - negative[i])

        # D_pos = torch.diag((anchor - positive) @ A @ (anchor - positive).T)
        # D_neg = torch.diag((anchor - negative) @ A @ (anchor - negative).T)

        D_pos = self.d_func(anchor, positive, A)
        D_neg = self.d_func(anchor, negative, A)

        losses = torch.relu(D_pos - D_neg + self.alpha)

        return torch.mean(losses)

# Training loop
def train_model(model: nn.Module, 
                criterion: nn.Module, 
                optimizer: optim.Optimizer, 
                train_loader: DataLoader, 
                num_epochs: int=10, 
                print_loss: bool=True, 
                save_model: bool=False,
                eval_every: int | None=None,
                plot_loss: bool=False,
                plot_eval: bool=False) -> None:
    """
    Train the model using the triplet loss function.

    Arguments:
        model: A PyTorch model.
        criterion: A PyTorch loss function.
        optimizer: A PyTorch optimizer.
        train_loader: A PyTorch DataLoader.
        num_epochs: An integer specifying the number of epochs.
        print_loss: A boolean specifying whether to print the training loss.
        save_model: A boolean specifying whether to save the model.
        eval_every: An integer specifying the number of epochs between evaluations. If None, no evaluations are done.
        plot_loss: A boolean specifying whether to plot the training loss.
        plot_eval: A boolean specifying whether to plot the evaluation results.

    Returns:
        None

    """

    model.train()
    running_train_loss = np.array([])
    running_topk_accuracy = np.array([])
    start_time = time.time()
    print(f'Starting training for {num_epochs} epochs at {time.strftime("%H:%M:%S", time.localtime(start_time))}...')

    for epoch in range(num_epochs):
        epoch_train_loss = np.array([])
        for i, (anchor, positive, negative) in enumerate(tqdm(train_loader, desc='Training', leave=False)):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            # Forward pass
            # outputs = model(anchor, positive, negative)
            loss = criterion(anchor, positive, negative, model.W, model.alpha, model.d_func)

            # Backward pass and optimization
            if loss.item() > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_train_loss = np.append(epoch_train_loss, loss.item())
            running_train_loss = np.append(running_train_loss, loss.item())

        if print_loss:
            print(f'Epoch {epoch+1}/{num_epochs}, Epoch Loss: {epoch_train_loss.mean()}, Running Training Loss: {running_train_loss.mean()}')
        
        if save_model:
            torch.save(model.state_dict(), 'triplet_model.pth')

        if eval_every is not None and eval_every > 0 and (epoch+1) % eval_every == 0:
            accuracy = compute_topk_accuracy(model, np.unique(DATASET[:,1], axis=0), test_loader, top_k, mini_eval=200, print_testing_time=True)
            running_topk_accuracy = np.append(running_topk_accuracy, accuracy)
            print(37*'-')
            print(f'Top-{top_k} Accuracy: {accuracy}')
            print(37*'-')

    end_time = time.time()
    print(f'Training finished. Took {time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))}')

    if plot_loss:
        fig, ax = plt.subplots()
        plt.plot(running_train_loss)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plot_text = f'Number of epochs: {num_epochs}\nOptimizer: Adam\nLearning rate: {lr}\nTraining size: {train_size}\nLoss function: Triplet Loss\nTime: {time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.45, 0.95, plot_text, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
        os.makedirs('Plots', exist_ok=True)
        plt.savefig(f'Plots/triplet_loss_training_{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}.png')
        plt.show()
    
    if plot_eval:
        fig, ax = plt.subplots()
        plt.plot(running_topk_accuracy)
        plt.xlabel('Epochs (x5)')
        plt.ylabel('Top-k Accuracy')
        plt.title(f'Top-{top_k} Accuracy')
        plot_text = f'Number of epochs: {num_epochs}\nOptimizer: Adam\nLearning rate: {lr}\nTraining size: {train_size}\nLoss function: Triplet Loss\nTime: {time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.45, 0.95, plot_text, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
        os.makedirs('Plots', exist_ok=True)
        plt.savefig(f'Plots/top_{top_k}_accuracy_{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}.png')
        plt.show()

# Evaluation loop
def compute_topk_accuracy(model: nn.Module, 
                          targets: np.ndarray,
                          test_loader: DataLoader,
                          k: int=20,
                          mini_eval: int=0,
                          print_testing_time: bool=True) -> float:
    """
    Compute the top-k accuracy of the model on the test set.

    Arguments:
        model: A PyTorch model.
        criterion: A PyTorch loss function.
        targets: A numpy array containing the target articles.
        test_loader: A PyTorch DataLoader.
        k: An integer specifying the top-k accuracy.
        mini_eval: An integer specifying the number of evaluations to perform. If 0, all evaluations are performed.
        print_testing_time: A boolean specifying whether to print the testing time.

    Returns:
        A float representing the top-k accuracy.
    
    """

    model.eval()
    topk_accuracy = 0

    if print_testing_time:
        start_time = time.time()
        print(f'Starting testing at {time.strftime("%H:%M:%S", time.localtime(start_time))}...')

    A = torch.diag(torch.exp(model.W))
    # A = A.to(model.device)

    if mini_eval != 0:
        total = mini_eval
    else:
        total = len(test_loader)

    with torch.no_grad():
        for i, (anchor, target_article) in enumerate(tqdm(test_loader, desc='Testing', total=total, leave=False)):
            if mini_eval != 0 and i > mini_eval:
                break
            anchor, target_article = anchor.to(device), target_article.to(device)
            # distance_to_target = (anchor - target_article) @ A @ (anchor - target_article).T
            distance_to_target = model.d_func(anchor, target_article, A)
            # distances = np.array([])
            closer_dist_counter = 0

            for article in targets:
                article = torch.from_numpy(article).to(device) #.unsqueeze(0)

                # Compute distances to all possible articles, and check if the distance to the target article is smaller
                # D = (anchor - article) @ A @ (anchor - article).T
                D = model.d_func(anchor, article, A)
                if D < distance_to_target:
                    closer_dist_counter += 1
                if closer_dist_counter >= k:
                    break

                # distances = np.append(distances, D.item())
            # If the target article is among the k closest articles, increment the top-k accuracy
            if closer_dist_counter < k:
                topk_accuracy += 1
            # Get the top k articles
            # topk = np.argsort(distances)[:k]

            # if i + len(train_loader) * train_loader.batch_size in topk:
            #     topk_accuracy += 1
    model.train()

    if print_testing_time:
        end_time = time.time()
        print(f'Testing finished. Took {time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))}')

    return topk_accuracy / len(test_loader)  

if __name__ == '__main__':
    DATASET = np.array(pd.read_pickle('transformed_dataset_length5000_contextsize300.pkl'))

    # Set the seed
    SEED = 2 # 3
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Define the device
    # MPS is only faster for very large tensors/batch sizes
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu') 
    device = torch.device('cpu')

    # Define variables
    num_features = 384
    batch_size = 64
    num_epochs = 10
    lr = 5e-3
    margin = 0.2
    top_k = 20
    train_size = int(len(DATASET) * 0.9)
    
    # Split the dataset into a training and test set, and create DataLoaders
    train_set = arXivDataset(DATASET[:train_size],
                            train=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True) # drop_last=True

    test_set = arXivDataset(DATASET[train_size:], 
                            train=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False) # Batch size must be 1 for top-k accuracy

    # Create instances of the model, loss function and optimizer
    model = TripletModel(num_features, alpha=margin, d_func=Distance.weighted_squared_euclidean).to(device)
    # model.load_state_dict(torch.load('triplet_model.pth')) # Load the model
    # model = torch.jit.script(model) # Using TorchScript for performance

    criterion = TripletLoss().to(device)
    # criterion = torch.jit.script(criterion) # Using TorchScript for performance
    # PyTorch also has a built-in TripletMarginLoss, but haven't tested whether it's compatible with the weight matrix approach
    # criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    # Remember to update plot functions when changing optimizer
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)  

    # Train the model
    train_model(model, 
                criterion, 
                optimizer, 
                train_loader, 
                num_epochs=num_epochs, 
                print_loss=True,
                save_model=False,
                eval_every=2, 
                plot_loss=True,
                plot_eval=True)

    # Evaluate the model
    # accuracy = compute_topk_accuracy(model, criterion, DATASET, test_loader, top_k)
    # print(f'Top-{top_k} Accuracy: {accuracy}')
