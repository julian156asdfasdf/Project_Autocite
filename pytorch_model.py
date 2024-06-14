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
    def __init__(self, dataset, train=True, device=torch.device('cpu')):
        self.is_train = train
        self.dataset = dataset
        self.device = device

        self.contexts = torch.tensor(dataset[:, 0], device=self.device)
        self.articles = torch.tensor(dataset[:, 1], device=self.device)
        # self.contexts = dataset[:, 0]
        # self.articles = dataset[:, 1]
        self.indexes = np.arange(len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        anchor = self.contexts[idx]
        target_article = self.articles[idx]

        # If training, sample a positive and negative article, otherwise return the target article
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
    @torch.jit.script
    def weighted_squared_euclidean(anchor: torch.Tensor, abstract: torch.Tensor, A: torch.Tensor):
        dist = torch.diag((anchor-abstract) @ A @ (anchor-abstract).T)    
        return dist
    
    @torch.jit.script
    def weighted_euclidean(anchor: torch.Tensor, abstract: torch.Tensor, A: torch.Tensor):
        dist = torch.sqrt(torch.diag((anchor-abstract) @ A @ (anchor-abstract).T))   
        return dist
    
    @torch.jit.script
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
        self.W = nn.Parameter(torch.zeros(num_features, device=self.device)) # torch.randn(num_features)
        self.alpha = torch.tensor(alpha, device=self.device)
        self.d_func = d_func

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        A = torch.diag(torch.exp(self.W))

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
                lr_scheduler: optim.lr_scheduler,
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
        lr_scheduler: A PyTorch learning rate scheduler.
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
            # anchor, positive, negative = anchor.to(model.device), positive.to(model.device), negative.to(model.device)

            # Forward pass
            # outputs = model(anchor, positive, negative)
            
            loss = criterion(anchor, positive, negative, model.W, model.alpha, model.d_func)

            # Backward pass and optimization
            if loss.item() > 0:
                # optimizer.zero_grad()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            epoch_train_loss = np.append(epoch_train_loss, loss.item())
            running_train_loss = np.append(running_train_loss, loss.item())

        # Update the learning rate
        lr_scheduler.step(epoch_train_loss.mean())

        if print_loss:
            print(f'Epoch {epoch+1}/{num_epochs}, Epoch Loss: {epoch_train_loss.mean()}, Running Training Loss: {running_train_loss.mean()}, LR: {optimizer.state_dict()["param_groups"][0]["lr"]}')
        
        if save_model:
            torch.save(model.state_dict(), 'triplet_model.pth')

        if eval_every is not None and eval_every > 0 and (epoch+1) % eval_every == 0:
            accuracy = compute_topk_accuracy(model, torch.tensor(np.unique(DATASET[:,1], axis=0), device=model.device), test_loader, top_k, mini_eval=0, print_testing_time=True)
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
        ax.text(0.45, 0.40, plot_text, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
        os.makedirs('Plots', exist_ok=True)
        plt.savefig(f'Plots/top_{top_k}_accuracy_{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}.png')
        plt.show()

    return None

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

    # If mini_eval is not 0, only evaluate the model on a subset of the test set
    if mini_eval != 0:
        total = mini_eval
    else:
        total = len(test_loader)

    with torch.no_grad():
        for i, (anchor, target_article) in enumerate(tqdm(test_loader, desc='Testing', total=total, leave=False)):
            if mini_eval != 0 and i > mini_eval:
                break
            # anchor, target_article = anchor.to(model.device), target_article.to(model.device)
            distance_to_target = model.d_func(anchor, target_article, A)
            closer_dist_counter = 0

            for article in targets:
                # article = torch.from_numpy(article).to(model.device) #.unsqueeze(0)

                # Compute the distance between the anchor and the article, and check if it's closer than the target article
                D = model.d_func(anchor, article, A)
                if D < distance_to_target:
                    closer_dist_counter += 1
                if closer_dist_counter >= k:
                    break

            # If the target article is among the k closest articles, increment the top-k accuracy
            if closer_dist_counter < k:
                topk_accuracy += 1

    model.train()

    if print_testing_time:
        end_time = time.time()
        print(f'Testing finished. Took {time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))}')

    return topk_accuracy / len(test_loader)  

if __name__ == '__main__':
    # DATASET = np.array(pd.read_pickle('Transformed_datasets_minilm/transformed_dataset_length5000_contextsize500.pkl'))
    # DATASET = np.array(pd.read_pickle('Transformed_datasets_snowflake/transformed_dataset_snowflake_len5000_context1000.pkl'))
    DATASET = np.array(pd.read_pickle('Transformed_datasets_snowflake/transformed_dataset_snowflake.pkl'))

    # Define the device
    # MPS is only faster for very large tensors/batch sizes
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu') 
    device = torch.device('cpu')

    # Set the seed
    SEED = 2 # 3
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Define variables
    num_features = 768
    batch_size = 64
    num_workers = 3
    pin_memory = True if device.type == 'cuda' else False
    num_epochs = 50
    lr = 1e-1
    margin = 0.6
    top_k = 20
    train_size = int(len(DATASET) * 0.9)
    
    # Split the dataset into a training and test set, and create DataLoaders
    train_set = arXivDataset(DATASET[:train_size],
                            train=True, 
                            device=device)
    train_loader = DataLoader(train_set, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=num_workers, 
                              pin_memory=pin_memory) # drop_last=True

    test_set = arXivDataset(DATASET[train_size:], 
                            train=False, 
                            device=device)
    test_loader = DataLoader(test_set, 
                             batch_size=1, 
                             shuffle=False,
                             num_workers=num_workers if len(DATASET) > 10000 else 0) # Batch size must be 1 for top-k accuracy

    # Create instances of the model, loss function, optimizer and learning rate scheduler
    model = TripletModel(num_features, alpha=margin, d_func=Distance.weighted_squared_euclidean, device=device).to(device)
    # model.load_state_dict(torch.load('triplet_model.pth')) # Load the model
    # model = torch.jit.script(model) # Using TorchScript for performance

    criterion = TripletLoss().to(device)
    # criterion = torch.jit.script(criterion) # Using TorchScript for performance
    # PyTorch also has a built-in TripletMarginLoss, but haven't tested whether it's compatible with the weight matrix approach
    # criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    # Remember to update plot functions when changing optimizer
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)  

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

    accuracy = compute_topk_accuracy(model, 
                                     torch.tensor(np.unique(DATASET[:,1], axis=0), device=model.device), 
                                     test_loader, 
                                     top_k, 
                                     mini_eval=0, 
                                     print_testing_time=True)
    print(37*'-')
    print(f'Top-{top_k} Accuracy Before Training: {accuracy}')
    print(37*'-')

    # Train the model
    train_model(model, 
                criterion, 
                optimizer, 
                lr_scheduler, 
                train_loader, 
                num_epochs=num_epochs, 
                print_loss=True,
                save_model=False,
                eval_every=5, 
                plot_loss=True,
                plot_eval=True)

    # Evaluate the model
    # accuracy = compute_topk_accuracy(model, 
    #                                  torch.tensor(np.unique(DATASET[:,1], axis=0), device=model.device), 
    #                                  test_loader, 
    #                                  top_k, 
    #                                  mini_eval=200, 
    #                                  print_testing_time=True)
    # print(f'Top-{top_k} Accuracy: {accuracy}')
