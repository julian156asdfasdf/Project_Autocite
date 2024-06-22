# Creates the class for the popularity model, and evaluates all the baseline models with the possibility of using different vector embedders.
# Requires the pytorch_model.py file to be in the same directory.
# Requires the dataset file to be created with the given embedder and in the stated subdirectory.

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
import operator

from Autocite.Autocite import compute_topk_accuracy, arXivDataset, TripletModel, Distance


class PopularityModel():
    def __init__(self, train_set, top_k):
        """Initialize the popularity model"""
        # Count the number of occurrences of each label in the training set
        popularity_counter = {}
        for i in range(len(train_set)):
            label = tuple(train_set[i,1])
            if label in popularity_counter:
                popularity_counter[label] += 1
            else:
                popularity_counter[label] = 1
        self.top_k_labels_dict = self.top_k_keys(popularity_counter, k=top_k)

    def top_k_keys(self, d, k=20):
        """return the k keys with the max value"""  
        return dict(sorted(d.items(), key=operator.itemgetter(1), reverse=True)[:k])

    def predict_score(self, anchor, target_article):
        """Predict the score of the target article based on the popularity model"""
        for key in self.top_k_labels_dict.keys():
            key = list(key)
            if list(target_article.squeeze(0).detach().numpy()) == key:
                return 1
        return 0
    
    def evaluate(self, test_loader):
        """Evaluate the popularity model on the test set and return the top-k accuracy"""
        topk_accuracy = 0
        total = len(test_loader)
        with torch.no_grad(): # <- Not sure if this is necessary or even works
            for i, (anchor, target_article) in enumerate(tqdm(test_loader, desc='Testing', total=total, leave=False)):
                topk_accuracy += self.predict_score(anchor, target_article)
        return topk_accuracy / len(test_loader)*100
    
    def return_arxiv_ids(self):
        """Return the top-k labels of the popularity model"""
        return self.top_k_labels_dict

if __name__ == '__main__':
    embedder = "mpnet"
    DATASET = np.array(pd.read_pickle(f'Transformed_datasets_{embedder}/transformed_dataset_{embedder}.pkl'))[:20000]

    # Define the device
    # MPS is only faster for very large tensors/batch sizes
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu') 
    device = torch.device('cpu')

    # Set the seed
    SEED = 3 # 3
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Define variables
    num_features = DATASET.shape[2]
    margin = 0.59327 # Irrelevant for baseline models
    top_k = 20
    train_size = int(len(DATASET) * 0.9)

    # Split the dataset into a training and test set, and create DataLoaders
    train_set = arXivDataset(DATASET[:train_size],
                            train=False)
    # Split the dataset into a training and test set, and create DataLoaders
    test_set = arXivDataset(DATASET[train_size:], 
                            train=False)
    test_loader = DataLoader(test_set, batch_size=1, num_workers = 0, shuffle=False) # Batch size must be 1 for top-k accuracy

    # Create instances of the model, loss function and optimizer
    d_func_names = ["weighted_squared_euclidean", "weighted_euclidean", "weighted_manhatten"]
    i = 0 # 0: weighted_squared_euclidean, 1: weighted_euclidean, 2: weighted_manhatten

    unweighted_model = TripletModel(num_features, alpha=margin, d_func=getattr(Distance, d_func_names[i])).to(device) # Margin is irrelevant for this test
    
    targets = {"dataset labels" : torch.tensor(np.unique(DATASET[:,1], axis=0), device=unweighted_model.device), 
               "validation labels" : torch.tensor(np.unique(np.asarray(test_set).T[:,1].T, axis=0), device=unweighted_model.device),
               "train labels" : torch.tensor(np.unique(np.asarray(train_set).T[:,1].T, axis=0), device=unweighted_model.device)}
    
    acc = compute_topk_accuracy(model = unweighted_model, 
                    targets = targets["dataset labels"],
                    test_loader = test_loader,
                    k = top_k,
                    mini_eval = 0,
                    print_testing_time = True)
    
    print(f"Top-{top_k} accuracy using dataset labels and {d_func_names[i]}: {acc:.3f}")

    # Popularity model
    popularity_model = PopularityModel(DATASET[:train_size], top_k)
    acc = popularity_model.evaluate(test_loader)

    print(f"Accuracy of popularity model: {acc:.3f}%") 