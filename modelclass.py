import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from scipy.spatial.distance import cosine
from scipy.special import softmax
from thefuzz import fuzz
import dataset_embedding
import pandas as pd
from scipy.spatial.distance import euclidean, cityblock






class model_training_and_testing:
    def __init__(self, weights = None, model = None, dataset = None, lr=0.001, num_epochs = 40, topk = 10):
        self.weights = weights
        self.model = model
        self.dataset = dataset
        self.split_index = 0 #int(len(self.dataset)*0.8)
        self.lr = lr
        self.num_epochs = num_epochs
        self.topk = topk

    def probs(self, dataset):
        # Defines the predicted probability 
        #  
        #
        probabilities = []
        for i in range(len(dataset)):
            probabilities.append(softmax(self.model(dataset[i][0], dataset[i][1], self.weights)))
        return probabilities

    def gradient(self, dataset) -> np.ndarray:
        probs = self.probs(dataset)
        grad = np.zeros(384)
        for i in range(len(dataset)):
            grad += (probs[i] - 1) * (dataset[i][0] - dataset[i][1])
        return grad
    
    def update_weights(self) -> np.ndarray:
        grad = self.gradient(self.dataset[:self.split_index])
        self.weights -= self.lr * grad

    def train_model(self) -> np.ndarray:
        for epoch in range(self.num_epochs):
            weights = self.update_weights()
            #print(f'Epoch {epoch+1}/{self.num_epochs}, trian Loss: {np.linalg.norm(self.gradient(dataset=dataset[:self.split_index]))}')
            #print(f'Epoch {epoch+1}/{self.num_epochs}, test Loss: {np.linalg.norm(self.gradient(dataset=dataset[self.split_index:]))}')

        return weights
    

    def top_k_accuracy_weighted(self) -> float:
        correct = 0
        for i in range(len(dataset[self.split_index:])):
            probs = [self.model(self.dataset[i+self.split_index][0], self.dataset[j+self.split_index][1], self.weights) for j in range(len(self.dataset[self.split_index:]))] # Guess among test points
 
            top_k_indices = np.argsort(probs)[:self.topk]
            if i in top_k_indices:
                correct += 1
        return correct / len(dataset[self.split_index:])



#---
# Defines possible distance measures, both naive ones where the weights are constant and diagonal matrices with weights
# And filled matrices with weights, the two distance metrics used is currently Euclidean distance and Cosine distance
#---


def diagonal_euclidean_func(x, y, weights) -> float:
        return euclidean(x, y, np.exp(weights))

def naive_euclidean(model):
    # Naive euclidean
    if model=="mpnet" or model=="snow":
        weights = np.zeros(768)
    else:
        weights = np.zeros(384)
    # elif model == "":
    
    naive_euclidean = model_training_and_testing(weights=weights, dataset = dataset, model = diagonal_euclidean_func)
    topk_performance = naive_euclidean.top_k_accuracy_weighted()
    print(f'Naive Euclidean: {topk_performance}')

def diagonal_euclidean(model="mpnet"):
    # Diagonal Euclidean
    weights = np.random.rand(384)
    cosine_class = model_training_and_testing(weights=weights, dataset = dataset, model = diagonal_euclidean_func)
    topk_performance = cosine_class.top_k_accuracy_weighted()
    print(f'Diagonal Euclidean: {topk_performance}')

def matrix_weighted_euclidean(x, y, weights):
    W = np.exp(weights)
    return ((W @ (x - y))@(W @ (x - y)))**(1/2)

def matrix_euclidean():
    # Initialize W
    weights = np.random.rand(384, 384)
    matrix_euclidean = model_training_and_testing(weights=weights, dataset = dataset, model = matrix_weighted_euclidean)
    weights = matrix_euclidean.train_model()
    topk_performance = matrix_euclidean.top_k_accuracy_weighted()
    print(f'Matrix euclidean: {topk_performance}')
   
def diagonal_cosine_func(x, y, weights) -> float:
    return cosine(x, y, np.exp(weights))

def naive_cosine(model):
    # Naive cosine
    if model=="mpnet" or model=="snow":
        weights = np.zeros(768)
    elif model=="MiniLM":
        weights = np.zeros(384)
    cosine_class = model_training_and_testing(weights=weights, dataset = dataset, model = diagonal_cosine_func)
    topk_performance = cosine_class.top_k_accuracy_weighted()
    print(f'naive cosine: {topk_performance}')

def diagonal_manhattan_func(x, y, weights) -> float:
    return cityblock(x, y, np.exp(weights))

def naive_manhattan(model):
    # Naive cosine
    if model=="mpnet" or model=="snow":
        weights = np.zeros(768)
    elif model=="MiniLM":
        weights = np.zeros(384)
    cosine_class = model_training_and_testing(weights=weights, dataset = dataset, model = diagonal_manhattan_func)
    topk_performance = cosine_class.top_k_accuracy_weighted()
    print(f'naive Manhattan: {topk_performance}')


def diagonal_cosine():
    # Diagonal Cosine
    weights = np.random.rand(384)
    cosine_class = model_training_and_testing(weights=weights, dataset = dataset, model = diagonal_cosine_func)
    weights = cosine_class.train_model()
    topk_performance = cosine_class.top_k_accuracy_weighted()
    print(f'diagonal cosine: {topk_performance}')

def matrix_weighted_cosine(x, y, weights) -> float:
        W = np.exp(weights)
        return 1 - (W @ x) @ (W @ y) / (np.linalg.norm(W @ x) * np.linalg.norm(W @ y))

def matrix_cosine():
    # Initialize W
    weights = np.random.rand(384, 384)
    cosine_matrix = model_training_and_testing(weights=weights, dataset = dataset, model = matrix_weighted_cosine)
    weights = cosine_matrix.train_model()
    topk_performance = cosine_matrix.top_k_accuracy_weighted()
    print(f'Matrix cosine: {topk_performance}')


def main(model):
    # naive_euclidean(model)
    # diagonal_euclidean(model)
    # matrix_euclidean()

    # naive_cosine(model)
    naive_manhattan(model)
    # diagonal_cosine()
    # matrix_cosine()

if __name__ == "__main__":
    dataset = pd.read_pickle('transformed_dataset_mpnet_contextsize300.pkl')[:5000]
    print("\nmpnet: ")
    main(model = "mpnet")
    dataset = pd.read_pickle('transformed_dataset_MiniLM_contextsize300.pkl')[:5000]
    print("\nMiniLM: ")
    main(model="MiniLM")    
    dataset = pd.read_pickle('transformed_dataset_snow_contextsize300.pkl')[:5000]
    print("\nsnow: ")
    main(model="snow")
    