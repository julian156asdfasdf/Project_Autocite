import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from scipy.spatial.distance import cosine
from scipy.special import softmax
from thefuzz import fuzz
import dataset_embedding
import pandas as pd


dataset = pd.read_pickle('transformed_dataset.pkl')

def naive_cosine(x, y) -> float:
    return cosine(x, y)

# Calculate the top-k accuracy of the naive cosine model
top_k = 10
def top_k_accuracy(dataset: list=dataset, top_k: int=top_k) -> float:
    correct = 0
    for i in range(len(dataset)):
        cosines = [naive_cosine(dataset[i][0], dataset[j][1]) for j in range(len(dataset))]

        # Sort the cosine distances and get the top-k indices with the smallest distances
        top_k_indices = np.argsort(cosines)[:top_k]
        if i in top_k_indices:
            correct += 1
    return correct / len(dataset)

#print(top_k_accuracy(dataset, 20))




class model_training_and_testing:
    def __init__(self, weights = None, model = None, dataset = dataset, lr=0.001, num_epochs = 40, topk = 20):
        self.weights = weights
        self.model = model
        self.dataset = dataset
        self.split_index = int(len(self.dataset)*0.8)
        self.lr = lr
        self.num_epochs = num_epochs
        self.topk = topk

    def probs(self, dataset):
        probabilities = []
        for i in range(len(dataset)):
            probabilities.append(weighted_cosine(dataset[i][0], dataset[i][1], weights))
        return softmax(probabilities)

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
            print(f'Epoch {epoch+1}/{self.num_epochs}, trian Loss: {np.linalg.norm(self.gradient(dataset=dataset[:self.split_index]))}')
            print(f'Epoch {epoch+1}/{self.num_epochs}, test Loss: {np.linalg.norm(self.gradient(dataset=dataset[self.split_index:]))}')

        return weights
    

    def top_k_accuracy_weighted(self) -> float:
        correct = 0
        for i in range(len(self.dataset[self.split_index:])):
            # cosines = [weighted_cosine(dataset[i+split_index][0], dataset[j+split_index][1], weights) for j in range(len(dataset[split_index:]))] # Guess among test points
            cosines = [weighted_cosine(self.dataset[i+self.split_index][0], self.dataset[j][1], self.weights) for j in range(len(self.dataset))] # Guess among all data points
            
            # Sort the cosine distances and get the top-k indices with the smallest distances
            top_k_indices = np.argsort(cosines)[:top_k]
            if i in top_k_indices:
                correct += 1
        return correct / len(dataset[self.split_index:])








if __name__ == "__main__":
    def weighted_cosine(x, y, weights) -> float:
        return cosine(x, y, np.exp(weights))
    # Initialize the weights
    weights = np.random.rand(384)
    cosine_class = model_training_and_testing(weights=weights, dataset = dataset, model = weighted_cosine)
    weights = cosine_class.train_model()
    topk_performance = cosine_class.top_k_accuracy_weighted()
    print(topk_performance)

