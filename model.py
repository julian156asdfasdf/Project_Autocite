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


# dataset_embedding.download_dataset('dataset.pkl')
# dataset = pd.DataFrame(pd.read_pickle('dataset.pkl'))
# dataset.columns = ['from_arxiv_id', 'to_arxiv_id', 'context']

# test = dataset_embedding.transform_dataset(dataset)
# test_random = dataset.sample(frac=1).reset_index(drop=True)

dataset = pd.read_pickle('transformed_dataset.pkl')



##### NAIVE COSINE MODEL #####
# def naive_cosine(x, y) -> float:
#     return cosine(x, y)

# # Calculate the top-k accuracy of the naive cosine model
top_k = 10
# def top_k_accuracy(dataset: list=dataset, top_k: int=top_k) -> float:
#     correct = 0
#     for i in range(len(dataset)):
#         cosines = [naive_cosine(dataset[i][0], dataset[j][1]) for j in range(len(dataset))]

#         # Sort the cosine distances and get the top-k indices with the smallest distances
#         top_k_indices = np.argsort(cosines)[:top_k]
#         if i in top_k_indices:
#             correct += 1
#     return correct / len(dataset)

# print(top_k_accuracy(dataset, 20))

# cosines = []
# cosines_bad = []
# for i in range(len(test)-1):
#     cosines.append(naive_cosine(test['context'][i], test['abstract'][i]))

#     cosines_bad.append(naive_cosine(test_random['context'][i], test_random['abstract'][i]))


# mean_1 =np.mean(cosines)
# mean_2= np.mean(cosines_bad)
# print(mean_1, mean_2)



##### WEIGHTED COSINE MODEL #####

# Define the loss function
def weighted_cosine(x, y, weights) -> float:
    return cosine(x, y, np.exp(weights))

# Initialize the weights
weights = np.random.rand(384)

# Calculate the probability distribution of the cosine similarities
def Probs(dataset: list=dataset, weights: np.ndarray=weights) -> np.ndarray:
    cosines = []
    for i in range(len(dataset)):
        cosines.append(weighted_cosine(dataset[i][0], dataset[i][1], weights))
    return softmax(cosines)

# Calculate the gradient of the loss function
def gradient(dataset: list=dataset, weights: np.ndarray=weights) -> np.ndarray:
    probs = Probs(dataset, weights)
    grad = np.zeros(384)
    for i in range(len(dataset)):
        grad += (probs[i] - 1) * (dataset[i][0] - dataset[i][1])
    return grad

# Update the weights using gradient descent (weights must be positive)
def update_weights(dataset: list=dataset, weights: np.ndarray=weights, lr: float=0.01) -> np.ndarray:
    grad = gradient(dataset, weights)
    weights -= lr * grad
    return weights

# Train the model
def train_model(dataset: list=dataset, weights: np.ndarray=weights, num_epochs: int=10, lr: float=0.01) -> np.ndarray:
    for epoch in range(num_epochs):
        weights = update_weights(dataset[:split_index], weights, lr)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {np.linalg.norm(gradient(dataset[:split_index], weights))}')
        print(f'Epoch {epoch+1}/{num_epochs}, Test Loss: {np.linalg.norm(gradient(dataset[split_index:], weights))}')
    return weights

# Split the dataset into training and testing sets
split_index = int(len(dataset) * 0.8)
weights = train_model(dataset, weights, num_epochs=40, lr=0.0005)

# Calculate the top-k accuracy of the weighted cosine model
def top_k_accuracy_weighted(dataset: list=dataset, weights: np.ndarray=weights, top_k: int=top_k, split_index: int=split_index) -> float:
    correct = 0
    for i in range(len(dataset[split_index:])):
        cosines = [weighted_cosine(dataset[i+split_index][0], dataset[j+split_index][1], weights) for j in range(len(dataset[split_index:]))] # Guess among test points
        # cosines = [weighted_cosine(dataset[i+split_index][0], dataset[j][1], weights) for j in range(len(dataset))] # Guess among all data points
        
        # Sort the cosine distances and get the top-k indices with the smallest distances
        top_k_indices = np.argsort(cosines)[:top_k]
        if i in top_k_indices:
            correct += 1
    return correct / len(dataset[split_index:])

print(top_k_accuracy_weighted(dataset, weights, 20, split_index))



# # Initialize weight matrix (diagonal matrix of size 384x384)
# def initialize_weights():
#     return np.eye(384)

# # Calculate probabilities of each class
# def Probs(x, y, W):
#     k = len(y)
#     # theta = np.zeros(k)
#     # for i in range(k):
#     #     theta[i] = np.exp(cosine(x, y[i], W)) / np.sum(np.exp(cosine(x, y, W)))
#     # probs = softmax(theta)
#     probs = softmax([cosine(x, y[i], W) for i in range(k)])
#     return probs

# # Calculate loss
# def Loss(x, y, W):
#     return -np.log(Probs(x, y, W))

# # Calculate gradient
# def Gradient(x, y, W):
#     k = len(y)
#     grad = np.zeros((k, 384))
#     for i in range(k):
#         grad[i] = x - y[i]
#     return grad

# # Update weights
# def Update(x, y, W, lr):
#     W -= lr * Gradient(x, y, W)
#     return W

# # Train model
# def train(x, y, W, lr, epochs):
#     for _ in tqdm(range(epochs)):
#         for i in range(len(x)):
#             W = Update(x[i], y, W, lr)
#     return W

# # Predict class
# def predict(x, y, W):
#     return Probs(x, y, W)

# # Evaluate model
# def evaluate(x, y, W):
#     return Loss(x, y, W)

# # Main function
# def main():
#     x = np.random.rand(100, 384)
#     y = np.random.rand(100, 384)
#     W = initialize_weights()
#     lr = 0.01
#     epochs = 10
#     W = train(x, y, W, lr, epochs)
#     print(predict(x, y, W))
#     print(evaluate(x, y, W))

# if __name__ == '__main__':
#     main()




##### NAIVE EUCLIDEAN MODEL #####

def naive_euclidean(x, y) -> float:
    return euclidean(x, y)

# Calculate the top-k accuracy of the naive euclidean model
def top_k_accuracy_euclidean(dataset: list=dataset, top_k: int=top_k) -> float:
    correct = 0
    for i in range(len(dataset)):
        euclideans = [naive_euclidean(dataset[i][0], dataset[j][1]) for j in range(len(dataset))]

        # Sort the euclidean distances and get the top-k indices with the smallest distances
        top_k_indices = np.argsort(euclideans)[:top_k]
        if i in top_k_indices:
            correct += 1
    return correct / len(dataset)

print(top_k_accuracy_euclidean(dataset, 20))



##### WEIGHTED EUCLIDEAN MODEL #####





##### PYTORCH MODEL #####

# TODO:
# Define loss function, i.e., distance function we discussed
# Decide on the model architecture
# Figure out how to evaluate the model, i.e., what accuracy metric to use

# Define the device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

# Helper class for the dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.tensor(self.df['input'].values[idx]), torch.tensor(self.df['arXiv-id'].values[idx])

# Helper class for the loss function
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, outputs, targets):
        # The loss function is the weighted cosine similarity between the outputs and targets
        # The weights are what the model is trying to learn by minimizing the loss
        loss = 1 - torch.nn.functional.cosine_similarity(outputs, targets)
        return torch.mean(torch.abs(outputs - targets))

# Helper function to split the dataset into training and validation sets and create the dataloaders
def create_dataloaders(df, batch_size=32, validation_size=0.2):
    # Split the dataset
    train_size = int((1 - validation_size) * len(df))
    train_df, val_df = df[:train_size], df[train_size:]
    
    # Create the dataloaders
    train_dataset = Dataset(train_df)
    val_dataset = Dataset(val_df)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# Helper function to train the model
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    model.to(device)
    best_loss = np.inf

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if (i) % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Step {i}/{len(train_loader)}, Loss: {loss.item()}')

        train_loss /= len(train_loader)
        print(f'Training Loss: {train_loss}')

        # # Save the best model
        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     torch.save(model.state_dict(), 'best_model.pth')
        #     print('Model saved')

# Helper function to evaluate the model
def evaluate_model(model, df):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(len(df)):
            inputs = torch.tensor(df['input'].values[i]).to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
    return predictions

    # # Validation loop
    # model.eval()
    # val_loss = 0
    # with torch.no_grad():
    #     for i, (inputs, targets) in enumerate(val_loader):
    #         inputs, targets = inputs.to(device), targets.to(device)

    #         # Forward pass
    #         outputs = model(inputs)
    #         loss = criterion(outputs, targets)
    #         val_loss += loss.item()

    # val_loss /= len(val_loader)
    # print(f'Validation Loss: {val_loss}')

# Define the model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(384, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out