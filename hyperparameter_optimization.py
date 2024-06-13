import time
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import pickle

# Install the following packages if not already installed
# > pip install optuna
# > pip install optuna-dashboard
# Install Optuna Dashboard extension on VS Code : https://marketplace.visualstudio.com/items?itemName=Optuna.optuna-dashboard#overview
import optuna
from optuna.trial import TrialState

##### TRIPLET LOSS PYTORCH HYPERPARAMETER OPTIMIZATION MODEL #####

# --------------------------------------------------------------
# Import the necessary functions and classes
from pytorch_model import arXivDataset, Distance, TripletLoss, TripletModel, compute_topk_accuracy
# --------------------------------------------------------------


def objective(trial):
    """
    The objective function for the hyperparameter optimization.

    Parameters:
    trial (optuna.trial.Trial): The trial object.

    Returns:
    float: The top-k accuracy of the model.
    """

    # Load the dataset with the context size hyperparameter
    
    context_size_idx = trial.suggest_int("context_size_idx", 0, len(context_sizes)-1)
    context_size = context_sizes[context_size_idx]
    
    path = transformed_dataset_filename_base[:-4] + str(context_size) + transformed_dataset_filename_base[-4:]
    dataset = np.array(pd.read_pickle(path))[:dataset_size]
    num_features = dataset.shape[2]
    train_size = int(len(dataset) * train_test_split)

    # Set the seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Get a sample of the hyperparameters alpha and distance measure
    alpha = trial.suggest_float("alpha", 0.05, 1.5, log=True)
    d_func_name = trial.suggest_categorical("distance_measure", ["weighted_squared_euclidean", "weighted_euclidean", "weighted_manhatten"]) #, "weighted_cosine"]) 
    d_func = getattr(Distance, d_func_name)

    # Initialize the model, criterion and optimizer
    model = TripletModel(num_features, alpha, d_func, device=device).to(device)
    criterion = TripletLoss().to(model.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initialize the data loaders
    train_set = arXivDataset(dataset[:train_size], train=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers_train, shuffle=True) # drop_last=True
    test_set = arXivDataset(dataset[train_size:], train=False)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=num_workers_test, shuffle=False) # Batch size must be 1 for top-k accuracy

    # Update the running weights with the initial weights
    running_weights[trial.number, :, 0] = model.W.detach().numpy()

    # Train the model
    print(f"\nTrial: {trial.number}. Training model with hyperparameters: Context Size: {context_size}, Alpha: {alpha}, Distance Measure: {d_func_name}")
    running_train_loss = np.array([]) # Running loss for each batch
    for epoch in range(num_epochs):
        # Get the start time of the epoch training
        start_time = time.time()
        test_start_localtime = time.strftime("%H:%M:%S", time.localtime(start_time))

        eopch_train_loss = np.array([]) # Loss for each epoch
        model.train() # Set the model to training mode
        for batch_idx, (anchor, positive, negative) in enumerate(tqdm(train_loader, desc=f'Training in epoch {epoch+1}', leave=False)):
            # Limiting training data for faster epochs.
            anchor, positive, negative = anchor.to(model.device), positive.to(model.device), negative.to(model.device)

            # Forward pass
            loss = criterion(anchor, positive, negative, model.W, model.alpha, model.d_func)

            # Backward pass and optimization
            if loss.item() > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # Store the loss, running loss and weights of batch and epoch    
            eopch_train_loss = np.append(eopch_train_loss, loss.item())
            running_train_loss = np.append(running_train_loss, loss.item())
        running_weights[trial.number, :, epoch+1] = model.W.detach().numpy()

        # Validation of the model.
        targets = np.unique(np.asarray(test_set).T[:,1].T, axis=0)
        # targets = np.unique(dataset[:,1], axis=0)
        accuracy = compute_topk_accuracy(model, targets, test_loader, top_k, mini_eval=mini_eval, print_testing_time=False)
        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Save the running weights
        with open('running_weights_hyper_opt.pkl', 'wb') as f:
            pickle.dump(running_weights, f)

        # Print the epoch statistics
        end_time = time.time()
        test_end_localtime = time.strftime("%H:%M:%S", time.localtime(end_time))
        print(f'Epoch {epoch+1:02d}, Top-{top_k} Accuracy: {accuracy:.3f}, Epoch Training Loss: {eopch_train_loss.mean():.4f}, Running Training Loss: {running_train_loss.mean():.4f}                 Time: {test_start_localtime}-{test_end_localtime}, Took: {time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))}')

    
    return accuracy

def get_study(storage, study_name, continue_existing, search_space=None, sampler=None):
    """
    Get the study from the storage or create a new study if it does not exist.

    Parameters:
    storage (str): The storage location for the study.
    study_name (str): The name of the study.
    continue_existing (bool): Whether to continue an existing study.
    search_space (dict): The search space for the hyperparameters.
    sampler (optuna.samplers.BaseSampler): The sampler for the study.

    Returns:
    optuna.study.Study: The study.
    """
    try:
        # Check if the study exists
        study = optuna.load_study(study_name=study_name, storage=storage, sampler=(sampler if sampler is not None else optuna.samplers.BaseSampler()))
        
        # If the study exists, delete it
        if not continue_existing:
            optuna.delete_study(study_name=study_name, storage=storage)
            print(f"Existing study '{study_name}' deleted.")

    except KeyError:
        # If the study does not exist, pass
        continue_existing = False
        print(f"No existing study named '{study_name}' found.")

    # Create a new study if not continuing an existing one
    if not continue_existing:
        # Create a new study
        study = optuna.create_study(
            storage=storage,
            study_name=study_name,
            direction="maximize",
            sampler=sampler
            )
    
    return study

if __name__ == "__main__":
    # Set the seed
    SEED = 2
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Define the device
    device = torch.device('cpu')

    # Define variables
    train_test_split = 0.9
    dataset_size = 5000
    batch_size = 64
    num_epochs = 20
    top_k = 20
    lr = 0.01
    mini_eval = 0
    num_workers_train = 6
    num_workers_test = 0

    # Hyperparameter optimization variables
    context_sizes = [50,100,200,300,400,500,600,700,800,900,1000]
    alpha_bound = [0.05, 1.5]
    distance_measures = ["weighted_squared_euclidean", "weighted_euclidean", "weighted_manhatten"] #, "weighted_cosine"]
    study_name = "Autocite_Hyperparam_Optim_Snowflake_AccOnTestOnly"
    storage = "sqlite:///Autocite.db"
    continue_existing = False # Set to True if you want to continue an existing study with the same name and stored at the same location.
    n_trials = 40

    # Choose the vector embedding model
    vector_embedding_model = 'Snowflake' # Choose from 'Snowflake', 'MiniLM'

    if vector_embedding_model == 'Snowflake':
        transformed_dataset_filename_base = "Transformed_datasets_snowflake/transformed_dataset_snowflake_len5000_context.pkl"
    elif vector_embedding_model == 'MiniLM':
        transformed_dataset_filename_base = "Transformed_datasets_minilm/transformed_dataset_length5000_contextsize.pkl"
    else:
        raise ValueError("Invalid vector_embedding_model. Choose from 'Snowflake', 'MiniLM'.")

    # Initialize the running weights
    with open(transformed_dataset_filename_base[:-4] + str(1000) + transformed_dataset_filename_base[-4:], 'rb') as f:
        running_weights = np.zeros((n_trials, np.asarray(pickle.load(f)).shape[2], num_epochs+1))
    
    study = get_study(storage, study_name, continue_existing, sampler = optuna.samplers.GPSampler()) # , search_space=search_space, sampler=optuna.samplers.GridSampler
    
    # Optimize the study
    study.optimize(objective, n_trials=n_trials)

    # Print the study statistics
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    # Print the best trial
    best_trial = study.best_trial
    print("Best trial:")
    print("  Value: ", best_trial.value)
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))
    
    # Plot the top-k accuracy for each trial
    accuracies = np.array([])
    for i, trial in enumerate(complete_trials):
        accuracies = np.append(accuracies, trial.values[0])

    fig, ax = plt.subplots()
    plt.plot(accuracies)
    plt.xlabel('Trial Number')
    plt.ylabel('Learned Top-k Accuracy')
    plt.title(f'Top-{top_k} Accuracy for each Trial')
    plot_text = f'Number of epochs: {num_epochs}\nOptimizer: Adam\nLearning rate: {lr}\nTraining size: {int(dataset_size*train_test_split)}\nLoss function: Triplet Loss'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.45, 0.35, plot_text, transform=ax.transAxes, fontsize=14,
    verticalalignment='top', bbox=props)
    os.makedirs('Plots', exist_ok=True)
    plt.savefig(f'Plots/hyper_opt_top_{top_k}_accuracy_{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}.png')
    plt.show()