# Run this to get the top-5 arXiv-IDs for the 30 contexts in the blind test for each of the models
# Requires the dataset file to be created with the snowflake embedder and in the stated subdirectory.

import pickle
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os

# Import the necessary functions and classes
from Autocite import arXivDataset, Distance, TripletModel
from baselines import PopularityModel

if __name__ == '__main__':

    SEED = 2 # 3
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device('cpu')

    # Create blind test integers
    import random
    random.seed(2)

    # Function to create a single list with 5 0's and 5 1's randomly ordered
    def create_balanced_list():
        balanced_list = [0] * 5 + [1] * 5
        random.shuffle(balanced_list)
        return balanced_list

    # Create a list of 30 such lists
    blind_test_choice_matrix = [create_balanced_list() for _ in range(30)]

    # The one used in the report
    # blind_test_choice_matrix = [[0, 1, 1, 1, 0, 1, 0, 1, 0, 0],
    #                             [0, 1, 0, 0, 1, 1, 1, 0, 1, 0],
    #                             [1, 0, 1, 0, 1, 1, 0, 0, 0, 1],
    #                             [0, 0, 1, 0, 1, 0, 1, 0, 1, 1],
    #                             [0, 0, 0, 1, 0, 1, 1, 1, 1, 0],
    #                             [1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
    #                             [1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
    #                             [0, 1, 0, 0, 1, 1, 0, 1, 1, 0],
    #                             [1, 1, 1, 0, 0, 0, 1, 0, 1, 0],
    #                             [0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
    #                             [0, 0, 1, 1, 1, 0, 1, 0, 0, 1],
    #                             [0, 1, 0, 1, 0, 1, 0, 0, 1, 1],
    #                             [0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
    #                             [1, 1, 1, 1, 0, 0, 1, 0, 0, 0],
    #                             [0, 0, 1, 1, 0, 1, 0, 1, 1, 0],
    #                             [1, 0, 1, 0, 1, 1, 0, 0, 0, 1],
    #                             [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
    #                             [0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    #                             [0, 1, 1, 0, 1, 1, 0, 1, 0, 0],
    #                             [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    #                             [0, 0, 1, 0, 1, 1, 1, 0, 0, 1],
    #                             [1, 1, 0, 0, 1, 0, 1, 1, 0, 0],
    #                             [0, 0, 1, 0, 0, 0, 1, 1, 1, 1],
    #                             [0, 1, 1, 0, 1, 1, 0, 0, 0, 1],
    #                             [1, 0, 1, 0, 0, 0, 1, 0, 1, 1],
    #                             [1, 1, 0, 1, 0, 1, 0, 1, 0, 0],
    #                             [0, 0, 1, 0, 1, 1, 1, 0, 0, 1],
    #                             [0, 0, 0, 1, 1, 0, 0, 1, 1, 1],
    #                             [1, 1, 0, 0, 1, 1, 0, 0, 0, 1],
    #                             [0, 1, 0, 1, 1, 0, 1, 1, 0, 0]]

    with open('TransformedRow_to_arXivID.pkl', 'rb') as f:
        reversed_mapping_dict = pickle.load(f)
    DATASET = np.array(pd.read_pickle('transformed_dataset.pkl'))[:20000]
    train_size = int(len(DATASET) * 0.9)
    
    # define 30 random contexts
    random_idx_30 = np.random.choice(range(train_size, len(DATASET)), 5, replace=False)

    # Define variables
    num_features = 768
    batch_size = 64
    num_workers = 6
    pin_memory = True if device.type == 'cuda' else False
    max_epochs = 50
    lr = 1e-2 # Must be adjusted to the actual learning rate, even when loading a pretrained model where the learning rate had decayed
    margin = 0.6
    top_k = 5
    
    time_file_save = '2024-06-15_12-59-14'

    d_func = Distance.weighted_euclidean

    test_set = arXivDataset(DATASET[train_size:], 
                            train=False, 
                            device=device)
    test_loader = DataLoader(test_set, 
                             batch_size=1, 
                             shuffle=False,
                             num_workers=0) # Batch size must be 1 for top-k accuracy
    
    # Define model 1
    model1 = TripletModel(num_features, alpha=margin, d_func=d_func, device=device).to(device)
    model1.load_state_dict(torch.load(f'Training_Variables/triplet_model_{time_file_save}.pth'))
    model1.eval()
    A1 = torch.diag(torch.exp(model1.W))

    # Define model 2
    model2 = TripletModel(num_features, alpha=margin, d_func=d_func, device=device).to(device) # Margin is irrelevant for this test
    A2 = torch.diag(torch.exp(model2.W))

    targets = torch.tensor(np.unique(DATASET[:,1], axis=0), device=device)

    result_table = [[[0, 0, 0, 0, 0, 0] for j in range(10)] for _ in range(30)]

    # Compute the top-5 arXiv-IDs for each context for both models
    with torch.no_grad():
        for i, random_idx in enumerate(random_idx_30):
            idx = i+1
            context_transformed = DATASET[random_idx][0]
            context_item = reversed_mapping_dict[tuple(context_transformed)]
            context_arXivID = context_item[0]
            context = context_item[1]
            anchor = torch.tensor(context_transformed, device=device).unsqueeze(0)
            model1_top_5 = [[],[]]
            model2_top_5 = [[],[]]
            for article_transformed in tqdm(targets):
                D1 = float(model1.d_func(anchor, article_transformed, A1))
                D2 = float(model2.d_func(anchor, article_transformed, A2))

                if len(model1_top_5[0]) >= 5:
                    max_dist_1 = [np.argmax(model1_top_5[1]), np.max(model1_top_5[1])]
                    arXivID_new = reversed_mapping_dict[tuple(article_transformed.detach().numpy())]
                    if D1 < max_dist_1[1] and arXivID_new not in model1_top_5[0]:
                        model1_top_5[0][max_dist_1[0]] = arXivID_new
                        model1_top_5[1][max_dist_1[0]] = D1
                else:
                    model1_top_5[0].append(reversed_mapping_dict[tuple(article_transformed.detach().numpy())])
                    model1_top_5[1].append(D1)

                if len(model2_top_5[0]) >= 5:
                    max_dist_2 = [np.argmax(model2_top_5[1]), np.max(model2_top_5[1])]
                    arXivID_new = reversed_mapping_dict[tuple(article_transformed.detach().numpy())]
                    if D1 < max_dist_1[1] and arXivID_new not in model2_top_5[0]:
                        model2_top_5[0][max_dist_2[0]] = reversed_mapping_dict[tuple(article_transformed.detach().numpy())]
                        model2_top_5[1][max_dist_2[0]] = D2
                else:
                    model2_top_5[0].append(reversed_mapping_dict[tuple(article_transformed.detach().numpy())])
                    model2_top_5[1].append(D2)
                
            print("\nFor context number: " + str(idx))
            print("The order is as follows: ")
            model1_counter = 0
            model2_counter = 0
            for j_idx, j in enumerate(blind_test_choice_matrix[i]):
                if j == 0:
                    arXivID = model1_top_5[0][model1_counter]
                    model1_counter += 1
                else:
                    arXivID = model2_top_5[0][model2_counter]
                    model2_counter += 1
                
                result_table[i][j_idx][0] = idx                                 # Context number
                result_table[i][j_idx][1] = j_idx                               # Abstract number
                result_table[i][j_idx][2] = j+1                                 # (Confidential) Model number
                result_table[i][j_idx][3] = arXivID                             # predicted target arXiv-ID
                result_table[i][j_idx][4] = context_arXivID                     # origin arXiv-ID
                result_table[i][j_idx][5] = context                             # context
                print(f"{j_idx}: Model {j+1}, arXiv-ID: " + arXivID)
    
    # Save the result table
    pickle.dump(result_table, open(os.path.join("Autocite","result_table_stat_test.pkl"), "wb"))

    # Return top 5 arXiv-IDS for the popularity model
    print(f"Popularity model top-{top_k} suggestions:")
    popularity_model = PopularityModel(DATASET[:train_size], top_k)
    for transformed_arXivID in popularity_model.return_arxiv_ids().keys():
        print(reversed_mapping_dict[transformed_arXivID])