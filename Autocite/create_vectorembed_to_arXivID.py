# Create a dictionary that maps from transformed datapoint to arXivID (and if possible context)
import pickle
from tqdm.auto import tqdm
import os

# Load data
with open('transformed_dataset.pkl', 'rb') as file:
    dataset_transformed = pickle.load(file)
with open('dataset.pkl', 'rb') as file:
    dataset = pickle.load(file)

assert len(dataset) == len(dataset_transformed)

# Create a dictionary that maps from transformed datapoint to arXivID (and if possible context)
TransformedRow_to_arXivID = {}
for i in tqdm(range(len(dataset_transformed))):
    transformed_row = dataset_transformed[i]
    context_transformed = transformed_row[0]
    abstract_transformed = transformed_row[1]
    TransformedRow_to_arXivID.update({tuple(context_transformed) : [dataset[i][0], dataset[i][2]]})
    TransformedRow_to_arXivID.update({tuple(abstract_transformed) : dataset[i][1]})

# Save the dictionary
pickle.dump(TransformedRow_to_arXivID, open(os.path.join("Autocite","TransformedRow_to_arXivID.pkl"), "wb"))