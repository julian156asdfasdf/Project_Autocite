import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data_Processing_Pipeline.step3_processing import ACCENT_CONVERTER # Import useful function from the step3 file

from tqdm.auto import tqdm
import json

# Create the arXivID to Abstract dictionary
def create_arXivID_to_Abstract_DB(filepath = "Kaggle_Dataset.json", categories = []):
    # Read the Kaggle dataset
    with open(filepath, 'r') as file:
        KaggleDB = file.readlines()
    
    # Loop through rows and extract the arXivID and Abstract and store in a dictionary
    ID_to_Abstract = {}
    rows_total = len(KaggleDB)
    for i in tqdm(range(rows_total), desc = "Creating arXivID to Abstract DB"):
        row = KaggleDB[i]
        # Load the json string into a dictionary
        if i == rows_total - 1:
            dictionary = json.loads(row)
        else:
            dictionary = json.loads(row[:-1])
        # Extracts only the rows with the specified category (len(categories)==0 means all categories are included)
        if len(categories) == 0:
            ID_to_Abstract.update({dictionary['id'] : ACCENT_CONVERTER(dictionary['abstract'])})
        else:
            for category in categories:
                if category in dictionary['categories']: # others are 'math':650k, 'cs':850k, 'stat', 'eess', 'physics':250k, 'q-bio', 'q-fin', 'quant-ph', 'hep':350k
                    ID_to_Abstract.update({dictionary['id'] : ACCENT_CONVERTER(dictionary['abstract'])})
                    continue
    
    category_string = ""
    if len(categories) > 0:
        category_string = "_Subset_" + "_".join(categories)
     # Writing the dictionary to a new json file    
    new_filepath = os.path.join("Autocite", "arXivIDs_to_Abstract" + category_string + ".json")
    with open(new_filepath, 'w') as file:
        json.dump(ID_to_Abstract, file)


if __name__ == '__main__':
    from RandomizeKaggleDB import read_json_DB
    
    categories=['physics']
    # create_arXivID_to_Abstract_DB(filepath = "Kaggle_Dataset.json", categories = categories)
    category_string = ""
    if len(categories) > 0:
        category_string = "_Subset_" + "_".join(categories)
        # Writing the randomized dataset to a new json file    
    new_filepath = os.path.join("Autocite", "arXivIDs_to_Abstract" + category_string + ".json")
    AtoA = read_json_DB(new_filepath)