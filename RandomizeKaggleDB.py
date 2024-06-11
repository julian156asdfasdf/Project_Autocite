import random
import json
import re
from tqdm.auto import tqdm

from parseBBL2 import remove_latex_commands

# Process the dictionary
def extract_info_title_authors(dictionary):
    # Extract the relevant information
    arxiv_id = dictionary['id']
    title = remove_latex_commands(dictionary['title']) # Remove all latex commands from the title
    authors = remove_latex_commands(dictionary['authors']) # Remove all latex commands from the authors
    authors = re.sub(r'\s+', ' ', authors)
    update_date = dictionary['update_date']
    info = authors + ". " + re.sub(r'\s+', ' ', title) + ". " + update_date
    # Remove all equations from title
    idx_dollar = 0
    while "$" in title[idx_dollar:]:
        idx_dollar = title.find("$", idx_dollar)
        if idx_dollar == len(title)-1:
            break
        if title[idx_dollar-1] == "\\":
            idx_dollar = idx_dollar+1
        elif title[idx_dollar+1] == "$":
            end_dollar_idx = title.find("$", idx_dollar+2)+1
            if end_dollar_idx == 0:
                end_dollar_idx = len(title)
            title = title[:idx_dollar] + title[end_dollar_idx+1:]
        else:
            end_dollar_idx = title.find("$", idx_dollar+1)
            if end_dollar_idx == -1:
                end_dollar_idx = len(title)
            title = title[:idx_dollar] + title[end_dollar_idx+1:]
    
    title = re.sub(r'\s+', ' ', title).strip() # Remove all extra whitespaces
    # Append the information to the new dataset as a dictionary
    return {arxiv_id: {"info": info, "title": title, "authors": authors}}

def randomizeKaggleDB(filepath = "Kaggle_Dataset.json", categories = []):
    # Load the Kaggle dataset
    with open(filepath, 'r') as file:
        KaggleDB = file.readlines()
        random.seed(42)
        random.shuffle(KaggleDB)
    
    LatexCleanedKaggleDB = {}
    rows_total = len(KaggleDB)
    for i in tqdm(range(rows_total)):
        row = KaggleDB[i]
        # Load the json string into a dictionary
        if i == rows_total - 1:
            dictionary = json.loads(row)
        else:
            dictionary = json.loads(row[:-1])

        # Extracts only the rows with the specified category
        if len(categories) == 0:
            LatexCleanedKaggleDB.update(extract_info_title_authors(dictionary))
        else:
            for category in categories:
                if category in dictionary['categories']: # others are 'math':650k, 'cs':850k, 'stat', 'eess', 'physics':250k, 'q-bio', 'q-fin', 'quant-ph', 'hep':350k
                    LatexCleanedKaggleDB.update(extract_info_title_authors(dictionary))
                    continue
    
    category_string = ""
    if len(categories) > 0:
        category_string = "_Subset_" + "_".join(categories)
     # Writing the randomized dataset to a new json file    
    new_filepath = "Randomized_Kaggle_Dataset" + category_string + ".json"
    with open(new_filepath, 'w') as file:
        json.dump(LatexCleanedKaggleDB, file)

def read_json_DB(filepath = "Randomized_Kaggle_Dataset_Subset_Physics.json"):
    try:
        with open(filepath, 'r') as file:
            KaggleDB = json.load(file)
    except Exception as e:
        print(f"Failed to load the randomized dataset with error: {e}")
        KaggleDB = {}
    return KaggleDB

if __name__ == '__main__':
    categories = ['physics']
    randomizeKaggleDB(categories=categories)
    KaggleDB = read_json_DB(filepath="Randomized_Kaggle_Dataset" + ("_Subset_" + "_".join(categories) if len(categories) > 0 else "") + ".json")
    pass