import random
import json
import re
from tqdm.auto import tqdm

from parseBBL2 import remove_latex_commands


def randomizeKaggleDB(filepath = "Kaggle_Dataset.json"):
    # Load the Kaggle dataset
    with open(filepath, 'r') as file:
        KaggleDB = file.readlines()
        random.seed(42)
        random.shuffle(KaggleDB)
    
    LatexCleanedKaggleDB = []
    rows_total = len(KaggleDB)
    for i in tqdm(range(rows_total)):
        row = KaggleDB[i]
        # Load the json string into a dictionary
        if i == rows_total - 1:
            dictionary = json.loads(row)
        else:
            dictionary = json.loads(row[:-1])
        
        # Extracts only the rows with the specified category
        if "physics" in dictionary['categories']: # others are 'math':650k, 'cs':850k, 'stat', 'eess', 'physics':250k, 'q-bio', 'q-fin', 'quant-ph', 'hep':350k
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
            LatexCleanedKaggleDB.append({"arxiv_id": arxiv_id, "info": info, "title": title, "authors": authors})

     # Writing the randomized dataset to a new json file       
    rows = len(LatexCleanedKaggleDB)
    with open("Randomized_Kaggle_Dataset_Subset_Physics.json", 'w') as file:
        for i in tqdm(range(rows)):
            json.dump(LatexCleanedKaggleDB[i], file)
            if i != rows-1:
                file.write('\n')  # Add a newline after each dictionary


def read_json_DB(filepath = "Randomized_Kaggle_Dataset_Subset_Physics.json"):
    # Load the json file into a list of dictionaries
    #print("Loading json Dataset... (Can take a minute or two)")
    KaggleDB = []
    try:
        with open(filepath, 'r') as file:
            KaggleDB_raw = file.readlines()
            rows = len(KaggleDB_raw)
            for i in tqdm(range(rows)):
                if i == rows - 1:
                    dictionary = json.loads(KaggleDB_raw[i])
                else:
                    dictionary = json.loads(KaggleDB_raw[i][:-1])
                KaggleDB.append(dictionary)
    except Exception as e:
        print(f"Failed to load the randomized dataset with error: {e}")
        KaggleDB = []
    return KaggleDB

def read_and_shuffle_KaggleDB(filepath = "Kaggle_Dataset.json"):
    # Load the json file into a list of dictionaries
    print("Loading Kaggle Dataset... (Can take a minute or two)")
    KaggleDB = []
    try:
        with open("Kaggle_Dataset.json", 'r') as file:
            KaggleDB_raw = file.readlines()
            random.seed(42)
            random.shuffle(KaggleDB_raw)
            rows = len(KaggleDB_raw)
            for i in tqdm(range(rows)):
                if i == rows - 1:
                    dictionary = json.loads(KaggleDB_raw[i])
                else:
                    dictionary = json.loads(KaggleDB_raw[i][:-1])
                KaggleDB.append(dictionary)
    except Exception as e:
        print(f"Failed to load the randomized dataset with error: {e}")
        KaggleDB = []
    return KaggleDB

if __name__ == '__main__':
    randomizeKaggleDB()
    KaggleDB = read_json_DB()
    new_author = remove_latex_commands(KaggleDB[0]['authors'])
    pass