import pandas as pd
import random
import pickle
import json

from parseBBL2 import remove_latex_commands


def randomizeKaggleDBjson():
    # Load the Kaggle dataset
    with open("Kaggle_Dataset.json", 'r') as file:
        KaggleDB = file.readlines()
        random.seed(42)
        random.shuffle(KaggleDB)
    
    LatexCleanedKaggleDB = []
    for i in range(len(KaggleDB)):
        row = KaggleDB[i]
        if i == len(KaggleDB) - 1:
            dictionary = json.loads(row)
        else:
            dictionary = json.loads(row[:-1])
        arxiv_id = dictionary['id']
        title = remove_latex_commands(dictionary['title'])
        authors = remove_latex_commands(dictionary['authors'])
        publish_date = dictionary['publish_time']
        info = authors + " " + title + " " + publish_date
        LatexCleanedKaggleDB.append({"arxiv_id": arxiv_id, "info": info, "title": title, "authors": authors, "publish_date": publish_date})
    try:
        with open("Randomized_Kaggle_Dataset.json", 'w') as file:
            json.dump(KaggleDB, file)
    except Exception as e:
        print(f"Failed to save the randomized dataset with error: {e}")


def randomizeKaggleDBpickle():
    # Load the Kaggle dataset
    with open("Kaggle_Dataset.json", 'r') as file:
        KaggleDB = file.readlines()
        random.seed(42)
        random.shuffle(KaggleDB)
    
    # Randomize the dataset
    
    # Save the randomized dataset
    try:
        with open("Randomized_Kaggle_Dataset.pkl", 'wb') as file:
            pickle.dump(KaggleDB, file)
    except Exception as e:
        print(f"Failed to save the randomized dataset with error: {e}")

def read_randomizedKaggleDBpickle():
    # Load the randomized dataset
    try:
        with open("Randomized_Kaggle_Dataset.pkl", 'rb') as file:
            KaggleDB = pickle.load(file)
    except Exception as e:
        print(f"Failed to load the randomized dataset with error: {e}")
        KaggleDB = []
    print(KaggleDB[:50])
    print("")
    print(len(KaggleDB))
    pass

#randomizeKaggleDB()
#read_randomizedKaggleDB()
randomizeKaggleDBjson()