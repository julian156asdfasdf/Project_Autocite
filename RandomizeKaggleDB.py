import pandas as pd
import random
import pickle



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
            dictionary = eval(row)
        else:
            dictionary = eval(row[:-1])
        row_list = "{"
        for key, value in dictionary.items():
            row_list += "\"" + key + "\":\"" + value + "\","
            #LatexCleanedKaggleDB.append(row.replace("\n", ""))


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