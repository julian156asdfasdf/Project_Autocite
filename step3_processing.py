from pathlib import Path
import pandas as pd
import os
from collections import defaultdict
import re

# Match step 2 references.tex titles with kaggle db,
# Extract ArxivID and abstract
# Create JSON {latexID: [ArxivID, ]}
# download following https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download
# and call json arxiv_metadata



class proccessing_3:
    def __init__(self, database_url, file_dir):
        self.kaggle_db = pd.read_json(database_url, lines=True)
        self.file_dir = file_dir
        self.author_db = defaultdict(set)
    def subdivide_by_author(self):
        """ 
        Argument: Kaggle_database
        Output: Author_db a dictionary
        This function takes the kaggle database, and makes a dictionary
        where each key is the name of an author
        and the value is a list of every article they have made
        where the value in the list is the arxiv id 
        """

        for index, row in self.kaggle_db.iterrows():
            arxiv_id = row['arxiv_id']
            authors = row['authors']
            authors_list = authors.split(',')
            for author in authors_list:
                if author[:4] == ' and ':
                    author = author[5:]
                for auth in author.split(' and '):
                    match = re.search(r'\w+$', auth)
                    if match:
                        self.author_db[match.group()].add(arxiv_id)

        #i = 50
        #succesfull_iterations = []
        #for author in self.author_db.keys():
        #    for other_author in self.author_db.keys():
        #        if str(author) in other_author:
        #            self.author_db[author] = self.author_db[author].union(self.author_db[other_author])
        #            succesfull_iterations.append((author, other_author))
        #            i+=10
        return self.author_db
    
       

    def find_authors_refs(self):
        """ 
        This function does not work yet
        Takes all of the references.json, and sees how many can be found in auther_db
        """
        N_total, N_hits = 0, 0
        for dir in os.listdir(self.file_dir):
            path = Path(os.path.join(self.file_dir,dir, 'references.json'))
            ref_json = pd.read_json(path, lines=True)
            for author in ref_json['author_ln']:
                N_total+=1
                if author in self.author_db:
                    N_hits += 1
        return N_total, N_hits




if __name__ == "__main__":
    processing = proccessing_3(Path('Randomized_Kaggle_Dataset_Subset_Physics.json'), Path('Step_2'))
    authors = processing.subdivide_by_author()
