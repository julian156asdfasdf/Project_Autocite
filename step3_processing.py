from pathlib import Path
import pandas as pd
import os
from collections import defaultdict
import re
from RandomizeKaggleDB import read_json_DB

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

    def regex_on_author(self, author):
        # helper class that does regex on an author and returns
        if not author:
            return author
        if len(author)>6:
            if author[:5] == ' and ':
                author = author[5:]
        for auth in author.split(' and '):
            match = re.search(r'\w+$', auth)
            if match:
                author = match.group().lower()
        return author
    
    def subdivide_by_author(self):
        # method 1, each key is name of author
        for _, row in self.kaggle_db.iterrows():
            arxiv_id = row['arxiv_id']
            authors = row['authors']
            authors_list = authors.split(',')
            for author in authors_list:
                author = self.regex_on_author(author)
                self.author_db[author].add(arxiv_id)

        #i = 50
        #succesfull_iterations = []
        #for author in self.author_db.keys():
        #    for other_author in self.author_db.keys():
        #        if str(author) in other_author:
        #            self.author_db[author] = self.author_db[author].union(self.author_db[other_author])
        #            succesfull_iterations.append((author, other_author))
        #            i+=10
        return self.author_db
    
        # Problem is that the authors keys are not good example
        # the following are the different keys:  " 'Eric H'ebrard"," 'Eric Herbert (LIED)"," 'Eric Herbert and Pierre-Philippe Cortet",
        # Can be seen when doing sorted(self.author_db)
        # possible solution, make it a set instead of list, and if two keys contain the same, then union between long and short key on the short one
        # do not collapse the long one as it can run multiple times
        # possible implementation above, problem is that it runs incredibly slow

        #
        #for dir in os.listdir(self.file_dir):
        #    author_db = defaultdict()
        #    path = Path(os.path.join(self.file_dir,dir, 'references.json'))
        #    ref_json = pd.read_json(path, lines=True)
        #    for author in ref_json['author_ln']:
        #        print(author)

    
       

    def find_authors_refs(self):
        """ 
        This function does not work yet
        Takes all of the references.json, and sees how many can be found in auther_db
        """
        N_total, N_hits = 0, 0
        None_articles = []
        N_none = 0
        for dir in os.listdir(self.file_dir):
            path = Path(os.path.join(self.file_dir,dir, 'references.json'))
            ref_json = read_json_DB(path)
            for key in ref_json[0]:
                author = ref_json[0][key]['author_ln']
                author_regex = self.regex_on_author(author)
                N_total+=1

                if not author:
                    N_none+=1
                    None_articles.append(dir)

                if author_regex in self.author_db:
                    N_hits += 1
                else:
                    print(author_regex, author)
        return N_total, N_hits, N_none, set(None_articles)




if __name__ == "__main__":
    kaggle_db_path = Path('Randomized_Kaggle_Dataset_Subset_Physics.json')
    processing = proccessing_3(kaggle_db_path, Path('Step_2'))
    authors = processing.subdivide_by_author()
    N_total, N_hits, N_none, None_articles = processing.find_authors_refs()

