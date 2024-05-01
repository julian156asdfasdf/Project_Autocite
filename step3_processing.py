from pathlib import Path
import pandas as pd
import os
from collections import defaultdict

# Match step 2 references.tex titles with kaggle db,
# Extract ArxivID and abstract
# Create JSON {latexID: [ArxivID, ]}
# download following https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download
# and call json arxiv_metadata



class proccessing_3:
    def __init__(self, database_url, file_dir):
        self.kaggle_db = pd.read_json(database_url, lines=True)
        self.file_dir = file_dir
        self.author_db = defaultdict(list)
    def subdivide_by_author(self):
        # method 1, each key is name of author
        i = 0
        for index, row in self.kaggle_db.iterrows():
            authors = row['authors']
            arxiv_id = row['arxiv_id']
            for author in authors.split(','):
                self.author_db[author].append(arxiv_id)
                i+=1
            
        return self.author_db

        #
        #for dir in os.listdir(self.file_dir):
        #    author_db = defaultdict()
        #    path = Path(os.path.join(self.file_dir,dir, 'references.json'))
        #    ref_json = pd.read_json(path, lines=True)
        #    for author in ref_json['author_ln']:
        #        print(author)




if __name__ == "__main__":
    processing = proccessing_3(Path('Randomized_Kaggle_Dataset_Subset_Physics.json'), Path('Step_2'))
    authors = processing.subdivide_by_author()
