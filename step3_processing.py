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

    def subdivide_by_author(self):
        # method 1, each key is name of author
        author_db = defaultdict(list)
        i = 0
        for authors in self.kaggle_db['authors']:
            for author in authors.split(','):
                author_db[author].append(self.kaggle_db[self.kaggle_db['authors']==authors])
                
            i+=1
            if i == 1000:
                break
        return author_db

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
