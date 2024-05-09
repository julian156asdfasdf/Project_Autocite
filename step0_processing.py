import os
from pathlib import Path
import requests
import os
import pandas as pd
import numpy as np
import shutil


class step0_processing:
    def __init__(self, KaggleDB, target_name="Step_0",start_idx=0, window_size=100, end_idx=100):
        self.target = target_name
        self.tar_links = []
        self.arxiv_papers = None
        self.round_number = 0
        self.start_idx = start_idx
        self.window_size = window_size
        self.end_idx = end_idx
        self.rounds = int(np.ceil((end_idx-start_idx)/window_size))
        self.KaggleDB = KaggleDB


    def create_target_folder(self):
        """
        Deletes the current Step_0 folder and creates a new empty one.
        """
        shutil.rmtree(self.target, ignore_errors=True)
        os.makedirs(self.target, exist_ok=False)

    def get_tar_links(self, start_id=0):
        """
        Gets links for .csv.xz files using ArXiV ID from Kaggle Dataset. Only gets the links within the sliding window.
        """
        def get_eprint_link(paper):
            return f'http://export.arxiv.org/e-print/{paper}'
        
        all_articles = []
        for article in self.KaggleDB[start_id:min(start_id+self.window_size, self.end_idx)]:
            all_articles.append(article['arxiv_id'])
        arxiv_papers = pd.DataFrame(all_articles, columns=['arxiv_id'])
        links = list(arxiv_papers['arxiv_id'].apply(get_eprint_link))

        self.links = links
        self.arxiv_papers = arxiv_papers

    def get_docs(self):
        """
        Uses the list of links from get_tar_links to download the papers as .tar files into the Step_0 directory.
        """
        for idx, link in enumerate(self.links, start=1):  # Just an example with `.tail()`, remove it to download all
            paper_id = self.arxiv_papers['arxiv_id'][idx - 1]  # Adjust index if necessary
            paper_id_edit = paper_id.replace("/", "_slash")
            file_path = Path("./"+self.target) / f"{paper_id_edit}.tar"
            print(f"Downloading {paper_id} to {file_path}... " + str(idx) + "/" + str(self.window_size) + " in round: " + str(self.round_number))

            # Download a paper from the given URL and save it to the given path.
            response = requests.get(link)
            if response.status_code == 200:
                try:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                except Exception as e:
                    print(f"Failed to download {link} with error: {e}")
            else:
                print(f"Failed to download {link}")


if __name__ == "__main__":
    from RandomizeKaggleDB import randomizeKaggleDB, read_and_shuffle_KaggleDB, read_json_DB
    #read_KaggleDB_Subset
    # KaggleDB = read_KaggleDB_Subset(filepath="Randomized_Kaggle_Dataset_Subset_Physics.json")
    KaggleDB = read_json_DB(filepath="Randomized_Kaggle_Dataset_Subset_Physics.json")
    
    step_0 = step0_processing(KaggleDB = KaggleDB, target_name="Step_0", start_idx=0, window_size=10, end_idx=100)

    # Create sliding window for step 0-2
    for i in range(step_0.rounds):
    # step 0 Download tar files
        print("Starting round " + str(step_0.round_number) + "...")
        print("Downloading tar files...")
        step_0.round_number += 1
        step_0.create_target_folder()
        step_0.get_tar_links(start_id=step_0.window_size*i+step_0.start_idx)
        step_0.get_docs()
        print("Downloaded tar files successfully.")