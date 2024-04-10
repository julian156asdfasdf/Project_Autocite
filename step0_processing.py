import os
from pathlib import Path
import requests
import os
import pandas as pd
import random
import numpy as np
import json
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
        
        #with open(self.Kaggle_dataset_path, 'r') as file:
            #data = file.readlines()
        all_articles = []
        for article in self.KaggleDB[start_id:min(start_id+self.window_size, self.end_idx)]:
            all_articles.append(json.loads(article)['id'])
        arxiv_papers = pd.DataFrame(all_articles, columns=['ArXiV ID'])
        links = list(arxiv_papers['ArXiV ID'].apply(get_eprint_link))

        self.links = links
        self.arxiv_papers = arxiv_papers

    def get_docs(self):
        """
        Uses the list of links from get_tar_links to download the papers as .tar files into the Step_0 directory.
        """
        for idx, link in enumerate(self.links, start=1):  # Just an example with `.tail()`, remove it to download all
            paper_id = self.arxiv_papers['ArXiV ID'][idx - 1]  # Adjust index if necessary
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


    #def get_tar_links(self):
    #    """
    #    Gets links from downloaded arxiv-papers.csv.xz file and shuffles them with seed=42.
    #    """
    #    def get_eprint_link(paper):
    #        return f'http://export.arxiv.org/e-print/{paper.arxiv_id}'

    #    V1_URL = 'https://github.com/paperswithcode/axcell/releases/download/v1.0/'
    #    ARXIV_PAPERS_URL = V1_URL + 'arxiv-papers.csv.xz'
    #    arxiv_papers = pd.read_csv(ARXIV_PAPERS_URL)
    #    arxiv_papers = arxiv_papers[arxiv_papers['status']=='success']
    #    links = list(arxiv_papers.apply(get_eprint_link, axis=1))
        
    #    random.seed(42)
    #    random.shuffle(links)
        
    #    self.links = links
    #    self.arxiv_papers = arxiv_papers


    # Specify the directory where you want to save the papers

    #def get_docs(self, k=2):
    #    """
    #    Takes a directory and a list of links to papers and downloads the papers as .tar files to the directory.
    #    """
    #    print(self.links)
    #    for idx, link in enumerate(self.links[:k], start=1):  # Just an example with `.tail()`, remove it to download all
    #        paper_id = self.arxiv_papers.iloc[idx - 1].arxiv_id  # Adjust index if necessary
    #        file_path = Path("./"+self.target) / f"{paper_id}.tar"
    #        print(f"Downloading {paper_id} to {file_path}... " + str(idx) + "/" + str(k))

            # Download a paper from the given URL and save it to the given path.
    #        response = requests.get(link)
    #        if response.status_code == 200:
    #            with open(file_path, 'wb') as f:
    #                f.write(response.content)
    #        else:
    #            print(f"Failed to download {link}")


if __name__ == "__main__":
    print("Loading Kaggle Dataset... (Should take around 30)")
    with open("Kaggle_Dataset.json", 'r') as file:
        KaggleDB = file.readlines()
        random.seed(42)
        random.shuffle(KaggleDB)
    
    process = step0_processing(KaggleDB = KaggleDB, target_name="Step_0", start_idx=57, window_size=10, end_idx=70)
    process.create_target_folder()
    
    for i in range(process.rounds):
        process.round_number += 1
        process.get_tar_links_test(start_id=process.window_size*i+process.start_idx)
        process.get_docs_test()