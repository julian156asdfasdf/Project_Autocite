import os
from pathlib import Path
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pandas as pd
import numpy as np
import shutil
from main import KAGGLEDB, ARXIV_IDS
from tqdm.auto import tqdm
import time
import random

class step0_processing:
    def __init__(self, target_name="Step_0",start_idx=0, window_size=100, end_idx=100):
        self.target = target_name
        self.tar_links = []
        self.arxiv_papers = None
        self.round_number = 1
        self.start_idx = start_idx
        self.window_size = window_size
        self.end_idx = end_idx
        self.rounds = int(np.ceil((end_idx-start_idx)/window_size))


    def create_target_folder(self) -> None:
        """
        Deletes the current Step_0 folder (if it exists) and creates a new empty one.
        """
        shutil.rmtree(self.target, ignore_errors=True)
        os.makedirs(self.target, exist_ok=False)

    def get_tar_links(self, start_id: int=0) -> None:
        """
        Gets links for .csv.xz files using ArXiV ID from Kaggle Dataset. Only gets the links within the sliding window.

        Arguments:
            start_id (int): Start index of the sliding window.

        Returns:
            None
        """
        def get_eprint_link(paper: str) -> str:
            return f'http://export.arxiv.org/e-print/{paper}'
        
        # Get the ArXiV IDs for the current sliding window and the corresponding links
        all_article_ids = ARXIV_IDS[start_id:min(start_id+self.window_size, self.end_idx)]
        arxiv_papers = pd.DataFrame(all_article_ids, columns=['arxiv_id'])
        links = list(arxiv_papers['arxiv_id'].apply(get_eprint_link))

        self.links = links
        self.arxiv_papers = arxiv_papers


    def download_paper(self, link: str, paper_id: str) -> str:
        """
        Downloads a paper from the given URL and saves it to the given path.

        Arguments:
            link (str): URL of the paper to download.
            paper_id (str): ArXiV ID of the paper.

        Returns:
            str: ArXiV ID of the downloaded paper.
        """
        paper_id_edit = paper_id.replace("/", "_slash") # Replace / with _slash to avoid errors
        file_path = Path("./"+self.target) / f"{paper_id_edit}.tar"
        # Try to download paper
        try: 
            response = requests.get(link, timeout=40)
        except TimeoutError:
            print(f"Timeout on {paper_id}")

        # Save the paper to the given path
        if response.status_code == 200:
            try:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                print(f"Failed to download {link} with error: {e}")
        else:
            print(f"Failed to download {link}")
        return paper_id

    def download_tar_files(self) -> None:
        """
        Downloads the papers as .tar files into the Step_0 directory.
        """
        files_left = self.window_size

        with ThreadPoolExecutor() as executor:
            futures = []
            for idx, link in enumerate(self.links, start=1):
                paper_id = self.arxiv_papers['arxiv_id'][idx - 1] # Adjust index if necessary
                futures.append(executor.submit(self.download_paper, link, paper_id))
                time.sleep(3 + random.gauss(7,5)) # waits 10 seconds before requesting the next paper

            for future in tqdm(as_completed(futures), desc="Downloading papers", total=self.window_size):
                result = future.result()
                files_left -= 1
                # print(f'Progress: {self.window_size-files_left}/{self.window_size}. File: {result}')

if __name__ == "__main__":    
    from RandomizeKaggleDB import read_json_DB

    KAGGLEDB = read_json_DB(filepath="Randomized_Kaggle_Dataset_Subset_physics.json")
    ARXIV_IDS = list(KAGGLEDB.keys())

    step_0 = step0_processing(target_name="Step_0", start_idx=0, window_size=100, end_idx=100)

    # Create sliding window for step 0-2
    for i in range(step_0.rounds):
    # step 0 Download tar files
        print("Starting round " + str(step_0.round_number) + "...")
        print("Downloading tar files...")
        step_0.round_number += 1
        step_0.create_target_folder()
        step_0.get_tar_links(start_id=step_0.window_size*i+step_0.start_idx)
        step_0.download_tar_files()
        print("Downloaded tar files successfully.")