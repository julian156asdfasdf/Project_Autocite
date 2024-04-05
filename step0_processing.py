import os
from pathlib import Path
import requests
import os
import pandas as pd
import random


class step0_processing:
    def __init__(self, target_name):
        self.target = target_name
        self.tar_links = []
        self.arxiv_papers = None


    def create_target_folder(self):
        if os.path.exists(self.target) == False:
            os.makedirs(self.target, exist_ok=True)

    def get_tar_links(self):
        """
        Gets links from downloaded arxiv-papers.csv.xz file and shuffles them with seed=42.
        """
        def get_eprint_link(paper):
            return f'http://export.arxiv.org/e-print/{paper.arxiv_id}'
        V1_URL = 'https://github.com/paperswithcode/axcell/releases/download/v1.0/'
        ARXIV_PAPERS_URL = V1_URL + 'arxiv-papers.csv.xz'
        arxiv_papers = pd.read_csv(ARXIV_PAPERS_URL)
        arxiv_papers = arxiv_papers[arxiv_papers['status']=='success']

        links = list(arxiv_papers.apply(get_eprint_link, axis=1))
        random.seed(42)
        random.shuffle(links)
        
        self.links = links
        self.arxiv_papers = arxiv_papers


    # Specify the directory where you want to save the papers

    def get_docs(self, k=2):
        """
        Takes a directory and a list of links to papers and downloads the papers as .tar files to the directory.
        """

        for idx, link in enumerate(self.links[:k], start=1):  # Just an example with `.tail()`, remove it to download all
            paper_id = self.arxiv_papers.iloc[idx - 1].arxiv_id  # Adjust index if necessary
            file_path = Path("./"+self.target) / f"{paper_id}.tar"
            print(f"Downloading {paper_id} to {file_path}... " + str(idx) + "/" + str(k))

            # Download a paper from the given URL and save it to the given path.
            response = requests.get(link)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            else:
                print(f"Failed to download {link}")


if __name__ == "__main__":
    process = step0_processing("Papers")
    process.create_target_folder()
    process.get_tar_links()
    process.get_docs(k=2)