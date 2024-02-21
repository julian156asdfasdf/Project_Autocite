# %%
#https://huggingface.co/datasets/ArtifactAI/arxiv_s2orc_parsed?row=0 
# Har fuld text

#https://huggingface.co/datasets/jamescalam/ai-arxiv-chunked?row=0 hhar reference liste

# https://huggingface.co/datasets/jamescalam/ai-arxiv?row=1 Lille subset i test fase med referenceliste

# udgangspunkt https://www.kaggle.com/datasets/Cornell-University/arxiv

# %%
from pathlib import Path
import pandas as pd
import requests
import tarfile
import os

def read_arxiv_papers(path):
    return pd.read_csv(path)

def get_eprint_link(paper):
    return f'http://export.arxiv.org/e-print/{paper.arxiv_id}'

def download_paper(url, path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download {url}")

V1_URL = 'https://github.com/paperswithcode/axcell/releases/download/v1.0/'
ARXIV_PAPERS_URL = V1_URL + 'arxiv-papers.csv.xz'
arxiv_papers = read_arxiv_papers(ARXIV_PAPERS_URL)

links = arxiv_papers.apply(get_eprint_link, axis=1)

# Specify the directory where you want to save the papers
save_dir = Path("./papers")
save_dir.mkdir(parents=True, exist_ok=True)

for idx, link in enumerate(links[:40], start=1):  # Just an example with `.tail()`, remove it to download all
    paper_id = arxiv_papers.iloc[idx - 1].arxiv_id  # Adjust index if necessary
    file_path = save_dir / f"{paper_id}.tar"
    print(f"Downloading {paper_id} to {file_path}...")
    download_paper(link, file_path)


export_dir = Path("./tex_files")
export_dir.mkdir(parents=True, exist_ok=True)
#%%

import tarfile
import os
# Path to the directory containing your tar files
directory_path = r'/Users/julianoll/Desktop/Fagprojekt/Project_Autocite'


# List all files in the directory
files = os.listdir(export_dir)
for file_name in files:
    if file_name[-4:]=='.tar':
        try:
            # Create a directory for each tar file
            tar_dir_name = os.path.splitext(file_name)[0]  # Use the file name without extension as the directory name
            tar_dir_path = os.path.join(export_dir, tar_dir_name)
            os.makedirs(tar_dir_path, exist_ok=True)  # Create directory if it doesn't exist
            
            # Open the tar file and extract its contents into the newly created directory
            with tarfile.open(os.path.join(directory_path, file_name), 'r:') as tar:
                tar.extractall(path=tar_dir_path)
            #os.remove(os.path.join(directory_path, file_name))

            print(f"{file_name} extracted successfully to {tar_dir_path}.")
        except tarfile.ReadError as e:
            print(f"Error extracting {file_name}: {e}")



# %%
