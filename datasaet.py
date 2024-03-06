# %%
#https://huggingface.co/datasets/ArtifactAI/arxiv_s2orc_parsed?row=0 
# Har fuld text

#https://huggingface.co/datasets/jamescalam/ai-arxiv-chunked?row=0 hhar reference liste

# https://huggingface.co/datasets/jamescalam/ai-arxiv?row=1 Lille subset i test fase med referenceliste

# udgangspunkt https://www.kaggle.com/datasets/Cornell-University/arxiv


'''
This program does the following:
 - Gets a subset of computer science arXiv articles, as a list
 - Downloads a subset of that dataset as tar files in local directory folder papers
 - unpacks tar files from papers, taking only tex, .bib and .bbl files to new directory tex_files
 - Displays the other files as a table, for each article ID
 - Displays tar files that could not be unpacked
 - Counts the total number of .bib and .bbl files in dataset
'''



# %%
from pathlib import Path
import pandas as pd
import requests
import tarfile
import os
import tabulate
from collections import defaultdict


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
arxiv_papers = arxiv_papers[arxiv_papers['status']=='success']

links = arxiv_papers.apply(get_eprint_link, axis=1)


#%%
# Specify the directory where you want to save the papers


def get_docs(dir ,links, k=100):
    for idx, link in enumerate(links[:k], start=1):  # Just an example with `.tail()`, remove it to download all
        paper_id = arxiv_papers.iloc[idx - 1].arxiv_id  # Adjust index if necessary
        file_path = dir / f"{paper_id}.tar"
        print(f"Downloading {paper_id} to {file_path}...")
        download_paper(link, file_path)


def tex_docs(save_dir,export_dir):
    unable_folder = set()
    files = os.listdir(save_dir)
    for file_name in files:
        if file_name[-4:].lower() =='.tar':
            try:
                #creates a folder in the directory folder for each paper
                tar_dir_name = os.path.splitext(file_name)[0] 
                tar_dir_path = os.path.join(export_dir, tar_dir_name)
                os.makedirs(tar_dir_path, exist_ok=True) 
                
                # Open the tar file and extract its contents into the newly created directory
                with tarfile.open(os.path.join(save_dir, file_name), 'r') as tar:
                    tar.extractall(path=tar_dir_path)
                #os.remove(os.path.join(save_dir, file_name))



                print(f"{file_name} extracted successfully to {tar_dir_path}.")
            except tarfile.ReadError as e:
                print(f"Error extracting {file_name}: {e}")
                unable_folder.add(file_name)
    with open("output_tar_unable_folders.txt", 'w') as f:
        f.write(str(unable_folder))
    return unable_folder
    

def delete_empty_folders(root):

    deleted = set()
    
    for current_dir, subdirs, files in os.walk(root, topdown=False):
        print(current_dir)
        print(subdirs)
        print(files)

        still_has_subdirs = False
        for subdir in subdirs:
            if os.path.join(current_dir, subdir) not in deleted:
                still_has_subdirs = True
                break
    
        if not any(files) and not still_has_subdirs:
            os.rmdir(current_dir)
            deleted.add(current_dir)
    with open("output_deleted_empty_folders.txt", 'w') as f:
        f.write(str(deleted))
    return deleted

def rmv_non_tex_files(directory):
    del_dict = {}
    for dir in os.listdir(directory):
        del_dict[dir] = set()

    del_dict = defaultdict(lambda:set())

    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.lower().endswith(".tex") and not file.lower().endswith(".bib") and not file.lower().endswith(".bbl"):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                del_dict[root[10:21]].add(file)
                print(f"Removed non-tex file: {file_path}")
    delete_empty_folders(directory)
    
    with open("output.txt", 'w') as f:
        f.write(tabulate.tabulate(del_dict, headers='keys'))
    return del_dict


# %%
save_dir = Path("./papers")
export_dir = Path("./tex_files")
current_dir = os.path.dirname(os.path.abspath(__file__))
#directory_path = r'/Users/julianoll/Desktop/Fagprojekt/Project_Autocite/papers'

get_docs(save_dir, links)

unable_folder = tex_docs(save_dir, export_dir)

dir = rmv_non_tex_files(export_dir)


#%%
from data_count import count_ref

bbl_count, bib_count = count_ref(export_dir)
print(f"Number of .bbl files: {bbl_count}")
print(f"Number of .bib files: {bib_count}")

# rmv_non_tex_files(export_dir)

# %%
