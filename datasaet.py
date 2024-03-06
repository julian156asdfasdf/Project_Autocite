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
import random

from data_count import count_ref


def read_arxiv_papers(path):
    return pd.read_csv(path)

def get_eprint_link(paper):
    return f'http://export.arxiv.org/e-print/{paper.arxiv_id}'

V1_URL = 'https://github.com/paperswithcode/axcell/releases/download/v1.0/'
ARXIV_PAPERS_URL = V1_URL + 'arxiv-papers.csv.xz'
arxiv_papers = read_arxiv_papers(ARXIV_PAPERS_URL)
arxiv_papers = arxiv_papers[arxiv_papers['status']=='success']

links = list(arxiv_papers.apply(get_eprint_link, axis=1))

random.shuffle(links)
# Specify the directory where you want to save the papers

def get_docs(dir, links, k=200):
    """
    Takes a directory and a list of links to papers and downloads the papers as .tar files to the directory.
    """
    for idx, link in enumerate(links[:k], start=1):  # Just an example with `.tail()`, remove it to download all
        paper_id = arxiv_papers.iloc[idx - 1].arxiv_id  # Adjust index if necessary
        file_path = dir / f"{paper_id}.tar"
        print(f"Downloading {paper_id} to {file_path}... " + str(idx) + "/" + str(k))

        # Download a paper from the given URL and save it to the given path.
        response = requests.get(link)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download {link}")

#%%
def tar_extractor(save_dir,export_dir):
    """
    Takes a Papers directory and extracts the contents of the .tar files into a new directory called tex_files. In the tex_files directory
    there will a subfolder for each paper containing the contents of the corresponding .tar file.
    """
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
    """
    A helper function for the rmv_irrelevant_files-function to delete empty folders in the directory tree after removing irrelevant files.

    Arguments:
    root: The root directory of the directory tree to be checked for empty folders. (The Processed folder)

    Returns:
    A set of the deleted folders.
    """

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

def rmv_irrelevant_files(directory):
    """
    Removes irrelevant files from the directory tree (The Processed Folder). The function removes all files that are not .tex, .bib or .bbl files.

    Arguments:
    directory: The root directory of the directory tree to be checked for irrelevant files. (The Processed folder)

    Returns: 
    A dictionary with the deleted files and their corresponding paper ID.
    """

    del_dict = defaultdict(lambda:set())

    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.lower().endswith(".tex") and not file.lower().endswith(".bib") and not file.lower().endswith(".bbl"):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                del_dict[root[16:27]].add(file)
                print(f"Removed non-tex file: {file_path}")
    delete_empty_folders(directory)
    
    with open("manifest.txt", 'w') as f:
        f.write(tabulate.tabulate(del_dict, headers='keys'))
    return del_dict


# %%

if __name__ == '__main__':
    save_dir = Path("./papers")
    os.makedirs(save_dir, exist_ok=True)
    export_dir = Path("./Processed_files")
    os.makedirs(export_dir, exist_ok=True)

    get_docs(save_dir, links)

    unable_folder = tar_extractor(save_dir, export_dir)

    dir = rmv_irrelevant_files(export_dir)

    bbl_count, bib_count = count_ref(export_dir)
    print(f"Number of .bbl files: {bbl_count}")
    print(f"Number of .bib files: {bib_count}")
# %%
