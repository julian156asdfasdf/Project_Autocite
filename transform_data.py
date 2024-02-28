
#%%
import pandas as pd
import json
import re
import requests
from bs4 import BeautifulSoup

# Defines the path for the complete and partial dataset
complete_dataset = "/Users/julianoll/Desktop/Fagprojekt/Project_Autocite/arxiv-metadata-oai-snapshot.json"
partial_dataset = "./arxiv_cs.json"
pattern = r"(^|\s)cs\."
def create_dataset_subset(pattern, fileout_name):
    with open(complete_dataset, 'r') as f_in, open(fileout_name, 'w') as f_out:
        for line in f_in:
            entry = json.loads(line)
            if re.search(pattern, entry["categories"]):
                json.dump(entry, f_out)
                f_out.write('\n')
#create_dataset_subset(pattern, partial_dataset)
                
pd.set_option("display.max_colwidth", None)
# Defines the CS-ArXiv-df
#%%
cs_arxiv_df = pd.read_json(partial_dataset, lines=True)
#%%
def get_eprint_link(id):
    return f'http://export.arxiv.org/e-print/{id}'
links = []
for id in cs_arxiv_df['id']:
    links.append(get_eprint_link(id))

cs_arxiv_df.insert(1,'links',links)
# %%
from tex2py import tex2py
from TexSoup import TexSoup

from pylatexenc.latexwalker import LatexWalker
path = '/Users/julianoll/Desktop/Fagprojekt/Project_Autocite/tex_files/0705.4676v8/viewsizeestimation.tex'
with open(path) as f: data = f.read()
toc = tex2py(data)
soup = TexSoup(data)


labels = set(label.string for label in soup.find_all('cite'))

# %%
def count(tex):
    """Extract all labels, then count the number of times each is referenced in
    the provided file. Does not follow \includes.
    """

    # soupify
    soup = TexSoup(tex)

    # extract all unique labels
    labels = set(label.string for label in soup.find_all('label'))

    # create dictionary mapping label to number of references
    label_refs = {}
    for label in labels:
        refs = soup.find_all('\\ref{%s}' % label)
        pagerefs = soup.find_all('\\pageref{%s}' % label)
        label_refs[label] = len(list(refs)) + len(list(pagerefs))

    return label_refs
count(data)
# %%
