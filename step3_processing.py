from pathlib import Path
import pandas as pd
import dask.dataframe as dd
import os

# Match step 2 references.tex titles with kaggle db,
# Extract ArxivID and abstract
# Create JSON {latexID: [ArxivID, ]}
# download following https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download
# and call json arxiv_metadata

# temporary only take the first 300k datapoints
db = pd.read_json(Path('Kaggle_Dataset.json'), lines=True, chunksize=50000)
for chunk in db:
    df = chunk
    break

df.to_json(Path('step0_kaggle_db.json'))
if not os.path.exists(Path('Step_3')):
    os.mkdir('Step_3')


""" 
for latexid in os.listdir(Path('Step2')):
    for json in os.listdir(Path('Step2'+str(latexid))):
        os.mkdir('Step3'+str(latexid))
        
 """
