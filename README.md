# Project_Autocite

## Description
The main parts of this code is produced in the processing.py files. The project itself consists of multiple steps, which will produce a data set capable of being used in a machine learning setting. Furthermore, models will then be trained on the dataset to predict which articles should be a source for a given context window when writing a paper.
The data processing is divided into the following steps:

### Step 0: Data Collection
The data is collected from arXiv.

### Step 1: Data extraction
The data is extracted from collected data, and the data is cleaned, meaning everything but the .tex documents and possible reference files are deleted.

### Step 2: Data processing
The data is processed, meaning the .tex files are converted to .txt files, and the references are parsed and put into a references.json file for each paper.

### Step 3:
For each paper, i.e., a main.txt and a references.json, each reference is matched to its context in the paper it was cited, as well as the arXiv id of it. From this, the dataset is created as a .pkl file.