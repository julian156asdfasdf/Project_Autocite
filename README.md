# Project_Autocite

## Description
This project is divided into two parts
- The first part involves creating the dataset using a Data Processing Pipeline, which in this repository is called dataset.pkl
- The second part involves creating and learning the ML model called Autocite, along with evaluating the model against different baselines.

### Description of Data Processing Pipeline
The idea is that, first make sure that a Randomized_Kaggle_Dataset json file has to exist in the directory specified in main.py. There are a few important notes to this:
- The first note is, that currently in the repository the file is called Randomized_Kaggle_Dataset_Subset_physics.json, which is a subset of 240.000 thousand rows in the original Kaggle Dataset.
- If there is no such json file, then go to the following link https://www.kaggle.com/datasets/Cornell-University/arxiv and download a file called Kaggle_Dataset.json from there. After that is done, then run the python script called RandomizeKaggleDB.py. In that file, it can be specified which categories of articles should be included.

After that, run the file called main.py in the folder called Dataset_Processing_Pipeline. In this file, it is important to specify some parameters first. 
- start_idx: This is the starting index in the Randomized_Kaggle_Dataset json file that is downloaded first.
- end_idx: The program will then download the articles of all rows from start_idx to end_idx. So make sure this number is larger than start_idx and smaller or equal to the total number of articles/rows in the Randomized_Kaggle_Dataset json file.
- window_size: A sliding window is implemented to avoid having downloaded too many articles' LaTeX projects at a time. window_size specifies how many should be downloaded and processed at a time.
- context_size: At the bottom of main.py, the context_size is specified, which is the amount of characters to have, at most, in the context of each row in the final dataset.pkl file.
- update: along with the context_size, the update parameter has to be decided on. True means that all new citations downloaded and processed are appended to an existing dataset.pkl file, removing duplicates, while False means, that it produces a new file, overwriting the old.

The Data Processing Pipeline is divided into four steps:

#### Step 0: Data Collection
The data is collected from arXiv.

#### Step 1: Data Extraction
The data is extracted from .tar file formats, and the data is cleaned, meaning everything but the .tex documents and possible reference files are deleted.

#### Step 2: Data Parsing
The .tex files are all concatenated into one .txt file, and the references are parsed and put into a references.json file for each paper.

#### Step 3: Dataset Creation
For each paper, i.e., a main.txt and a references.json, each citation's context is mapped to the arXiv-ID of the article that is referenced, using the library called TheFuzz. 
After that is done, all succesfully mapped citations are inserted in the dataset.pkl file.



## Training and Evaluating Autocite
This part should be understood in the following steps
- First create a vector embedded version of the dataset.pkl file. Make sure in all following documents that the correct path is used ofr the transformed_dataset.pkl file.
- Secondly, the ML model is implemented using PyTorch in the file called pytorch_model.py in the folder called Autocite. Here the model is defined, along with a loss class, an arXivDataset class and a class with the distance functions used. Further, a function for evaluating the model on a validation set is defined. Do not run this file yet.
- Next, the above-mentioned classes and functions are used in the file called hyperparameter_optimization.py, where the Optuna library is used to find the optimal hyperparameters. Code for relevant plotting can be found in plot_hyperopt_data.py.
- When the optimal hyperparameters are found, insert them in the pytorch_model.py script and then that script can be run to train the parameters for Autocite. All the data from training Autocite are saved in a folder called Training_Variables. Code for plotting training variables can be found in plot_training_data.py.
- Then the baseline models are created and evaluated in the baselines.py script.
- For performing the statistical test mentioned in the paper, run the file called create_stat_blindtest_data.py, and then check relevance between all the contexts and the corresponding abstracts and note the results in the wilcoxon_test.py, where the results achieved in the paper are already noted. Then run the wilcoxon_test.py script to get the p-value.