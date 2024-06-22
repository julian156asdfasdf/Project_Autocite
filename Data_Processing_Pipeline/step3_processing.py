import sys
import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.getcwd())

import os
from collections import defaultdict
import re
from RandomizeKaggleDB import read_json_DB
import json
import pickle
from pylatexenc.latex2text import LatexNodes2Text
import shutil
from collections import defaultdict

from main import KAGGLEDB, ARXIV_IDS
from thefuzz import fuzz
from tqdm.auto import tqdm

ACCENT_CONVERTER_bad = LatexNodes2Text()

def ACCENT_CONVERTER(text: str) -> str:
    """
    Cleans the text from all latex equations and figures.
    """
    # Remove all citations, label, and references:
    text = re.sub(r'\\cite\{.*?\}', '', text)
    text = re.sub(r'\\citep\{.*?\}', '', text)
    text = re.sub(r'\\citet\{.*?\}', '', text)
    text = re.sub(r'\\footcite\{.*?\}', '', text)
    text = re.sub(r'\\label\{.*?\}', '', text)
    text = re.sub(r'\\ref\{.*?\}', '', text)

    # Remove all begin-equations and begin-figures
    keywords_to_remove = ["figure", "equation", r"equation\*", "align", r"align\*", "gather", r"gather\*"]
    for keyword in keywords_to_remove:
        string_pattern = r'\\begin\{'+keyword+r'\}.*?\\end\{'+keyword+r'\}'
        text = re.sub(string_pattern, '', text, flags=re.DOTALL)
    
    dollar_index = text.find("$")
    if dollar_index != -1:
        # Remove all inline equations
        equation_indexes = []
        last_dollar_idx = -1
        while "$" in text[last_dollar_idx+1:]:
            idx_dollar = text.find("$", last_dollar_idx+1)
            # If the dollar sign is the last character in the text, then break
            if idx_dollar == len(text)-1:
                break
            # If the dollar sign is a command (Meaning not an equation)
            if text[idx_dollar-1] == "\\":
                last_dollar_idx = idx_dollar
            # If there are two dollar signs in a row, then it is an row-equation
            elif text[idx_dollar+1] == "$":
                last_dollar_idx = text.find("$$", idx_dollar+2)+1
                last_dollar_idx = (last_dollar_idx if last_dollar_idx != 0 else len(text))
                equation_indexes.append([idx_dollar, last_dollar_idx])
            else: # If there is only one dollar sign, then it is an inline-equation
                last_dollar_idx = text.find("$", idx_dollar+1)
                last_dollar_idx = (last_dollar_idx if last_dollar_idx != -1 else len(text))
                equation_indexes.append([idx_dollar, last_dollar_idx])
        for interval in reversed(equation_indexes):
            text = text[:interval[0]] + " " + text[interval[1]+1:]

    # Remove all latex commands
    text_subtraction_size = 0
    idx_backslash = text.find("\\")
    while idx_backslash != -1:
        # If the backslash is the last character in the text, then break
        if idx_backslash == len(text)-1:
            text = text[:idx_backslash]
            break

        # If the backslash is simply to write a special character, then remove it the backslash and keep the special character
        if text[idx_backslash+1] in ["'", "`", "´"]:
            text = text[:idx_backslash] + text[idx_backslash+1:]
            idx_backslash = text.find("\\", idx_backslash+1)
            continue 

        # Define the characters that can break the command
        break_chars = [" ", "{", "}", "\"", "'", "´", "`", "\\", "_", "^"]
        end_backslash_idx = len(text)
        extra_chars_removed = 0
        for break_char in break_chars:
            poss_end_backslash_idx = text.find(break_char, idx_backslash+1)
            if poss_end_backslash_idx != -1: # If the break_char is in the text after the backslash
                # If the new break_char appears closer to the backslash than the previous one, then update the end_backslash_idx
                if poss_end_backslash_idx < end_backslash_idx and poss_end_backslash_idx > idx_backslash:
                    end_backslash_idx = poss_end_backslash_idx
                    if break_char in ["\\", "}", "_", "^"]:  # In these cases keep the name of the command
                        extra_chars_removed = -(end_backslash_idx-idx_backslash)+1
                    else: # Only remove the name of the command but keep break_char
                        extra_chars_removed = 0
        
        text = text[:idx_backslash] + text[end_backslash_idx+extra_chars_removed:] # Remove the command from the text
        text_subtraction_size += end_backslash_idx-idx_backslash+extra_chars_removed
        idx_backslash = text.find("\\", idx_backslash) # Find the next backslash
    text = text.replace("{","").replace("}","").strip() # Clean the string and return it
    
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = text.strip()

    return text

# Match step 2 references.tex titles with kaggle db,
# Extract ArxivID and abstract
# Create JSON {latexID: [ArxivID, ]}
# download following https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download
# and call json arxiv_metadata

class step3_processing:
    def __init__(self, directory, target_name):
        self.file_dir = directory
        self.author_db = defaultdict(set)
        self.target = target_name

    def regex_on_author(self, author: str) -> str:
        """
        Finds the last word in the author string, which is the last name of the author.

        Arguments:
            author (str): The author string.

        Returns:
            str: The last name of the author.
        """

        if not author:
            return author
        if len(author)>6:
            if author[:5] == ' and ':
                author = author[5:]
        for auth in author.split(' and '):
            match = re.search(r'\w+$', auth)
            if match:
                author = match.group().lower()
        return author
    
    def infobbl_to_article_name(self, info: str) -> str:
        '''
        Finds the most likely title of the article from the info element of the bbl file

        Arguments:
            info: string of info from bbl file

        Returns:
            string: the most likely title of the article
        '''

        if not info:
            return info
        info_list = []
        for poss_title in info.split(';'):
            info_list.append(poss_title)
        
        target = max(info_list, key = len)

        return target
    
    def create_author_dict(self) -> dict:
        """
        Creates a dictionary with authors as keys and their corresponding ArXiV IDs as values.

        Arguments:
            None

        Returns:
            dict: A dictionary with authors as keys and their corresponding ArXiV IDs as values.
        """

        # method 1, each key is name of author
        for arxiv_id, value in tqdm(KAGGLEDB.items(), desc='Building author dictionary', leave=False):
            authors = value['authors']
            authors_list = authors.split(',')
            for author in authors_list:
                author = self.regex_on_author(author)
                self.author_db[author].add(arxiv_id)
        return self.author_db

    def fuzzy_string_match(self, target: str, poss_match_list: set) -> str | None:
        '''
        Finds the the highest match to target from a list of possible candidates
        
        Arguments:
            target: string to match
            poss_match_list: list of possible matches

        Returns:
            string: the best match
        '''
        poss_match_list = list(poss_match_list)
        best_match = (0,poss_match_list[0])

        # Find the best match
        for index, article in enumerate(poss_match_list):
            if KAGGLEDB[article]['info']:
                bm_query = KAGGLEDB[article]['info']
            elif KAGGLEDB[article]['title']:
                bm_query = KAGGLEDB[article]['title']
            else:
                continue
            ratio = fuzz.ratio(target, bm_query)
            if ratio > best_match[0]:
                best_match = (ratio, index)
        if best_match[0] > 77:
            return poss_match_list[best_match[1]]
        else:
            return None

    def ref_matcher(self) -> None:
        """ 
        Runs through all references.json, where it goes through each cite and attempts to find the arxiv id of the cite.

        Arguments: 
            None

        Returns:
            None
        
        # Output: Key metrics on performance and which articles worked and which didnt even have info
        """

        N_total, N_hits = 0, 0
        None_articles = []
        N_none = 0

        for dir in tqdm(os.listdir(self.file_dir), desc='Matching references', leave=False):
            path = os.path.join(self.file_dir,dir, 'references.json')
            ref_json = read_json_DB(path)
            for latex_id, ref in ref_json.items():
                author = ref['author_ln']
                author_regex = self.regex_on_author(author)
                N_total+=1

                if not author:
                    N_none+=1
                    None_articles.append(dir)

                if author_regex in self.author_db:
                    if ref['title']:
                        title = ref['title']
                        ref_json[latex_id]['ArXiV-ID'] = self.fuzzy_string_match(title,self.author_db[author_regex])

                    elif ref['info']:
                        title = self.infobbl_to_article_name(ref['info'])
                        ref_json[latex_id]['ArXiV-ID'] = self.fuzzy_string_match(title,self.author_db[author_regex])
                    N_hits += 1
                else:
                    # Author not found in self.author_db
                    pass

            new_ref_json = {}
            for key, value in ref_json.items():
                if value['ArXiV-ID']:
                    new_ref_json[key] = value
            if len(new_ref_json.keys()) == 0:
                # Delete the directory tree if no references were matched
                shutil.rmtree(os.path.join(self.file_dir, dir), ignore_errors=True)
                continue
            with open(path, 'w') as f:
                json.dump(new_ref_json, f)
        # return N_total, N_hits, N_none, set(None_articles)#, it_worked
        return None

    def map_context(self, main_txt: str, ref_json: str, context_size: int=500) -> None:
        """
        Maps the context of a citation in a .txt file to the corresponding arXivID and adds it to a dataset.pkl file along with the main_txt and arXivID.

        Arguments:
            main_txt: The .txt file containing the citations and main text.
            ref_json: The .json file containing the mapping between LaTeXID and arXivID.
            dataset_pkl: The .pkl file containing the dataset.
            context_size: The maximum size of the context.

        Returns:
            None
        """

        with open(main_txt, 'r', encoding='ISO-8859-1') as f:
            text = f.read()

        with open(ref_json, 'r') as f:
            ref_dict = json.load(f)

        # Find all occurrences of the LaTeXID in the text and extract the context
        dataset_add_counter = 0
        for LaTeXID, ref in ref_dict.items():
            arXivID = ref['ArXiV-ID']
            cite_indices = [m.start() for m in re.finditer(re.escape(LaTeXID), text)]
            for index in cite_indices:
                if index > 5000: # Limit the context to 2000 characters before the LaTeXID
                    context = text[index-5000:index]
                else:
                    context = text[:index]

                # Remove all LaTeX commands from the context
                new_context = ACCENT_CONVERTER(context)[-context_size:]

                # Check if the context contains too many math characters, and if so then skip it
                def check_freq(x) -> dict:
                    freq = defaultdict(lambda: 0)
                    for c in set(x):
                        freq[c] = x.count(c)
                    return freq
                char_freq = check_freq(new_context)
                math_chars = ['\\', '_', '^', '+', '-', '*', '/', '=', '(', ')', '<', '>', '|', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
                math_sum = 0
                for char in math_chars:
                    math_sum += char_freq[char]
                dirID = main_txt[:-4].split('/')[-1].split('\\')[-1]
                if math_sum >= len(new_context)/10:
                    continue
                else:
                    self.processedIDs.add(dirID)

                # Append the context to the dataset
                self.dataset.append([dirID, arXivID, new_context])
                dataset_add_counter += 1
        if dataset_add_counter == 0:
            shutil.rmtree(os.path.join(self.file_dir, main_txt[:-4].split('/')[-1].split('\\')[-1]), ignore_errors=True)
        return None

    def build_dataset(self, update: bool=True, context_size: int=1000) -> None:
        """
        Builds the dataset.pkl file containing the context of each citation in the main.txt files.

        Arguments:
            None

        Returns:
            None
        """

        if update:
            try:
                with open(self.target, 'rb') as f:
                    self.dataset = pickle.load(f)
                    self.processedIDs = set([d[0] for d in self.dataset])
            except:
                update = False
                self.dataset = []
                self.processedIDs = set()
        else:
            self.dataset = []
            self.processedIDs = set()

        self.notprocessedIDs = set()
        # Skip the directory if it has already been processed
        for dir in os.listdir(self.file_dir):
            if dir == '.DS_Store':
                continue
            if dir not in self.processedIDs:
                self.notprocessedIDs.add(dir)
                
        if len(self.notprocessedIDs) == 0:
            print("All directories have already been processed.")

        for dir in tqdm(self.notprocessedIDs, desc='Building dataset', leave=False):
            # Get the main.txt and references.json file paths and map the context to the referenced arXivID and add it self.dataset
            main_txt = os.path.join(self.file_dir,dir, dir +'.txt')
            ref_json = os.path.join(self.file_dir,dir, 'references.json')
            self.map_context(main_txt, ref_json, context_size=context_size)

        # Remove duplicates from the dataset
        data_tuple = [tuple(lst) for lst in self.dataset]
        data_set = set(data_tuple)
        self.dataset = [list(tup) for tup in data_set]

        # Save the dataset to a .pkl file
        with open(self.target, 'wb') as f:
            pickle.dump(self.dataset, f)
    
        return None


if __name__ == "__main__":
    processing = step3_processing(os.path.join('Data_Processing_Pipeline','Step_2'), 'dataset2.pkl')
    authors = processing.create_author_dict()
    processing.ref_matcher()
    processing.build_dataset(update=True, context_size=1000)