from pathlib import Path
import pandas as pd
import os
from collections import defaultdict
import re
from RandomizeKaggleDB import read_json_DB
import json
import pickle
from pylatexenc.latex2text import LatexNodes2Text

from main import KAGGLEDB, ARXIV_IDS
from thefuzz import fuzz
from tqdm.auto import tqdm

ACCENT_CONVERTER_bad = LatexNodes2Text()

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
        for arxiv_id, value in tqdm(KAGGLEDB.items(), desc='Building author dictionary'):
            authors = value['authors']
            authors_list = authors.split(',')
            for author in authors_list:
                author = self.regex_on_author(author)
                self.author_db[author].add(arxiv_id)

        #i = 50
        #succesfull_iterations = []
        #for author in self.author_db.keys():
        #    for other_author in self.author_db.keys():
        #        if str(author) in other_author:
        #            self.author_db[author] = self.author_db[author].union(self.author_db[other_author])
        #            succesfull_iterations.append((author, other_author))
        #            i+=10
        return self.author_db
    
        # Problem is that the authors keys are not good example
        # the following are the different keys:  " 'Eric H'ebrard"," 'Eric Herbert (LIED)"," 'Eric Herbert and Pierre-Philippe Cortet",
        # Can be seen when doing sorted(self.author_db)
        # possible solution, make it a set instead of list, and if two keys contain the same, then union between long and short key on the short one
        # do not collapse the long one as it can run multiple times
        # possible implementation above, problem is that it runs incredibly slow

        #
        #for dir in os.listdir(self.file_dir):
        #    author_db = defaultdict()
        #    path = Path(os.path.join(self.file_dir,dir, 'references.json'))
        #    ref_json = pd.read_json(path, lines=True)
        #    for author in ref_json['author_ln']:
        #        print(author)

#Arg: target to match 
        # Out: list of possible matches

    def fuzzy_string_match(self, target: str, poss_match_list: list) -> str:
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
            ratio = fuzz.ratio(target, article)
            if ratio > best_match[0]:
                best_match = (index, ratio)
        
        return poss_match_list[best_match[0]]

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
        # it_worked = []

        for dir in tqdm(os.listdir(self.file_dir), desc='Matching references'):
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
                        #ref['ArXiV-ID'] = self.fuzzy_string_match(title,self.author_db[author_regex])
                        ref_json[latex_id]['ArXiV-ID'] = self.fuzzy_string_match(title,self.author_db[author_regex])
                        #it_worked.append(ref['ArXiV-ID'], title)


                    elif ref['info']:
                        title = self.infobbl_to_article_name(ref['info'])
                        #ref['ArXiV-ID'] = self.fuzzy_string_match(title,self.author_db[author_regex])
                        ref_json[latex_id]['ArXiV-ID'] = self.fuzzy_string_match(title,self.author_db[author_regex])
                        #it_worked.append((ref['ArXiV-ID'], title, ref['info']))
                    N_hits += 1
                else:
                    # Author not found in self.author_db
                    # print(author_regex, author)
                    pass
            
            new_ref_json = {}
            for key, value in ref_json.items():
                new_ref_json[key] = value
            with open(path, 'w') as f:
                json.dump(new_ref_json, f)
        # return N_total, N_hits, N_none, set(None_articles)#, it_worked
        return None

    def map_context(self, main_txt: str, ref_json: str, context_size: int=300) -> None:
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

        latex_commands = ['\\begin{', '\\cite{', '\\citet{', '\\citep{', '\\footcite{', '\\end{', 
                        '\\figure{', '\\includegraphics{', '\\includegraphics[', '\\label{', '\\ref{', '\\section{', 
                        '\\subsection{', '\\subsubsection{', '\\textcolor{', '\\textsubscript{', 
                        '\\textsuperscript', r'\usepackage[.*?]{', '\\usepackage{', '\\documentclass{', r'\frac{.*?}{',
                        '\\overline{']
        # '$.*?$', '$$.*?$$'
        
        with open(main_txt, 'r', encoding='ISO-8859-1') as f:
            text = f.read()

        with open(ref_json, 'r') as f:
            ref_dict = json.load(f)

        # with open(dataset_pkl, 'rb') as f:
        #     dataset = pickle.load(f)

        # Find all occurrences of the LaTeXID in the text and extract the context
        for LaTeXID, ref in ref_dict.items():
            arXivID = ref['ArXiV-ID']
            indices = [m.start() for m in re.finditer(re.escape(LaTeXID), text)]
            for index in indices:
                if index > 2000: # Limit the context to 2000 characters before the LaTeXID
                    context = text[index-2000:index]
                else:
                    context = text[:index]

                context = context.split() # Tokenization
                new_context = []

                # Connect the tokens that are part of the same command
                for i, token in enumerate(context):
                    if '{' in token:
                        j = i
                        while j < len(context) and '}' not in context[j]:
                            j += 1
                        new_context.append(' '.join(context[i:j+1]))
                    elif any('{' in token for token in new_context) and any('}' in token for token in context[i:]):
                        continue
                    else:
                        new_context.append(token)

                # Clean the context
                new_context = [token for token in new_context if not any(command in token for command in latex_commands)]
                try:
                    new_context = [ACCENT_CONVERTER.latex_to_text(token) for token in new_context]
                except:
                    continue
                new_context = [token.strip() for token in new_context]
                new_context = [token for token in new_context if token]
                new_context = ' '.join(new_context)

                # Limit the context size
                if len(new_context) > context_size:
                    new_context = new_context[-context_size:]

                # Append the context to the dataset
                self.dataset.append([main_txt[:-4].split('/')[-1], arXivID, new_context])

        return None


    def build_dataset(self, update: bool=True) -> None:
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
            except:
                update = False
                self.dataset = []
        else:
            self.dataset = []

        for dir in tqdm(os.listdir(self.file_dir), desc='Building dataset'):
            main_txt = os.path.join(self.file_dir,dir, dir +'.txt')
            ref_json = os.path.join(self.file_dir,dir, 'references.json')
            self.map_context(main_txt, ref_json, context_size=300)

        if update:
            pd_dataset = pd.DataFrame(self.dataset, columns=['main_arxiv_id', 'target_arxiv_id', 'context'])
            pd_dataset.drop_duplicates(inplace=True)
            pd_dataset.to_pickle(self.target)
        else:
            with open(self.target, 'wb') as f:
                pickle.dump(self.dataset, f)
    
        return None


if __name__ == "__main__":
    processing = step3_processing('Step_2', 'dataset.pkl')
    authors = processing.create_author_dict()
    processing.ref_matcher()
    processing.build_dataset(update=False)

    # print(N_total, N_hits, N_none)
    # print(f'None_articles:{None_articles}')
    # print(f'It_worked: {it_worked}')
