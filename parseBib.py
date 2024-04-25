import bibtexparser
import re
from parseBBL2 import remove_latex_commands

def parseBib(bibtex_str=None, bibtex_filepath=None):
    """
    Parses a bibtex string or file and returns a dictionary with the latex \cite-key as the keys and the value is a dict with title and authors as strings.
    """
    try:
        if bibtex_str:
            bib = bibtexparser.loads(bibtex_str)
        elif bibtex_filepath:
            bib = bibtexparser.load(open(bibtex_filepath, encoding='utf-8'))
        else:
            print("No input given to the parseBib function. Exiting.")
            return {}
    except:
        print("Error parsing bibtex file. Exiting.")
        return {}
    bib_dict = {}
    for entry in bib.entries:
        if 'title' in entry.keys():
            title = remove_latex_commands(entry['title'])
            idx_dollar = 0
            while "$" in title[idx_dollar:]:
                idx_dollar = title.find("$", idx_dollar)
                if idx_dollar == len(title)-1:
                    break
                if title[idx_dollar-1] == "\\":
                    idx_dollar = idx_dollar+1
                elif title[idx_dollar+1] == "$":
                    end_dollar_idx = title.find("$", idx_dollar+2)+1
                    if end_dollar_idx == 0:
                        end_dollar_idx = len(title)
                    title = title[:idx_dollar] + title[end_dollar_idx+1:]
                else:
                    end_dollar_idx = title.find("$", idx_dollar+1)
                    if end_dollar_idx == -1:
                        end_dollar_idx = len(title)
                    title = title[:idx_dollar] + title[end_dollar_idx+1:]
            
            title = re.sub(r'\s+', ' ', title).strip() # Remove all extra whitespaces
        else:
            title = None
        bib_dict[entry['ID']] = {'title': title, "info": None, "author_ln": None, "ArXiV-ID": None}
    return bib_dict # Should maybe be written to a text file