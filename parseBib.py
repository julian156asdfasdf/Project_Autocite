import bibtexparser

def parseBib(bibtex_str=None, bibtex_filepath=None):
    """
    Parses a bibtex string or file and returns a dictionary with the latex \cite-key as the keys and the value is a dict with title and authors as strings.
    """
    if bibtex_str:
        bib = bibtexparser.loads(bibtex_str)
    elif bibtex_filepath:
        bib = bibtexparser.load(open(bibtex_filepath, encoding='utf-8'))
    else:
        print("No input given to the parseBib function. Exiting.")
        return {}
    bib_dict = {}
    for entry in bib.entries:
        bib_dict[entry['ID']] = {'title': entry['title'] if 'title' in entry.keys() else ''}
    return bib_dict # Should maybe be written to a text file