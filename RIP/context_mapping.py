import json
import pickle
from pylatexenc.latex2text import LatexNodes2Text

ACCENT_CONVERTER = LatexNodes2Text()

def map_context(main_txt, ref_json, dataset_pkl='dataset.pkl', context_size=300):
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
                    '\\textsuperscript']

    with open(main_txt, 'r') as f:
        text = f.read()

    with open(ref_json, 'r') as f:
        ref_dict = json.load(f)

    # If the dataset_pkl file does not exist, create an empty list
    try:
        with open(dataset_pkl, 'rb') as f:
            dataset = pickle.load(f)
    except:
        dataset = []

    # with open(dataset_pkl, 'rb') as f:
    #     dataset = pickle.load(f)

    # Find the LaTeXID in the text and extract the context
    for LaTeXID, arXivID in ref_dict.items():
        if LaTeXID in text:
            if text.index(LaTeXID) > 2000: # Limit the context to 2000 characters before the LaTeXID
                context = text[text.index(LaTeXID)-2000:text.index(LaTeXID)]
            else:
                context = text[:text.index(LaTeXID)]

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
            new_context = [ACCENT_CONVERTER.latex_to_text(token) for token in new_context]
            new_context = [token.strip() for token in new_context]
            new_context = [token for token in new_context if token]
            new_context = ' '.join(new_context)

            # Limit the context size
            if len(new_context) > context_size:
                new_context = new_context[:context_size]

            # Append the context to the dataset
            dataset.append([main_txt[:-4], arXivID, new_context])

    # Pickle the dataset.pkl file
    with open(dataset_pkl, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"Context for {main_txt[:-4]} has been added to the dataset.pkl file.")

    return None