import re

def isolate_cites(tex_file):
    """
    Find every instance of citations (\cite{}, \footcite{}, \citep, etc.) in a .tex file and inserts \n before and after the instance.

    Arguments:
    tex_file: The .tex file to be searched for citations.

    Returns:
    None
    """
    with open(tex_file, 'r') as f:
        lines = f.readlines()

    with open(tex_file, 'w') as f:

        # For every line in the file, if the line contains a citation, add \n before and after the citation
        for line in lines:
            if "\\cite{" in line: # \cite
                line = re.sub(r"(\\cite{.*?})", r"\n\1\n", line)
            if "\\footcite{" in line: # \footcite
                line = re.sub(r"(\\footcite{.*?})", r"\n\1\n", line)
            if "\\citep{" in line: # \citep
                line = re.sub(r"(\\citep{.*?})", r"\n\1\n", line)
            if "\\citet{" in line: # \citet
                line = re.sub(r"(\\citet{.*?})", r"\n\1\n", line)
            f.write(line)
            
    return None