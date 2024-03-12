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
                line = line.replace("\\cite{", "\n\\cite{")
                line = line.replace("}", "}\n")
            if "\\footcite{" in line: # \footcite
                line = line.replace("\\footcite{", "\n\\footcite{")
                line = line.replace("}", "}\n")
            if "\\citep{" in line: # \citep
                line = line.replace("\\citep{", "\n\\citep{")
                line = line.replace("}", "}\n")
            if "\\citet{" in line: # \citet
                line = line.replace("\\citet{", "\n\\citet{")
                line = line.replace("}", "}\n")
            f.write(line)
            
    return None