def find_cite(tex_file):
    """
    Find every instance of \cite{} in a .tex file and inserts \n before and after the instance.

    Arguments:
    tex_file: The .tex file to be searched for citations.

    Returns:
    None
    """
    with open(tex_file, 'r') as f:
        lines = f.readlines()

    with open(tex_file, 'w') as f:

        # For every line in the file, if the line contains \cite{}, add \n before and after the citation
        for line in lines:
            if "\\cite{" in line:
                line = line.replace("\\cite{", "\n \\cite{")
                line = line.replace("}", "}\n ")
            f.write(line)
            
    return None
