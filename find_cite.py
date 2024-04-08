import re

def split_cites(tex_file):
    """
    If an instance of a citation contains multiple sources, split them into separate instances, e.g., \cite{a,b} -> \cite{a} \cite{b}.

    Arguments:
    tex_file: The .tex file to be processed.

    Returns:
    None
    """
    with open(tex_file, 'r') as f:
        lines = f.readlines()

    with open(tex_file, 'w') as f:
        for line in lines:
            if "\\cite{" in line: # \cite
                cites = re.findall(r"\\cite{.*?}", line)
                for cite in cites:
                    if "," in cite:
                        split_cites = re.split(r",", re.search(r"\\cite{.*?}", cite).group()[6:-1]) # Splits up the citation. The group()[6:-1] is to remove the \cite{} part.
                        split_cites = [f"\\cite{{{split_cite.strip()}}}" for split_cite in split_cites] # Removes whitespace
                        line = line.replace(cite, ' '.join(split_cites))
            if "\\footcite{" in line: # \footcite
                cites = re.findall(r"\\footcite{.*?}", line)
                for cite in cites:
                    if "," in cite:
                        split_cites = re.split(r",", re.search(r"\\footcite{.*?}", cite).group()[10:-1]) # Splits up the citation. The group()[10:-1] is to remove the \footcite{} part.
                        split_cites = [f"\\footcite{{{split_cite.strip()}}}" for split_cite in split_cites] # Removes whitespace
                        line = line.replace(cite, ' '.join(split_cites))
            if "\\citep{" in line: # \citep
                cites = re.findall(r"\\citep{.*?}", line)
                for cite in cites:
                    if "," in cite:
                        split_cites = re.split(r",", re.search(r"\\citep{.*?}", cite).group()[7:-1]) # Splits up the citation. The group()[7:-1] is to remove the \citep{} part.
                        split_cites = [f"\\citep{{{split_cite.strip()}}}" for split_cite in split_cites] # Removes whitespace
                        line = line.replace(cite, ' '.join(split_cites))
            if "\\citet{" in line: # \citet
                cites = re.findall(r"\\citet{.*?}", line)
                for cite in cites:
                    if "," in cite:
                        split_cites = re.split(r",", re.search(r"\\citet{.*?}", cite).group()[7:-1]) # Splits up the citation. The group()[7:-1] is to remove the \citet{} part.
                        split_cites = [f"\\citet{{{split_cite.strip()}}}" for split_cite in split_cites] # Removes whitespace
                        line = line.replace(cite, ' '.join(split_cites)) 
            f.write(line)
    
    return None

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