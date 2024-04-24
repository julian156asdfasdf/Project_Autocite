import time
#from thefuzz import fuzz
import re

#from pylatexenc.latex2text import LatexNodes2Text


#ACCENT_CONVERTER = LatexNodes2Text()

def remove_latex_commands(text):
    text = text.replace("et~al.", " ").replace("\n", ' ')
    equation_indexes = []
    last_dollar_idx = -1
    while "$" in text[last_dollar_idx+1:]:
        idx_dollar = text.find("$", last_dollar_idx+1)
        if text[idx_dollar-1] == "\\":
            last_dollar_idx = idx_dollar
        elif text[idx_dollar+1] == "$":
            last_dollar_idx = text.find("$$", idx_dollar+2)+1
            equation_indexes.append([idx_dollar, last_dollar_idx])
        else:
            last_dollar_idx = text.find("$", idx_dollar+1)
            equation_indexes.append([idx_dollar, last_dollar_idx])
    

    def check_if_in_equation(idx, equation_indexes, text_subtraction_size):
        for i in range(len(equation_indexes)):
            if idx >= equation_indexes[i][0]-text_subtraction_size and idx <= equation_indexes[i][1]-text_subtraction_size:
                return True
        return False
    
    text_subtraction_size = 0
    idx_backslash = text.find("\\")
    while idx_backslash != -1:
        if check_if_in_equation(idx_backslash, equation_indexes, text_subtraction_size):
            idx_backslash = text.find("\\", idx_backslash+1)
            continue
        break_chars = [" ", "{", "}", "\"", "\'", "´", "`", "\\", "_", "^"]
        if text[idx_backslash+1] in ["\'", "`", "´"]:
            text = text[:idx_backslash] + text[idx_backslash+1:]
            for i in range(len(equation_indexes)):
                equation_indexes[i][0] -= 1
                equation_indexes[i][1] -= 1
            idx_backslash = text.find("\\", idx_backslash+1)
            continue
        end_backslash_idx = len(text)
        extra_chars_removed = 0
        for break_char in break_chars:
            if text.find(break_char, idx_backslash+1) != -1:
                if text.find(break_char, idx_backslash+1) < end_backslash_idx and text.find(break_char, idx_backslash+1) > idx_backslash:
                    end_backslash_idx = text.find(break_char, idx_backslash+1)
                    if break_char in ["`", "´", "\""]:
                        extra_chars_removed = 1
                    elif break_char in ["\\", "}", "_", "^", "`", "´"]: 
                        extra_chars_removed = -(end_backslash_idx-idx_backslash)+1
                    else:
                        extra_chars_removed = 0
        
        text = text[:idx_backslash] + text[end_backslash_idx+extra_chars_removed:]
        text_subtraction_size += end_backslash_idx-idx_backslash+extra_chars_removed
        idx_backslash = text.find("\\", idx_backslash)
    return text.replace("{","").replace("}","").replace("\\ ", " ").strip()


def get_first_author_lastname_from_info(info):
    # Get the first author's full name (There might only be one author, in which case this does nothing)
    info_search = info.split(",")[0].split(" and ")[0].strip()
    # Identify the first author's last name (They are almost always marked with ~)
    strange_sign = -1
    if "~" in info_search:
        strange_sign = info_search.find("~")
        # If there is a \url command before the ~, then the ~ is not the last name but simply a part of the url
        if "\\url" in info_search and info_search.find("\\url") < strange_sign:
            strange_sign = -1
    info_search = info_search[strange_sign+1:].strip()

    # Remove the shortened first names of the author
    while len(info_search) >= 2:
        if info_search[0] in ' .~-' or info_search[1] in ' .~-':
            info_search = info_search[1:].strip()
        else:
            break
    # Mark the end of the authors last name
    if "." in info_search:
        first_dot = info_search.find(".")
        info_search = info_search[:first_dot].strip()
    first_author_name = info_search

    # Removes all latex comments in the text
    first_author_lastname = remove_latex_commands(first_author_name)
    # Retrieve the last of the names
    first_author_lastname = first_author_lastname.split(" ")[-1].split("~")[-1].strip()
    return first_author_lastname if len(first_author_lastname) > 1 else None


def parsebbl(file_path = None, bbl_str = None):
    """
    Parses a bbl file and returns a dictionary with the latex \cite-key as the keys and the value is a dict with title and authors as strings.
    """
    if file_path:
        try:
            with open(file_path, 'r', encoding='ISO-8859-1') as file:
                bbl_str = file.read()
        except Exception as e:
            print(f"Error reading file {file_path} in parsebbl: {e}")
            return {}
    elif not bbl_str:
        print("No input given to the parsebbl function. Exiting.")
        return {}
    
    if "\\begin{thebibliography}" not in bbl_str:
        print("No bibliography found in the bbl file. Exiting.")
        return {}

    # Remove comments in the bbl file
    bbl_str = bbl_str.split("\\begin{thebibliography}")[1].split("\\end{thebibliography}")[0]
    cleaned_str = re.sub(r"\\begin{comment}.*?\\end{comment}", "", bbl_str, flags=re.DOTALL | re.MULTILINE)
    cleaned_str = re.sub(r"%.*", "", cleaned_str)

    # Remove lines starting with "\\providecommand"
    cleaned_str = re.sub(r"\\providecommand.*\n", "", cleaned_str)

    ss = cleaned_str.split(r"\bibitem")[1:]
    
    bbl_dict = {}
    for s in ss:
        arxiv_id = None
        first_author_lastname = None
        # Case 1
        if s.strip().startswith("["):
            j = s.find("]")

            ref_name = s[j:].split("}")[0].replace("]", '').replace("{", '')
            info = s[j:]
            info = info[info.find("}")+1:].strip()

            # Subcase 1
            if r"\newblock" in cleaned_str or r"\newline" in cleaned_str:
                info = remove_latex_commands(info)
                # info = info.replace(r"\newblock", ' ').replace(r"\newline", ' ').replace(r"\em", ' ').replace(r"\emph", '').replace(r"\textbf", '')
                first_author_lastname = get_first_author_lastname_from_info(info)

            # Subcase 2
            elif info.startswith("{"):
                if info.find("\\eprint{") != -1:
                    arxiv_id = info.split("\\eprint{")[-1].split("}")[0]
                first_author_lastname = info[1:info.find("}")].split(" ")[-1]
                info = remove_latex_commands(info)
            # Subcase 3
            elif "\\bibfield" in info:
                info = info.split("\\BibitemOpen")[-1].split("\\BibitemShut")[0].strip()
                try:
                    first_author_lastname = info[info.find("{",info.find("\\bibnamefont"))+1:]
                    start_brackets = 0
                    for i in range(len(first_author_lastname)):
                        char = first_author_lastname[i]
                        if char == "{":
                            start_brackets += 1
                        elif char == "}":
                            start_brackets -= 1
                        if start_brackets == -1:
                            first_author_lastname = first_author_lastname[:i]
                            break
                    first_author_lastname = remove_latex_commands(first_author_lastname)
                except:
                    first_author_lastname = None
                while "\\bibinfo" in info:
                    idx = info.find("\\bibinfo")
                    info = info[:idx] + info[info.find("}", idx)+1:]
                while "\\bibfield" in info:
                    idx = info.find("\\bibfield")
                    info = info[:idx] + info[info.find("}", idx)+1:]
                idx_href = info.find("\\href")
                if idx_href != -1:
                    info = info[:idx_href] + info[info.find("}", idx_href)+1:]
                info = remove_latex_commands(info)
                # In some occatinos \bibnamefont isn't removed


        # Case 2
        elif s.strip().startswith("{"):
            s = s.replace(r"\newline", " ").replace(r"\newblock", ' ').replace("\n", ' ')
            ref_name = s[s.find("{")+1:s.find("}")]
            info = s[s.find("}")+1:]
            first_author_lastname = get_first_author_lastname_from_info(info)

        info = re.sub(r'\s+', ' ', info)
        bbl_dict[ref_name] = {"title":None, "info": info, "first_author_lastname": first_author_lastname, "ArXiV-ID": arxiv_id}

    return bbl_dict


if __name__ == "__main__":
    #filepaths = [
    #    "Step_1/0809.0840/heralhc_jhepwork.bbl", 
    #    "Step_1/1601.03559/paper_info.bbl", 
    #    "Step_1/1301.6295/GAMPFixPoint.bbl", 
    #    "Step_1/1411.7988/Thom.bbl", 
    #    "Step_1/1410.6029/AARevSpringer.bbl", 
    #    "Step_1/0912.0308/QuantitativeNon-abelianIdempotent.bbl", 
    #    "Step_1/1506.06908/Meister15.bbl", 
    #    "Step_1/1612.02830/CWBL1.tex"
    #]
    filepaths = ["Step_t/2003.04721/Bursi_text.bbl"]
    bbl_dict = {}
    t0 = time.time()
    for file_path in filepaths:
        print(f"\n{file_path}")
        bbl_dict.update(parsebbl(file_path=file_path))
    seconds = time.time() - t0
    print("\n")
    
    bad_authors = []
    for key, value in bbl_dict.items():
        print(f"{key=}, {value['first_author_lastname']=}, {value['info'][:30]}")
        if value["first_author_lastname"] is None:
            bad_authors.append(key)
    print(f"\nTime taken to process {len(filepaths)} bbl/tex files:", seconds)
    print(f"\nNo authors: {bad_authors}")