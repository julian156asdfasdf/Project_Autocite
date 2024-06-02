import re

def remove_latex_commands(text: str) -> None:
    """
    Removes all latex commands from a string.
    """

    text = text.replace("et~al.", " ").replace("\n", ' ')

    # Assigns the intervals of the equations in the text (Commands are not to be removed from them)
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
    
    # Checks if the index is within an equation after some text might been removed
    def check_if_in_equation(idx, equation_indexes, text_subtraction_size):
        for i in range(len(equation_indexes)):
            if idx >= equation_indexes[i][0]-text_subtraction_size and idx <= equation_indexes[i][1]-text_subtraction_size:
                return True
        return False
    text = text.replace("\'", "").replace("\`", "").replace("\´", "")
    # Remove all latex commands
    text_subtraction_size = 0
    idx_backslash = text.find("\\")
    while idx_backslash != -1:
        # If the backslash is the last character in the text, then break
        if idx_backslash == len(text)-1:
            text = text[:idx_backslash]
            break
        # If the backslash is part of an equation, then skip
        if check_if_in_equation(idx_backslash, equation_indexes, text_subtraction_size):
            idx_backslash = text.find("\\", idx_backslash+1)
            continue

        # If the backslash is simply to write a special character, then remove it the backslash and keep the special character
        if text[idx_backslash+1] in ["\'", "`", "´"]:
            text = text[:idx_backslash] + text[idx_backslash+1:]
            for i in range(len(equation_indexes)):
                equation_indexes[i][0] -= 1
                equation_indexes[i][1] -= 1
            idx_backslash = text.find("\\", idx_backslash+1)
            continue

        # Define the characters that can break the command
        break_chars = [" ", "{", "}", "\"", "\'", "´", "`", "\\", "_", "^"]
        end_backslash_idx = len(text)
        extra_chars_removed = 0
        for break_char in break_chars:
            if text.find(break_char, idx_backslash+1) != -1: # If the break_char is in the text after the backslash
                # If the new break_char appears closer to the backslash than the previous one, then update the end_backslash_idx
                if text.find(break_char, idx_backslash+1) < end_backslash_idx and text.find(break_char, idx_backslash+1) > idx_backslash:
                    end_backslash_idx = text.find(break_char, idx_backslash+1)
                    if break_char in [" ", "\"", "`", "´", "\'"]: # Remove the name of command and break_char
                        extra_chars_removed = 1
                    elif break_char in ["\\", "}", "_", "^"]:  # In these cases keep the name of the command
                        extra_chars_removed = -(end_backslash_idx-idx_backslash)+1
                    else: # Only remove the name of the command but keep break_char
                        extra_chars_removed = 0
        
        text = text[:idx_backslash] + text[end_backslash_idx+extra_chars_removed:] # Remove the command from the text
        text_subtraction_size += end_backslash_idx-idx_backslash+extra_chars_removed
        idx_backslash = text.find("\\", idx_backslash) # Find the next backslash
    return text.replace("{","").replace("}","").replace("\\ ", " ").strip() # Clean the string and return it


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

    # Retrieve the last of the names
    first_author_lastname = first_author_name.split(" ")[-1].split("~")[-1].strip()
    return first_author_lastname if len(first_author_lastname) > 1 else None


def parsebbl(file_path = None, bbl_str = None):
    """
    Parses a bbl file and returns a dictionary with the latex \cite-key as the keys and the value is a dict containing:
        string : arxiv_id if possible else its None
        string : title = None (For easier comparison with the parseBib function)
        string : first_authors_lastname (For possible easier search through KaggleDB)
        string : info = the rest of the information in the bbl file
    """

    # Read the bbl file or bbl string
    if file_path:
        try:
            with open(file_path, 'r', encoding='ISO-8859-1') as file:
                bbl_str = file.read()
        except Exception as e:
            # print(f"Error reading file {file_path} in parsebbl: {e}")
            return {}
    elif not bbl_str:
        # print("No input given to the parsebbl function. Exiting.")
        return {}
    
    # Check if the file contains a bibliography
    if "\\begin{thebibliography}" not in bbl_str:
        # print("No bibliography found in the bbl file. Exiting.")
        return {}

    # Remove comments in the bbl file
    bbl_str = bbl_str.split("\\begin{thebibliography}")[1].split("\\end{thebibliography}")[0]
    cleaned_str = re.sub(r"\\begin{comment}.*?\\end{comment}", "", bbl_str, flags=re.DOTALL | re.MULTILINE)
    cleaned_str = re.sub(r"%.*", "", cleaned_str)

    # Remove lines starting with "\\providecommand"
    cleaned_str = re.sub(r"\\providecommand.*\n", "", cleaned_str)

    # Splits the bbl file into the different bibitems
    ss = cleaned_str.split(r"\bibitem")[1:]
    
    bbl_dict = {} # The dictionary to be returned

    # Processes each bibitem
    for s in ss:
        arxiv_id = None
        first_author_lastname = None
        # Case 1 (If the bibitem starts with a bracket)
        if s.strip().startswith("["):
            j = s.find("]")

            # The ref name is the string inside the curly brackets following the square brackets
            info = s[j+1:]
            ref_name = info.split("}")[0].replace("{", '')
            info = info[info.find("}")+1:].strip()

            # Subcase 1
            # If the info starts with a curly bracket, then parse as such
            if info.startswith("{"):
                # If the info contains \eprint, then extract the arxiv_id
                if info.find("\\eprint{") != -1:
                    arxiv_id = info.split("\\eprint{")[-1].split("}")[0]
                    continue
                first_author_lastname = info[1:info.find("}")].replace("{","").split(" ")[-1] # Get the last name of the first author (It is within the first curly brackets)
                info = remove_latex_commands(info)

            # Subcase 2
            # If the info contains \bibfield, then parse as such
            elif "\\bibfield" in info or "\\bibinfo" in info:
                info = info.split("\\BibitemOpen")[-1].split("\\BibitemShut")[0].strip() # Remove the \BibitemOpen and \BibitemShut commands
                try:
                    # Get the beginning of the first author's last name 
                    first_author_lastname = info[info.find("{",info.find("\\bibnamefont"))+1:]
                    # Find the end of the first author's last name
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
                # Remove all \bibinfo and \bibfield commands
                while "\\bibinfo" in info:
                    idx = info.find("\\bibinfo")
                    info = info[:idx] + info[info.find("}", idx+1)+1:]
                while "\\bibfield" in info:
                    idx = info.find("\\bibfield")
                    info = info[:idx] + info[info.find("}", idx+1)+1:]
                # Remove all \url commands
                idx_href = info.find("\\href")
                if idx_href != -1:
                    info = info[:idx_href] + info[info.find("}", idx_href)+1:]
                info = remove_latex_commands(info)
            else:
                # Remove urls
                idx_url = info.find("\\url")
                if idx_url != -1:
                    info = info[:idx_url] + info[info.find("}", idx_url)+1:]
                info = remove_latex_commands(info) 
                first_author_lastname = get_first_author_lastname_from_info(info)

        # Case 2
        # If the bibitem starts with a curly bracket
        elif s.strip().startswith("{"):
            # Remove unnecessary characters
            s = s.replace(r"\newline", " ").replace(r"\newblock", ' ').replace("\n", ' ')
            # The ref name is the string inside the first set of curly brackets
            ref_name = s[s.find("{")+1:s.find("}")]
            info = s[s.find("}")+1:]
            # remove urls
            idx_url = info.find("\\url")
            if idx_url != -1:
                info = info[:idx_url] + info[info.find("}", idx_url)+1:]
            info = remove_latex_commands(info)
            first_author_lastname = get_first_author_lastname_from_info(info)
        else:
            info = s

        info = info.replace("\\textbf", "").replace("\\textit", "").replace("\\emph", "")
        info = re.sub(r'\s+', ' ', info) # Remove all extra whitespaces
        # Append the information to the dictionary
        bbl_dict[ref_name] = {"title":None, "info": info, "author_ln": first_author_lastname, "ArXiV-ID": arxiv_id}

    return bbl_dict


if __name__ == "__main__":
    import time
    # All the test files
    filepaths = [
        "Step_1/1403.1499/Manuscript.tex"
    ]
    bbl_dict = {}
    t0 = time.time()
    for file_path in filepaths:
        print(f"\n{file_path}")
        bbl_dict.update(parsebbl(file_path=file_path))
    seconds = time.time() - t0
    print("\n")
    
    bad_authors = []
    for key, value in bbl_dict.items():
        print(f"{key=}, {value['author_ln']=}, {value['info'][:30]}")
        if value["author_ln"] is None:
            bad_authors.append(key)
    print(f"\nTime taken to process {len(filepaths)} bbl/tex files:", seconds)
    print(f"\nNo authors: {bad_authors}")