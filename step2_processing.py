import os
import json
from collections import defaultdict
from step1_processing import delete_empty_folders
from pathlib import Path
import re
import shutil
from parseBib import parseBib
from parseBBL2 import *
from tqdm.auto import tqdm

class step2_processing:
    def __init__(self, directory, target_name):
        self.data = Path("./"+directory)
        self.target = target_name
        self.encoder = 'ISO-8859-1'
        self.manifest = defaultdict(lambda: set())
        
        
    def create_target_folder(self) -> None:
        """
        Copies the directory structure of the self.data directory and appends it to the self.target directory.

        Arguments:
            None

        Returns:
            None
        """

        for dir in os.listdir(self.data):
            new_folder = os.path.join(self.target, dir)
            os.makedirs(new_folder, exist_ok=True)
    

    def merge_and_clean_tex_files(self, root: str):
        """
        Merges all the tex files in a subdirectory into one tex string.
        Also cleans the file by removing comments.

        Arguments:
            root: The root directory of the subdirectory.
        files: The files in the subdirectory.

        Returns:
            The path to the new main file and the cleaned tex string.
        """

        def clean_tex_string(tex_string: str) -> str:
            """
            Cleans the tex string by removing all comments.

            Arguments:
                tex_string: The tex string to be cleaned.

            Returns:
                The cleaned tex string.
            """

            # Remove all comments
            cleaned_tex_string = re.sub(r"\\begin{comment}.*?\\end{comment}", "", tex_string, flags=re.DOTALL | re.MULTILINE)
            cleaned_tex_string = re.sub(r"(?<!\\)%.*", "", clean_tex_string)

            return cleaned_tex_string

        # Get all files (not from subfolders as the main.tex file is assumed to be in the root folder)
        files = os.listdir(os.path.join(self.data, root))
        files = [os.path.join(self.data,root,file) for file in files if file.endswith(".tex")]

        # Check if there is only one \documentclass in the folder
        doc_class_n = 0
        doc_main = None
        for file in files:
            with open(file, 'r', encoding=self.encoder) as f:  # Encoding is ISO-8859-1. The only working encoder for latex.
                try:
                    content = str(f.read())
                    if r"\documentclass" in content or r"\documentstyle" in content:
                        #print(f"Found \documentclass or \documentstyle in {file}")
                        all_dc_idx = [m.start() for m in re.finditer(r"\\documentclass", content)]
                        for m in re.finditer(r"\\documentstyle", content):
                            all_dc_idx.append(m.start())
                        # Check if there is a % before the \documentclass or \documentstyle
                        true_dc_idx = []
                        for idx in all_dc_idx:
                            if r"%" in content[max(content.rfind("\n", 0, idx,),0):idx]:
                                continue
                            else:
                                true_dc_idx.append(idx)
                        # If there is more than 0 \documentclass or \documentstyle in the file, set the content as the main file and increment the amount of files with \documentclass or \documentstyle in root
                        if len(true_dc_idx) > 0:
                            doc_class_n += 1
                            doc_main = content
                except:
                    # print(f"Error reading file in merge_tex_files: {file}")
                    # Delete the folder if there is an error reading a file in it
                    return None, None 
        # If there is not 1 \documentclass in the folder, print an error and continue to the next folder
        if doc_class_n != 1:
            # print(f"Warning (can be ignored): {doc_class_n} \documentclass in {root}. Must be 1.")
            # Delete the folder if there is an error reading a file in it
            return None, None

        # Create a new main file in the step_2 folder
        new_main_file = os.path.join(Path(self.target), root, root+".txt")
        
        # Iteratively input all the \input files into the main file
        while True:
            doc_main = clean_tex_string(doc_main)
            all_inputs_idx = [m.start() for m in re.finditer(r"\\input\{(.+?)\}", doc_main)] # Finds all \input{...}
            
            # If there are no \input files, break the loop
            if len(all_inputs_idx) == 0:
                break
            else:
                # Iteratively input all the \input files into the main file from the last to the first
                for i in reversed(range(len(all_inputs_idx))):
                    idx = all_inputs_idx[i]
                    end_idx = doc_main.find("}", idx)
                    # the file being inputted:
                    input_path_name = doc_main[idx+7:end_idx].replace("/", "\\") + (".tex" if not doc_main[idx+7:end_idx].endswith((".tex",".bbl",".bib")) else "")
                    file_path = os.path.join(self.data, root, input_path_name)
                    # Read the file and input it into the main file
                    try:
                        with open(file_path, 'r', encoding=self.encoder) as input_file:  # Encoding is ISO-8859-1. The only working encoder for latex.
                            input_content = str(input_file.read())
                            doc_main = doc_main[:idx] + input_content + doc_main[end_idx+1:]
                    except Exception as e:
                        doc_main = doc_main[:idx+4] + "l" + doc_main[idx+5:]
                        # print(f"Warning (can be ignored) for inputting path in merge_tex_files: {file_path}.")
                        continue
        return new_main_file, clean_tex_string(doc_main)


    def split_cites(self, doc_contents: str) -> str:
        """
        If an instance of a citation contains multiple sources, split them into separate instances, e.g., \cite{a,b} -> \cite{a} \cite{b}.

        Arguments:
            String doc_contents: The contents of the .tex file to be searched for citations.

        Returns:
            doc_contents: The contents of the .tex file with the citations split up.
        """

        cites = re.findall(r"\\cite{.*?}", doc_contents)
        for cite in cites:
            if "," in cite:
                split_cites = re.split(r",", re.search(r"\\cite{.*?}", cite).group()[6:-1]) # Splits up the citation. The group()[6:-1] is to remove the \cite{} part.
                split_cites = [f"\\cite{{{split_cite.strip()}}}" for split_cite in split_cites] # Removes whitespace
                doc_contents = doc_contents.replace(cite, ' '.join(split_cites))

        cites = re.findall(r"\\footcite{.*?}", doc_contents)
        for cite in cites:
            if "," in cite:
                split_cites = re.split(r",", re.search(r"\\footcite{.*?}", cite).group()[10:-1]) # Splits up the citation. The group()[10:-1] is to remove the \footcite{} part.
                split_cites = [f"\\footcite{{{split_cite.strip()}}}" for split_cite in split_cites] # Removes whitespace
                doc_contents = doc_contents.replace(cite, ' '.join(split_cites))

        cites = re.findall(r"\\citep{.*?}", doc_contents)
        for cite in cites:
            if "," in cite:
                split_cites = re.split(r",", re.search(r"\\citep{.*?}", cite).group()[7:-1]) # Splits up the citation. The group()[7:-1] is to remove the \citep{} part.
                split_cites = [f"\\citep{{{split_cite.strip()}}}" for split_cite in split_cites] # Removes whitespace
                doc_contents = doc_contents.replace(cite, ' '.join(split_cites))

        cites = re.findall(r"\\citet{.*?}", doc_contents)
        for cite in cites:
            if "," in cite:
                split_cites = re.split(r",", re.search(r"\\citet{.*?}", cite).group()[7:-1]) # Splits up the citation. The group()[7:-1] is to remove the \citet{} part.
                split_cites = [f"\\citet{{{split_cite.strip()}}}" for split_cite in split_cites] # Removes whitespace
                doc_contents = doc_contents.replace(cite, ' '.join(split_cites)) 
        
        return doc_contents


    def isolate_cites(self, doc_contents: str) -> str:
        """
        Find every instance of citations (\cite{}, \footcite{}, \citep, etc.) in a .text string and inserts \n before and after the instance.

        Arguments:
            String doc_contents: The contents of the .tex file to be searched for citations.

        Returns:
            doc_contents: The contents of the .tex file with the citations isolated.
        """

        doc_contents = re.sub(r"(\\cite{.*?})", r"\n\1\n", doc_contents)
        doc_contents = re.sub(r"(\\footcite{.*?})", r"\n\1\n", doc_contents)
        doc_contents = re.sub(r"(\\citep{.*?})", r"\n\1\n", doc_contents)
        doc_contents = re.sub(r"(\\citet{.*?})", r"\n\1\n", doc_contents)
               
        return doc_contents


    def move_references(self) -> None:
        """
        Copies the references of .bbl and .bib files already in the self.data directory
        into the step_2 directory and renames them as ref.bbl or ref.bib depending on the format.

        Arguments:
            None

        Returns:
            None
        """

        count = 0
        base_folder = os.path.dirname(self.data)
        step_2_folder = os.path.join(base_folder, self.target)
        for root, dirs, files in os.walk(self.data):
            i_bib = 1
            i_bbl = 1
            for file in files:
                if file.lower().endswith((".bbl", ".bib")):
                    source_path = os.path.join(root, file)
                    destination_folder = os.path.join(step_2_folder, os.path.relpath(root, self.data))
                    os.makedirs(destination_folder, exist_ok=True)
                    if file.lower().endswith(".bbl"):
                        new_file_name = f"ref_{i_bbl}.bbl"
                        destination_path = os.path.join(destination_folder, new_file_name)
                        i_bbl += 1
                    elif file.lower().endswith(".bib"):
                        new_file_name = f"ref_{i_bib}.bib"
                        destination_path = os.path.join(destination_folder, new_file_name)
                        i_bib += 1
                    shutil.copy(source_path, destination_path)
                    count += 1
        # print(f"Moved {count} already established references to {self.target} folder")


    def create_main_txt(self) -> None:
        """
        Creates a main.txt file for each paper in the self.data directory into the self.target directory.

        Arguments:
            None

        Returns:
            None
        """

        for root in tqdm(os.listdir(self.data), desc="Creating main.txt files"):
            new_main_file, doc_contents = self.merge_and_clean_tex_files(root)

            if new_main_file is not None:
                # Split the citations in the main file
                doc_contents = self.split_cites(doc_contents)
                # Isolate the citations in the main file
                doc_contents = self.isolate_cites(doc_contents)
                # Write the main file to the step_2 folder
                file_write = open(new_main_file, 'w', encoding=self.encoder, errors='replace')
                file_write.write(doc_contents)
                file_write.close()
            else:
                shutil.rmtree(os.path.join(self.target, root), ignore_errors=True)

    def extract_references(self, file: str) -> str:
        """
        Helper function. Extracts the references from the given .tex file and returns it as a string
       
        Arguments:
            file: The file path
       
        Returns:
            bibliography section: The bibliography section of the .tex file as a string
        """

        with open(file, 'r', encoding=self.encoder) as f:
            try:
                text_content = f.read()
            except UnicodeDecodeError:
                #  print(f"Error reading file: {file}")
                return "" # Return empty string if there is an error reading the file
       
        # Find the start and end indices of the bibliography section
        start_index = text_content.find(r"\begin{thebibliography}")
        end_index = text_content.find(r"\end{thebibliography}")
       
        # Check if the bibliography section exists in the tex_content
        if start_index != -1 and end_index != -1:
            # Extract the bibliography section
            bibliography_section = text_content[start_index:end_index + len(r"\end{thebibliography}")]
            return bibliography_section
        return "" # Return empty string if there is no bibliography section in the file


    def create_references_json(self) -> None:
        """
        Creates a references.json file for each paper in the self.data directory into the self.target directory.
        
        Arguments:
            None
        
        Returns:
            None
        """

        base_folder = os.path.dirname(self.data)
        step_2_folder = os.path.join(base_folder, self.target)
        # Walk through directory
        for dir in tqdm(os.listdir(self.data), desc="Creating references.json files"):
            parsed = {}
            if dir not in os.listdir(self.target):
                continue
            ## LOOPING THROUGH THE DATA DIRECTORY ##
            for root, dirs, files in os.walk(os.path.join(self.data, dir)):
                for file in files:
                    if file.lower().endswith(".bib"):
                        # Read the .bib file and save it in a variable, that will update the dictionary
                        with open(os.path.join(root, file), 'r', encoding=self.encoder) as f:
                            bib_content = f.read()
                        
                        # Parse the .bib file and append it to the list
                        parsed.update(parseBib(bib_content))

                    elif file.lower().endswith(".bbl"):
                        # use the BblFile class to parse the .bbl file
                        with open(os.path.join(root, file), 'r', encoding=self.encoder) as f:
                            bbl_content = f.read()
                        # Parse the .bbl file and append it to the list
                        parsed.update(parsebbl(bbl_str=bbl_content))

            ## LOOKING AT THE TARGET DIRECTORY ##
            file_directory = os.path.join(os.path.join(step_2_folder, dir), dir+".txt")
            extracted_ref = self.extract_references(file_directory)
            if extracted_ref != "": # if it found some references in the .txt file, update dict
                parsed.update(parsebbl(bbl_str=extracted_ref))
            
            ## CREATING THE JSON FILE ##
            # Dont make the json file if there are no references
            if len(parsed) == 0:
                shutil.rmtree(os.path.join(self.target, dir), ignore_errors=True)
                continue
            
            # Add title, arXiv-id, info and author last name as None if they are not already in the
            for key in parsed.keys():
                parsed[key] = {**{'title': None, 'ArXiV-ID': None, 'info': None, 'author_ln': None}, **parsed[key]}
            
            destination_dir = os.path.join(step_2_folder, dir)
            with open(os.path.join(destination_dir, "references.json"), 'w') as f:
                json.dump(parsed, f)


if __name__ == "__main__":
    process = step2_processing("Step_1", "Step_2")
    process.create_target_folder()
    process.create_main_txt()
    process.create_references_json()