import os
import json
from collections import defaultdict
from step1_processing import delete_empty_folders
from pathlib import Path
import re
import shutil
from parseBib import parseBib
from parseBbl import *


class step2_processing:
    def __init__(self, directory, target_name):
        self.data = Path("./"+directory)
        self.target = target_name
        self.encoder = 'ISO-8859-1'
        self.manifest = defaultdict(lambda: set())
        
        
    def create_target_folder(self):
        """
        Copies the directory structure of the self.data directory and appends it to the self.target directory.
        """
        for root, dirs, files in os.walk(self.data):
            relative_path = os.path.relpath(root, self.data)
            new_folder = os.path.join(self.target, relative_path)
            os.makedirs(new_folder, exist_ok=True)
    

    def merge_and_clean_tex_files(self, root):
        """
        Merges all the tex files in a subdirectory into one tex string.
        Also cleans the file by removing comments.

        Arguments:
        root: The root directory of the subdirectory.
        files: The files in the subdirectory.

        Returns:
        The path to the new main file and the cleaned tex string.
        """

        def clean_tex_string(tex_string):
            """
            Cleans the tex string by removing all comments.

            Arguments:
            tex_string: The tex string to be cleaned.

            Returns:
            The cleaned tex string.
            """
            # Remove all comments
            cleaned_tex_string = re.sub(r"\\begin{comment}.*?\\end{comment}", "", tex_string, flags=re.DOTALL | re.MULTILINE)
            cleaned_tex_string = re.sub(r"%.*", "", cleaned_tex_string)
            
            return cleaned_tex_string

        # Get all files (also from subfolders)
        files = []
        for r, d, f in os.walk(root):
            for file in f:
                files.append(file)

        # Check if there is only one \documentclass in the folder
        doc_class_n = 0
        doc_main = None
        for file in files:
            if file.lower().endswith(".tex"): 
                with open(os.path.join(root, file), 'r', encoding=self.encoder) as f:  # Encoding is ISO-8859-1. The only working encoder for latex.
                    try:
                        content = str(f.read())
                        if r"\documentclass" in content or r"\documentstyle" in content:
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
                    except UnicodeDecodeError:
                        print(f"Error reading file in merge_tex_files: {file}")
                        # Delete the folder if there is an error reading a file in it
                        return None, None 
        # If there is not 1 \documentclass in the folder, print an error and continue to the next folder
        if doc_class_n != 1:
            print(f"Error: {doc_class_n} \documentclass in {root}. Must be 1.")
            # Delete the folder if there is an error reading a file in it
            return None, None

        # Create a new main file in the step_2 folder
        new_main_file = os.path.join(os.path.dirname(self.data), Path("./"+self.target), os.path.relpath(root, self.data), "main.txt")
        
        # Iteratively input all the \input files into the main file
        while True:
            all_inputs_idx = [m.start() for m in re.finditer(r"\\input\{(.+?)\}", doc_main)] # Finds all \input{...}
            # Check if there is a % before the \input
            true_inputs_idx = []
            for idx in all_inputs_idx:
                if "%" in doc_main[doc_main.rfind("\n", 0, idx):idx]:
                    continue
                else:
                    true_inputs_idx.append(idx)
            # If there are no \input files, break the loop
            if len(true_inputs_idx) == 0:
                break
            else:
                # Iteratively input all the \input files into the main file from the last to the first
                for i in range(len(true_inputs_idx)-1, -1, -1):
                    idx = true_inputs_idx[i]
                    end_idx = doc_main.find("}", idx)
                    # the file being inputted:
                    file_path = os.path.join(root, doc_main[idx+7:end_idx].replace("/", "\\") + (".tex" if not doc_main[idx+7:end_idx].endswith((".tex",".bbl",".bib")) else ""))
                    # Read the file and input it into the main file
                    try:
                        input_file = open(file_path, 'r', encoding=self.encoder)
                        try:
                            input_content = str(input_file.read())
                            doc_main = doc_main[:idx] + input_content + doc_main[end_idx+1:]
                        except UnicodeDecodeError:
                            print(f"Error reading file for inputting in merge_tex_files: {file_path}")
                            continue 
                        input_file.close()
                    except Exception as e:
                        doc_main = doc_main[:idx+4] + "l" + doc_main[idx+5:]
                        print(f"Error for inputting path in merge_tex_files: {file_path}. Error: {e}")
                        continue
        return new_main_file, clean_tex_string(doc_main)

    def split_cites(self, doc_contents):
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

    def isolate_cites(self, doc_contents):
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


    def extract_references(self, file):
        """
        Helper function. Extracts the references from the given .tex file and returns it as a string
       
        Arguments:
        The file path [str]
       
        Returns:
        bibliography section [str], if it found any,
        else empty string [str]
        """
        with open(file, 'r', encoding=self.encoder) as f:
            try:
                tex_content = f.read()
            except UnicodeDecodeError:
                print(f"Error reading file: {file}")
                return ""
       
        # Find the start and end indices of the bibliography section
        start_index = tex_content.find(r"\begin{thebibliography}")
        end_index = tex_content.find(r"\end{thebibliography}")
       
        # Check if the bibliography section exists in the tex_content
        if start_index != -1 and end_index != -1:
            # Extract the bibliography section
            bibliography_section = tex_content[start_index:end_index + len(r"\end{thebibliography}")]
            return bibliography_section
        return ""
    

    def remove_bib_from_main(self):
        for root, dirs, files in os.walk(self.encoder):
            main_file_name = "main.txt"
            if os.path.isfile(os.path.join(root, main_file_name)):
                file = open(os.path.join(root, main_file_name), 'r', encoding=self.encoder)
                doc_content = file.read()
                print(doc_content)
                file.close()
                ref_start = doc_content.find(r"\begin{thebibliography}")
                ref_end = doc_content.find(r"\end{thebibliography}")+21
                if ref_start != -1 and ref_start != -1:
                    doc_content = doc_content[:ref_start] + doc_content[ref_end:]
                file = open(os.path.join(root, main_file_name), 'w', encoding=self.encoder)
                file.write(str(doc_content))
                file.close()


    def move_references(self):
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
        print(f"Moved {count} already established references to {self.target} folder")


    def create_main_txt(self):
        """
        Creates a main.txt file for each paper in the self.data directory into the self.target directory.
        """
        for root in os.listdir(self.data):
            files = []
            for r, d, f in os.walk(root):
                for file in f:
                    files.append(file)

            new_main_file, doc_contents = self.merge_and_clean_tex_files(root)

            if new_main_file is not None:
                # Split the citations in the main file
                doc_contents = self.split_cites(doc_contents)
                # Isolate the citations in the main file
                doc_contents = self.isolate_cites(doc_contents)
                # Write the main file to the step_2 folder
                file_write = open(new_main_file, 'w', encoding=self.encoder)
                file_write.write(doc_contents)
                file_write.close()
                pass
            else:
                shutil.rmtree(root.replace(str(self.data),self.target), ignore_errors=True)


    def create_references_json(self):
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
        for root, dirs, files in os.walk(self.data):
            parsed = {}
            for file in files:
                if file.lower().endswith(".bib"):
                    # Read the .bib file and save it in a variable, that will update the dictionary
                    with open(os.path.join(root, file), 'r', encoding=self.encoder) as f:
                        bib_content = f.read()
                       
                    # Parse the .bib file and append it to the list
                    parsed.update(parseBib(bib_content))
                       
                elif file.lower().endswith(".bbl"):
                    # use the BblFile class to parse the .bbl file
                    bbl_path = os.path.join(root, file)
                    try:
                        bbl_parsed = BblFile(bbl_path)
                    except Exception as e:
                        print(f"Error parsing bbl file: {bbl_path}. Error: {e}")
                        continue
                    # Parse the .bbl file and append it to the list
                    parsed.update(bbl_parsed.bib_dict)
 
                elif file.lower().endswith(".tex"):
                    # First use the helper function to extract the references from the .tex file
                    extracted_ref = self.extract_references(os.path.join(root,file))
                    if extracted_ref != "": # if it found some references in the .tex file, update dict
                        # Then make a temporary .bbl file such that the BblFile class can parse it
                        temp_bbl_path = os.path.join(root, "temp.bbl")
                        with open(temp_bbl_path, 'w', encoding=self.encoder) as f:
                            f.write(extracted_ref)
                        # Parse the temporary .bbl file
                        try: 
                            bbl_parsed = BblFile(temp_bbl_path)
                        except Exception as e:
                            print(f"Error parsing bbl file: {temp_bbl_path}. Error: {e}")
                            continue
                        # Append the parsed .bbl file to the list
                        parsed.update(bbl_parsed.bib_dict)
            
            if root is self.data.stem:
                continue
            
            # Dont make the json file if there are no references
            if len(parsed) == 0:
                continue
            
            # Add 'arXiv-id' and 'abstract' keys to the dictionary
            for key in parsed.keys():
                parsed[key] = {**parsed[key], **{'arXiv-id': None, 'abstract': None}}
           
            # Make dictionary into a .json file
            destination_dir = os.path.join(step_2_folder, os.path.relpath(root, self.data))
            with open(os.path.join(destination_dir, "references.json"), 'w') as f:
                json.dump(parsed, f)
           
        print("Created references.json file and removed all .bib and .bbl files.")
    

if __name__ == "__main__":
    process = step2_processing("Step_1", "Step_2")
    # process.create_references_json()
    # process.create_target_folder()
    process.create_main_txt()
    # #process.merge_tex_files()
    # process.extract_references()
    # process.remove_bib_from_main()
    # process.move_references() 
    # delete_empty_folders(process.target)
    # process.create_references_json()