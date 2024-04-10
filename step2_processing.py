import os
import json
from collections import defaultdict
from step1_processing import delete_empty_folders
from pathlib import Path
import re
import shutil
from parseBib import parseBib


class step2_processing:
    def __init__(self, directory, target_name):
        self.data = Path("./"+directory)
        self.target = target_name
        self.encoder = 'ISO-8859-1'
        self.manifest = defaultdict(lambda: set())
        
        
    def create_target_folder(self):
        """
        Creates a "step_2" folder outside the directory specified by self.data,
        copying the directory structure but without any files.

        Arguments:
        None

        Returns:
        None
        """
        if not os.path.exists(self.target):
            for root, dirs, files in os.walk(self.data):
                relative_path = os.path.relpath(root, self.data)
                new_folder = os.path.join(self.target, relative_path)
                os.makedirs(new_folder)
            print(f"Created folder: '{self.target}'")
        else:
            print(f"The {self.target} folder already exists.")
    

    def merge_tex_files(self, root, files):
        """
        Merges all the tex files in a subdirectory into one file called main.tex in the step_2 folder.

        Arguments:
        None

        Returns:
        None
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

        # Check if there is only one \documentclass in the folder
        doc_class_n = 0
        doc_main = None
        for file in files:
            if file.lower().endswith(".tex"): 
                with open(os.path.join(root, file), 'r', encoding=self.encoder) as f:  # Encoding is ISO-8859-1. The only working encoder for latex.
                    try:
                        content = str(f.read())
                        if r"\documentclass" in content:
                            all_dc_idx = [m.start() for m in re.finditer(r"\\documentclass", content)]
                            for m in re.finditer(r"%\documentclass", content): 
                                all_dc_idx.append(m.start()) 
                            # Check if there is a % before the \documentclass
                            true_dc_idx = []
                            for idx in all_dc_idx:
                                if r"%" in content[max(content.rfind("\n", 0, idx,),0):idx]:
                                    continue
                                else:
                                    true_dc_idx.append(idx)
                            # If there is more than 0 \documentclass in the file, set the content as the main file and increment the amount of files with \documentclass in root
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
        tex_file: The .tex file to be processed.

        Returns:
        None
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
        

        #with open(tex_file, 'r', encoding=self.encoder) as f:
        #    lines = f.readlines()

        #with open(tex_file, 'w', encoding=self.encoder) as f:
        #    for line in lines:
        #        if "\\cite{" in line: # \cite
        #            cites = re.findall(r"\\cite{.*?}", line)
        #            for cite in cites:
        #                if "," in cite:
        #                    split_cites = re.split(r",", re.search(r"\\cite{.*?}", cite).group()[6:-1]) # Splits up the citation. The group()[6:-1] is to remove the \cite{} part.
        #                    split_cites = [f"\\cite{{{split_cite.strip()}}}" for split_cite in split_cites] # Removes whitespace
        #                    line = line.replace(cite, ' '.join(split_cites))
        #        if "\\footcite{" in line: # \footcite
        #            cites = re.findall(r"\\footcite{.*?}", line)
        #            for cite in cites:
        #                if "," in cite:
        #                    split_cites = re.split(r",", re.search(r"\\footcite{.*?}", cite).group()[10:-1]) # Splits up the citation. The group()[10:-1] is to remove the \footcite{} part.
        #                    split_cites = [f"\\footcite{{{split_cite.strip()}}}" for split_cite in split_cites] # Removes whitespace
        #                    line = line.replace(cite, ' '.join(split_cites))
        #        if "\\citep{" in line: # \citep
        #            cites = re.findall(r"\\citep{.*?}", line)
        #            for cite in cites:
        #                if "," in cite:
        #                    split_cites = re.split(r",", re.search(r"\\citep{.*?}", cite).group()[7:-1]) # Splits up the citation. The group()[7:-1] is to remove the \citep{} part.
        #                    split_cites = [f"\\citep{{{split_cite.strip()}}}" for split_cite in split_cites] # Removes whitespace
        #                    line = line.replace(cite, ' '.join(split_cites))
        #        if "\\citet{" in line: # \citet
        #            cites = re.findall(r"\\citet{.*?}", line)
        #            for cite in cites:
        #                if "," in cite:
        #                    split_cites = re.split(r",", re.search(r"\\citet{.*?}", cite).group()[7:-1]) # Splits up the citation. The group()[7:-1] is to remove the \citet{} part.
        #                    split_cites = [f"\\citet{{{split_cite.strip()}}}" for split_cite in split_cites] # Removes whitespace
        #                    line = line.replace(cite, ' '.join(split_cites)) 
        #        f.write(line)
        
        return doc_contents

    def isolate_cites(self, doc_contents):
        """
        Find every instance of citations (\cite{}, \footcite{}, \citep, etc.) in a .tex file and inserts \n before and after the instance.

        Arguments:
        tex_file: The .tex file to be searched for citations.

        Returns:
        None
        """

        doc_contents = re.sub(r"(\\cite{.*?})", r"\n\1\n", doc_contents)
        doc_contents = re.sub(r"(\\footcite{.*?})", r"\n\1\n", doc_contents)
        doc_contents = re.sub(r"(\\citep{.*?})", r"\n\1\n", doc_contents)
        doc_contents = re.sub(r"(\\citet{.*?})", r"\n\1\n", doc_contents)
        #with open(tex_file, 'r', encoding=self.encoder) as f:
        #    lines = f.readlines()

        #with open(tex_file, 'w', encoding=self.encoder) as f:

            # For every line in the file, if the line contains a citation, add \n before and after the citation
            #for line in lines:
            #    if "\\cite{" in line: # \cite
            #        line = re.sub(r"(\\cite{.*?})", r"\n\1\n", line)
            #    if "\\footcite{" in line: # \footcite
            #        line = re.sub(r"(\\footcite{.*?})", r"\n\1\n", line)
            #    if "\\citep{" in line: # \citep
            #        line = re.sub(r"(\\citep{.*?})", r"\n\1\n", line)
            #    if "\\citet{" in line: # \citet
            #        line = re.sub(r"(\\citet{.*?})", r"\n\1\n", line)
            #    f.write(line)
                
        return doc_contents



    def extract_references(self):
        """
        Extracts the references from the .tex files in the directory tree and writes them to a .bbl file.
        
        Arguments:
        None
        
        Returns:
        None
        """
        count = 0
        for root, dirs, files in os.walk(self.data):
            for file in files:
                if file.lower().endswith(".tex"):
                    with open(os.path.join(root, file), 'r', encoding=self.encoder) as f:
                        try:
                            tex_content = f.read()
                        except UnicodeDecodeError:
                            print(f"Error reading file: {file}")
                            continue

                    
                    # Find the start and end indices of the bibliography section
                    start_index = tex_content.find(r"\begin{thebibliography}")
                    end_index = tex_content.find(r"\end{thebibliography}")
                    
                    # Check if the bibliography section exists in the tex_content
                    if start_index != -1 and end_index != -1:
                        # Extract the bibliography section
                        bibliography_section = tex_content[start_index:end_index + len(r"\end{thebibliography}")]
                    
                        # Write the bibliography section to the ref.bbl file in the step_2 folder
                        step_2_folder = os.path.join(os.path.dirname(self.data), self.target)
                        new_folder = os.path.join(step_2_folder, os.path.relpath(root, self.data))
                        os.makedirs(new_folder, exist_ok=True)
                        with open(os.path.join(new_folder, "ref_0.bbl"), 'w', encoding=self.encoder) as f:
                            f.write(bibliography_section)
                            self.manifest[root[16:27]].add(file)
                            count += 1
        print(f"Extracted references from {count} tex files")
    

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
        prev_root = "sdfgdgdfjsdfg"
        for root, dirs, files in os.walk(self.data):
            if Path("./"+root) == self.data or prev_root in root:
                continue
            prev_root = root

            new_main_file, doc_contents = self.merge_tex_files(root, files)

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
                
                



    def create_references_json(self):
        """
        parsed_bib_files = [] # List of dictionaries
        parsed_bbl_files = [] # List of dictionaries

        for each .bib file:
            parsed_bib_files.append(parse_bib(bib_file))
        for each .bbl file:
            parsed_bbl_files.append(parse_bbl(bbl_file))
        
        Combine the parsed_bib_files and parsed_bbl_files into a single dictionary

        Create json file containing the combined dictionary
        
        delete all the .bib and .bbl files from folder
        """
        
        # Walk through directory
        for root, dirs, files in os.walk(self.target):
            parsed_bib_files = {}
            parsed_bbl_files = {}

            for file in files:
                if file.lower().endswith(".bib"):
                    with open(os.path.join(root, file), 'r', encoding=self.encoder) as f:
                        bib_content = f.read()
                        
                        # Parse the .bib file and append it to the list
                        parsed_bib_files.update(parseBib(bib_content))
                        
                elif file.lower().endswith(".bbl"):
                    with open(os.path.join(root, file), 'r', encoding=self.encoder) as f:
                        bbl_content = f.read()
                        
                        # Parse the .bbl file and append it to the list
                        parsed_bbl_files.update(parseBbl(bbl_content))
            
            if root is self.target:
                continue

            # Combine the parsed files into one dictionary
            combined_dict = {**parsed_bib_files, **parsed_bbl_files}
            
            # Add 'arXiv-id' and 'abstract' keys to the dictionary
            for key in combined_dict.keys():
                combined_dict[key] = {**combined_dict[key], **{'arXiv-id': None, 'abstract': None}}
            
            # Make combined dictionary into a .json file
            with open(os.path.join(root, "references.json"), 'w') as f:
                json.dump(combined_dict, f)
            
            # Delete all .bib and .bbl files from the folder
            for file in files:
                if file.lower().endswith((".bib", ".bbl")):
                    os.remove(os.path.join(root, file))

        print("Created references.json file and removed all .bib and .bbl files.")
    

if __name__ == "__main__":
    process = step2_processing("Processed_files", "Step_2")
    process.create_target_folder()
    process.create_main_txt()
    #process.merge_tex_files()
    process.extract_references()
    process.remove_bib_from_main()
    process.move_references() 
    delete_empty_folders(process.target)
    process.create_references_json()