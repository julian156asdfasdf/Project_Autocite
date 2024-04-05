import os
from collections import defaultdict
from step1_processing import delete_empty_folders
from pathlib import Path
import re
import shutil

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
    

    def merge_tex_files(self):
        """
        Merges all the tex files in each subdirectory into one file called main.tex in the step_2 folder.

        Arguments:
        None

        Returns:
        None
        """

        
        prev_root = "sdfgdgdfjsdfg"
        for root, dirs, files in os.walk(self.data):
            if Path("./"+root) == self.data or prev_root in root:
                continue
            prev_root = root

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
                            print(f"Error reading file: {file}")
                            # Delete the folder if there is an error reading a file in it
                            break 
            # If there is not 1 \documentclass in the folder, print an error and continue to the next folder
            if doc_class_n != 1:
                print(f"Error: {doc_class_n} \documentclass in {root}. Must be 1.")
                # Delete the folder if there is an error reading a file in it
                continue
            
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
                            except UnicodeDecodeError:
                                print(f"Error reading file: {file_path}")
                                continue ##### ?????????
                            input_file.close()
                            doc_main = doc_main[:idx] + input_content + doc_main[end_idx+1:]
                        except Exception as e:
                            doc_main = doc_main[:idx+4] + "l" + doc_main[idx+5:]
                            print(f"Error for inputting path: {file_path}. Error: {e}")
                            continue
            # Write the main file to the step_2 folder
            file_write = open(new_main_file, 'w', encoding=self.encoder)
            file_write.write(str(doc_main))
            file_write.close()


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
### IF TIME, ADD THIS FUNCTIONALITY ###

# # Write a manifest of the merged files if manifest == True
#     if manifest:
#         with open("Merge_manifest.txt", 'w') as f:
#             f.write(tabulate.tabulate(merge_dict, headers='keys'))
#         with open("Merge_error_log.txt", 'w') as f:
#             f.write(tabulate.tabulate(error_dict, headers='keys'))

if __name__ == "__main__":
    process = step2_processing("Processed_files", "Step_2")
    process.create_target_folder()
    #process.merge_tex_files()
    process.extract_references()
    process.remove_bib_from_main()
    process.move_references() 
    delete_empty_folders(process.target)
    process.create_references_json()
    