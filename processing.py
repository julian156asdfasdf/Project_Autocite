import os
from collections import defaultdict
import shutil

class processing:
    def __init__(self, directory):
        self.data = directory
        self.manifest = defaultdict(lambda: set())
        self.did_merge = False
        
        
    def create_step_2_folder(self):
        """
        Creates a "step_2" folder outside the directory specified by self.data,
        copying the directory structure but without any files.

        Arguments:
        None

        Returns:
        None
        """
        base_folder = os.path.dirname(self.data)
        step_2_folder = os.path.join(base_folder, "step_2")
        if not os.path.exists(step_2_folder):
            for root, dirs, files in os.walk(self.data):
                relative_path = os.path.relpath(root, self.data)
                new_folder = os.path.join(step_2_folder, relative_path)
                os.makedirs(new_folder)
            print(f"Created folder: 'step_2' in {base_folder}")
        else:
            print("The step_2 folder already exists.")
    
    
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
        step_2_folder = os.path.join(base_folder, "step_2")
        for root, dirs, files in os.walk(self.data):
            for file in files:
                if file.lower().endswith((".bbl", ".bib")):
                    source_path = os.path.join(root, file)
                    destination_folder = os.path.join(step_2_folder, os.path.relpath(root, self.data))
                    os.makedirs(destination_folder, exist_ok=True)
                    if file.lower().endswith(".bbl"):
                        destination_path = os.path.join(destination_folder, "ref.bbl")
                    elif file.lower().endswith(".bib"):
                        destination_path = os.path.join(destination_folder, "ref.bib")
                    shutil.copy(source_path, destination_path)
                    count += 1
        print(f"Moved {count} already established references to step_2 folder")


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
                    with open(os.path.join(root, file), 'r') as f:
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
                        step_2_folder = os.path.join(os.path.dirname(self.data), "step_2")
                        new_folder = os.path.join(step_2_folder, os.path.relpath(root, self.data))
                        os.makedirs(new_folder, exist_ok=True)
                        with open(os.path.join(new_folder, "ref.bbl"), 'w') as f:
                            f.write(bibliography_section)
                            self.manifest[root[16:27]].add(file)
                            count += 1
        print(f"Extracted references from {count} tex files")

    
    def merge_tex_files(self):
        """
        Merges all the tex files in each subdirectory into one file called main.tex in the step_1 folder.

        Arguments:
        None

        Returns:
        None
        """
        if self.did_merge:
            print("The tex files have already been merged.")
            return
        
        self.did_merge = True
        
        for root, dirs, files in os.walk(self.data):
            for file in files:
                if file.lower().endswith(".tex"):
                    with open(os.path.join(root, file), 'r') as f:
                        try:
                            tex_content = f.read()
                        except UnicodeDecodeError:
                            print(f"Error reading file: {file}")
                            continue
                    
                    # Remove the bibliography section from the tex_content
                    ref_start = tex_content.find(r"\begin{thebibliography}")
                    ref_end = tex_content.find(r"\end{thebibliography}")
                    if ref_start != -1 and ref_start != -1:
                        tex_content = tex_content[:ref_start] + tex_content[ref_end + len(r"\end{thebibliography}"):]
                    
                    # Append for every .tex file the tex_content to the main.tex file in the step_2 folder
                    step_2_folder = os.path.join(os.path.dirname(self.data), "step_2")
                    new_folder = os.path.join(step_2_folder, os.path.relpath(root, self.data))
                    os.makedirs(new_folder, exist_ok=True)
                    with open(os.path.join(new_folder, "main.tex"), 'a') as f:
                        f.write(tex_content)
        print("Merged all tex files into main.tex for each paper")


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
    directory = "./processed_tiles" # a dummy directory for testing
    process = processing(directory)
    process.create_step_2_folder()
    process.move_references()
    process.extract_references()
    process.merge_tex_files()
    # process.extract_references()
    # with open("Extracted_references.txt", 'w') as f:
    #     f.write(str(process.manifest))