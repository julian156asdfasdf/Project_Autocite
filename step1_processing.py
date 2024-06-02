import os
from collections import defaultdict
from pathlib import Path
import tarfile
import os
import tabulate
import shutil
from tqdm.auto import tqdm

def delete_empty_folders(root: str) -> set:
    """
    A helper function for the rmv_irrelevant_files-function to delete empty folders in the directory tree after removing irrelevant files.

    Arguments:
        root: The root directory of the directory tree to be checked for empty folders. (The Processed folder)

    Returns:
        A set of the deleted folders.
    """

    deleted = set()
    
    for current_dir, subdirs, files in os.walk(root, topdown=False):
        #print(current_dir)
        #print(subdirs)
        #print(files)

        still_has_subdirs = False
        for subdir in subdirs:
            if os.path.join(current_dir, subdir) not in deleted:
                still_has_subdirs = True
                break
    
        if not any(files) and not still_has_subdirs:
            os.rmdir(current_dir)
            deleted.add(current_dir)
    with open("output_deleted_empty_folders.txt", 'w') as f:
        f.write(str(deleted))
    return deleted


class step1_processing:
    def __init__(self, directory, target_name):
        self.data = Path("./"+directory)
        self.target = target_name

    def create_target_folder(self) -> None:
        """
        Deletes the current Step_1 folder and creates a new empty one.
        """
        
        shutil.rmtree(self.target, ignore_errors=True)
        os.makedirs(self.target, exist_ok=False)

    def tar_extractor(self) -> set:
        """
        Takes a Papers directory and extracts the contents of the .tar files into a new directory called tex_files. In the tex_files directory
        there will a subfolder for each paper containing the contents of the corresponding .tar file.
        """

        unable_folder = set()
        files = os.listdir(self.data)
        for file_name in tqdm(files, desc="Extracting .tar files"):
            if file_name[-4:].lower() =='.tar':
                try:
                    #creates a folder in the directory folder for each paper
                    tar_dir_name = os.path.splitext(file_name)[0] 
                    tar_dir_path = os.path.join(self.target, tar_dir_name) #####
                    os.makedirs(tar_dir_path, exist_ok=True) 
                    
                    # Open the tar file and extract its contents into the newly created directory
                    with tarfile.open(os.path.join(self.data, file_name), 'r') as tar:
                        tar.extractall(path=tar_dir_path)
                    #os.remove(os.path.join(self.data, file_name))
                    # print(f"{file_name} extracted successfully to {tar_dir_path}.")
                except tarfile.ReadError as e:
                    print(f"Error extracting {file_name}: {e}")
                    unable_folder.add(file_name)
        with open("output_tar_unable_folders.txt", 'w') as f:
            f.write(str(unable_folder))
        return unable_folder    

    def rmv_irrelevant_files(self, manifest_title: str) -> dict:
        """
        Removes irrelevant files from the directory tree (The Processed Folder). The function removes all files that are not .tex, .bib or .bbl files.

        Arguments:
            manifest_title: The title of the manifest file that will be created.

        Returns: 
            A dictionary with the deleted files and their corresponding paper ID.
        """

        del_dict = defaultdict(lambda:set())

        # Walk through the directory tree and remove all files that are not .tex, .bib or .bbl files
        for root, dirs, files in os.walk(self.target):
            for file in files:
                if not file.lower().endswith(".tex") and not file.lower().endswith(".bib") and not file.lower().endswith(".bbl"):
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    del_dict[root[16:27]].add(file)
                    #print(f"Removed non-tex file: {file_path}")
        delete_empty_folders(self.target) #####
        
        with open(manifest_title, 'w') as f:
            f.write(tabulate.tabulate(del_dict, headers='keys'))
        return del_dict
    



if __name__ == "__main__":
    process = step1_processing("Step_0", "Step_1")
    process.create_target_folder()
    process.tar_extractor()
    process.rmv_irrelevant_files(manifest_title="manifest.txt")