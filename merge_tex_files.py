import os
import tabulate
from collections import defaultdict

def tex_merge(directory, delete_originals=False, manifest=True):
    """
    For each subdirectory, all .tex files are merged into one file named after paper ID. The function also writes a manifest of the merged files.

    #### Arguments:
    directory : The root directory of the directory tree to be checked for .tex files. (The Processed folder)
    delete_originals (optional): If True, the function will remove the original .tex files after merging. Default is False. THE DELETION IS NOT IRREVERSIBLE, SO BE CAREFUL!
    manifest (optional): If True, the function will write a manifest of the merged files. Default is True.

    #### Returns:
    None
    """
    merge_dict = defaultdict(lambda: set())
    error_dict = defaultdict(list)
    
    for root, dirs, files in os.walk(directory):
        paper_id = root[16:27]

        # Skip merging if the file already exists
        if os.path.exists(os.path.join(root, f"{paper_id}.tex")):
            continue

        # For every file in the dir, add every .tex file together into one file named after the paper_id
        for file in files:
            if file.lower().endswith(".tex"):

                with open(os.path.join(root, file), 'r') as f:
                    try:
                        tex_content = f.read()
                    except UnicodeDecodeError:
                        print(f"Error reading file: {file}")
                        error_dict[paper_id].append(file)
                        continue
                    tex_content = f.read()

                merge_dict[paper_id].add(file)
                
                with open(os.path.join(root, f"{paper_id}.tex"), 'a') as f:
                    f.write(tex_content)
                
                # remove the file after merging if replace == True
                if delete_originals:
                    os.remove(os.path.join(root, file)) 
                    
                print(f"Added file: {file} to {paper_id}.tex")

    # Write a manifest of the merged files if manifest == True
    if manifest:
        with open("Merge_manifest.txt", 'w') as f:
            f.write(tabulate.tabulate(merge_dict, headers='keys'))
        with open("Merge_error_log.txt", 'w') as f:
            f.write(tabulate.tabulate(error_dict, headers='keys'))

if __name__ == "__main__":
    test_dir = "temporary_files"
    if os.path.exists(test_dir) == False:
        raise FileNotFoundError("The directory 'temporary_files' does not exist. Please create the directory and add some .tex files to it.")
    tex_merge("processed_tiles")