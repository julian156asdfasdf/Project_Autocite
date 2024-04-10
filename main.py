from step0_processing import step0_processing
from step1_processing import step1_processing
from step2_processing import step2_processing
from step1_processing import delete_empty_folders
#import os
#from datasaet import get_tar_links, get_docs, tar_extractor, rmv_irrelevant_files
#from data_count import count_ref

if __name__ == '__main__':
# step 0 Download Data
    print("Starting Step 0...")
    step_0_target_name = "Step_0"
    step_1 = step0_processing(target_name = step_0_target_name)
    step_1.create_target_folder()
    step_1.get_tar_links()
    step_1.get_docs(k=100)
    print("Downloaded papers successfully.")
    #if os.path.exists(step_0) == False:
    #    os.makedirs(step_0, exist_ok=True)
    #    tar_links, arxiv_papers = get_tar_links()
    #    get_docs(step_0, tar_links, arxiv_papers)

# Step 1 Extract from .tar and remove irrelevant files
    print("\nStarting Step 1...")
    step_1_target_name = "Step_1"
    step_1 = step1_processing(directory = step_0_target_name, target_name = step_1_target_name)
    step_1.create_target_folder()
    print("Created target folder.")
    step_1.tar_extractor()
    print("Extracted .tar files.")
    step_1.rmv_irrelevant_files(manifest_title="manifest_irrelevant_files.txt")
    print("Removed irrelevant files.")
    #Path("./Processed_files")
    #if os.path.exists(step_1) == False:
    #    os.makedirs(step_1, exist_ok=True)
    #    tar_extractor(step_0, step_1)
    #    rmv_irrelevant_files(step_1)

# step 2 Create main.tex and references.json
    print("\nStarting Step 2...")
    step_2_target_name = "Step_2"
    step_2 = step2_processing(directory = step_1_target_name, target_name = step_2_target_name)
    step_2.create_target_folder()
    print("Created target folder.")
    step_2.merge_tex_files()
    print("Merged tex files.")
    step_2.extract_references()
    print("Extracted references.")
    step_2.remove_bib_from_main()
    print("Removed bibliographies from main.txt files")
    step_2.move_references()
    print("Moved reference files.")
    delete_empty_folders(step_2.target)
    step_2.create_references_json()
    print("Created references.json.")

# step 3 Match references, find citations and create dataset
    
    
    