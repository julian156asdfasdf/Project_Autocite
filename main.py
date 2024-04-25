from step0_processing import step0_processing
from step1_processing import step1_processing
from step2_processing import step2_processing
from step1_processing import delete_empty_folders
from RandomizeKaggleDB import randomizeKaggleDB, read_and_shuffle_KaggleDB, read_KaggleDB_Subset

import random

if __name__ == '__main__':
    # step 0 Initialization
    # Load Kaggle Dataset
    KaggleDB = read_KaggleDB_Subset(filepath="Randomized_Kaggle_Dataset_Subset_Physics.json")

    step_0_target_name = "Step_0"
    step_1_target_name = "Step_1"
    step_2_target_name = "Step_2"
    # Remember to update the start_idx, window_size and end_idx to the desired values
    step_0 = step0_processing(KaggleDB=KaggleDB, target_name=step_0_target_name, start_idx=0, window_size=50, end_idx=200)
    step_1 = step1_processing(directory = step_0_target_name, target_name = step_1_target_name)
    step_2 = step2_processing(directory = step_1_target_name, target_name = step_2_target_name)

    # Create sliding window for step 0-2
    for i in range(step_0.rounds):
    # step 0 Download tar files
        print("Starting round " + str(step_0.round_number) + "...")
        print("Downloading tar files...")
        step_0.round_number += 1
        step_0.create_target_folder()
        step_0.get_tar_links(start_id=step_0.window_size*i+step_0.start_idx)
        step_0.get_docs()
        print("Downloaded tar files successfully.")

    # Step 1 Extract from .tar and remove irrelevant files
        print("\nStarting Step 1...")
        step_1.create_target_folder()
        print("Created target folder.")
        step_1.tar_extractor()
        print("Extracted .tar files.")
        step_1.rmv_irrelevant_files(manifest_title="manifest_irrelevant_files.txt")
        print("Removed irrelevant files.")

    # step 2 Create main.tex and references.json
        print("\nStarting Step 2...")
        step_2.create_target_folder()
        print("Created target folder.")
        step_2.create_main_txt()
        print("Created main.txt.")
        # step_2.create_references_json()
        # print("Created references.json.")
    # End of sliding window
    delete_empty_folders(step_2.target)

    # step 3 Match references, find citations and create dataset
    
    
    