from RandomizeKaggleDB import read_json_DB

# Load Kaggle Dataset
KAGGLEDB = read_json_DB(filepath="Randomized_Kaggle_Dataset_Subset_Physics.json")
ARXIV_IDS = list(KAGGLEDB.keys())

if __name__ == '__main__':
    from step0_processing import step0_processing
    from step1_processing import step1_processing
    from step2_processing import step2_processing
    from step3_processing import step3_processing
    from step1_processing import delete_empty_folders

    # step 0 Initialization
    step_0_target_name = "Step_0"
    step_1_target_name = "Step_1"
    step_2_target_name = "Step_2"
    step_3_target_name = "dataset.pkl"
    # Remember to update the start_idx, window_size and end_idx to the desired values
    step_0 = step0_processing(target_name=step_0_target_name, start_idx=57000, window_size=250, end_idx=60000)
    step_1 = step1_processing(directory = step_0_target_name, target_name = step_1_target_name)
    step_2 = step2_processing(directory = step_1_target_name, target_name = step_2_target_name)
    step_3 = step3_processing(directory = step_2_target_name, target_name = step_3_target_name)

    # Create sliding window for step 0-2
    for i in range(step_0.rounds):
    # step 0 Download tar files
        print("Starting round " + str(step_0.round_number) + "...")
        print("Downloading tar files...")
        step_0.round_number += 1
        step_0.create_target_folder()
        step_0.get_tar_links(start_id=step_0.window_size*i+step_0.start_idx)
        step_0.download_tar_files()
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
        step_2.create_references_json()
        print("Created references.json.")
    # End of sliding window
    delete_empty_folders(step_2.target)

    # step 3 Match references, find citations and create dataset
    print("\nStarting Step 3...")
    authors = step_3.create_author_dict()
    print("Author dictionary created.")
    step_3.ref_matcher()
    print("References matched.")
    step_3.build_dataset(update=True, context_size=1000)
    print("Dataset built.")
    