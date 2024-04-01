import os
from pathlib import Path
from processing import processing
from datasaet import get_tar_links, get_docs, tar_extractor, rmv_irrelevant_files
from data_count import count_ref

# step 0
if __name__ == '__main__':
    step_0_1 = Path("./papers")
    step_0_2 = Path("./Processed_files")
    
    if os.path.exists(step_0_1) == False:
        os.makedirs(step_0_1, exist_ok=True)
        tar_links, arxiv_papers = get_tar_links()
        get_docs(step_0_1, tar_links, arxiv_papers)

    if os.path.exists(step_0_2) == False:
        os.makedirs(step_0_2, exist_ok=True)
        tar_extractor(step_0_1, step_0_2)
        rmv_irrelevant_files(step_0_2)
# step 1
    step_1 = processing(step_0_2)
    step_1.create_step_1_folder()
    step_1.move_references()
    step_1.extract_references()
    step_1.merge_tex_files()