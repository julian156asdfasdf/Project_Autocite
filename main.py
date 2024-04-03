import os
from pathlib import Path
from processing import processing
from datasaet import get_tar_links, get_docs, tar_extractor, rmv_irrelevant_files
from data_count import count_ref


if __name__ == '__main__':
# step 0 Download Data
    step_0 = Path("./papers")
    if os.path.exists(step_0) == False:
        os.makedirs(step_0, exist_ok=True)
        tar_links, arxiv_papers = get_tar_links()
        get_docs(step_0, tar_links, arxiv_papers)
# Step 1 Extract from .tar and remove irrelevant files
    step_1 = Path("./Processed_files")
    if os.path.exists(step_1) == False:
        os.makedirs(step_1, exist_ok=True)
        tar_extractor(step_0, step_1)
        rmv_irrelevant_files(step_1)
# step 2 Create main.tex and references.json
    step_2 = processing(step_1)
    step_2.create_step_2_folder()
    step_2.move_references()
    step_2.extract_references()
    step_2.merge_tex_files()