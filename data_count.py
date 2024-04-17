import os
from pathlib import Path

def count_ref(directory):
    """
    Counts the number of .bbl files and .bib in the chosen directory and its subdirectories.

    :param: str, path to the directory

    :returns: tuple, (bbl_count: int, bib_count: int), number of .bbl files and .bib files in the directory and its subdirectories
    """
    bbl_count = 0
    bib_count = 0
    tex_count = 0

    for _, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".bbl"):
                bbl_count += 1
            elif file.endswith(".bib"):
                bib_count += 1
            elif file.endswith(".tex"):
                tex_count += 1

    return bbl_count, bib_count, tex_count

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    directory = current_dir
    directory = 'Step_1'
    bbl_count, bib_count, tex_count = count_ref(directory)
    dir_count = len(os.listdir(directory))
    print(f"Number of .bbl files: {bbl_count}")
    print(f"Number of .bib files: {bib_count}")
    print(f"Number of .tex files: {tex_count}")
    print(f"Number of directories: {dir_count}")