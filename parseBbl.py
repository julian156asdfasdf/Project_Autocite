


import re
from typing import Union

from pylatexenc.latex2text import LatexNodes2Text


ACCENT_CONVERTER = LatexNodes2Text()


class BibItem:
    """
    Class to collect the information of a bibliography item in the bbl file.

    The information can be accessed as a dictionary or through the info
    attribute.

    Parameters
    ----------
    bib_string : str
        The block of text of the bibliography item in a bbl file.

    """

    def __init__(self, bib_string: str):
        self.info: dict[str, Union[str, list[str]]] = {"authors": []}
        self.key = ""
        self.bib_id = ""
        self._parse(bib_string)

    def _parse(self, bib_string: str):
        bib_string = (
            bib_string.replace("\n", "").replace("~", " ").replace(r"\newblock", "")
        )
        self.key, self.bib_id, self.text = self._parse_bibitem_key(
            bib_string.split(r"\bibitem")[1]
        )
        #print(self.key, self.bib_id)

        self.info[self.bib_id] = ACCENT_CONVERTER.latex_to_text(self.text)

        if bib_string.split(r"\bibinfo"):
            self._parse_bib_info_lines(bib_string)

    def _parse_bib_info_lines(self, bib_string):    
        bib_info_lines = bib_string.split(r"\bibinfo")[1:]
        if bib_info_lines:
            print('heya')
        for line in bib_info_lines:
            key, value, rest = self._parse_bibinfo(line)
            #print('val_before',value)
            value = value.replace("{", "").replace("}", "") + rest
            #print('val_after',value)
            if key == "author":
                self.info["authors"].append(ACCENT_CONVERTER.latex_to_text(value))
            else:
                self.info[key] = ACCENT_CONVERTER.latex_to_text(value)

    def _parse_bibinfo(self, line):
        print('line', line)
        match = re.match(r"^\[(.*?)\]\{(.*?)\}(.*)$", line, re.DOTALL)
        print('match', match)

        if not match:
            raise IOError(f"Bad bibinfo line: {line}")
        return match.group(1), match.group(2), match.group(3)


    def _parse_bibitem_key(self, line: str) -> tuple[str, str]:
        match = re.search(r"^\[(.*?)\]\{(.*?)\}(.*)$", str(line), re.DOTALL)
        #print(match.group(1), match.group(2))
        if match:
            return match.group(1), match.group(2), match.group(3)
        else:
            raise IOError(f"Bad bibitem line: {line}")

    def __str__(self) -> str:
        final_str = ""
        for key, value in self.info.items():
            if key:
                final_str += "".join(value)
            else:
                final_str += f"{value}"
        # Replace duplicate spaces
        final_str = re.sub(r"\s+", " ", final_str)
        return final_str

    def __repr__(self) -> str:
        return f"BibItem({self.key}, {self.bib_id})"

    def __getitem__(self, key: str) -> Union[str, list[str]]:
        return self.info[key]



class BblFile:
    """
    Class to collect the information of  bibliography items in a .bbl file.

    Parameters
    ----------
    bbl_file : str
        The path to the bbl file.

    Attributes
    ----------
    fname : str
        The path to the bbl file.
    header : str
        The text in the file before the bibliography items.
    bib_items : list[BibItem]
        The bibliography items in the bbl file.

    """

    def __init__(self, fname: str):
        self.fname = fname
        self.header = ""
        self.bib_items: list[BibItem] = []
        self._parse()

    def _parse(self):
        with open(self.fname, "r") as f:
            lines = f.readlines()
        header_lines = []
        bibitem_lines = []
        counter = 0
        for line in lines:
            if line.strip().startswith(r"%") or not line.strip():
                continue
            if not self.header:
                if line.startswith(r"\bibitem"):
                    self.header = "\n".join(header_lines)
                    bibitem_lines.append(line)
                else:
                    header_lines.append(line)
            else:
                if line.strip().startswith(r"\end{thebibliography}"):
                    break
                bibitem_lines.append(line)
        bibitem_text = "\n".join(bibitem_lines)

        #for item in bibitem_text.split(r"\bibitem")[1:]:
            #print('hello', item)
        #print(bibitem_text.split(r"\bibitem")[1])
        self.bib_items = [
            BibItem(r"\bibitem" + item) for item in bibitem_text.split(r"\bibitem")[1:]
        ]
        #print('finished', self.bib_items)

    def __str__(self) -> str:
        final_str = ""
        for index, item in enumerate(self.bib_items, start=1):
            final_str += f"[{index}] {item}\n"
        return final_str

    def __repr__(self) -> str:
        return f"BblFile({self.fname})"









#Virker for den her C:\Users\julia\Desktop\Fagprojekt\Project_Autocite\Processed_files\1008.1024v2\main.bbl




if __name__ == "__main__":
    import sys

    #bib_info = BblFile(sys.argv[1])
    path = r"C:\Users\julia\Desktop\Fagprojekt\Project_Autocite\Processed_files\1008.1024v2\main.bbl"
    bblfile = path


    bib_info = BblFile(bblfile)


    print(bib_info)






'''

import re

text = "[Barabási and Pósfai(2016)]{barabasi2016network}\n\nAlbert-László Barabási and Márton Pósfai.\n\n\\newblock \\emph{Network Science}.\n\n\\newblock Cambridge University Press, Cambridge, 2016."

print(text)

print(ACCENT_CONVERTER.latex_to_text(text))

#%%
match = re.match(r"^\[(.*?)\]\{(.*?)\}(.*)$", text, re.DOTALL)

if match:
    square_bracket_contents = match.group(1)
    curly_bracket_contents = match.group(2)
    rest_of_text = match.group(3)
    print("Square bracket contents:", square_bracket_contents)
    print("Curly bracket contents:", curly_bracket_contents)
    print("Rest of the text:", rest_of_text)
else:
    print("No match found.")





'''

hello = 1.2