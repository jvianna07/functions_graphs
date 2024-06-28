import os
from pathlib import PurePath

def setrootdir(root: str):
    current_absolute_path: list = os.getcwd().split(str(PurePath("/")))

    if root in current_absolute_path:
        while current_absolute_path[-1] != root:
            os.chdir(PurePath("../"))
            current_absolute_path.pop()
        return True
    return False
