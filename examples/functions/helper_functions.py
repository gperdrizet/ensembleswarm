'''Helper functions for example/demonstration notebooks.'''

import os
import re

def delete_old_logs(directory:str, basename:str) -> None:
    '''Deletes old log files from previous optimization runs on the
    same dataset.'''

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if re.search(basename, filename):
            os.remove(file_path)
