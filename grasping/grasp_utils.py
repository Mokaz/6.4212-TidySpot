
import sys
from pathlib import Path

def add_anygrasp_to_path(anygrasp_path):
    anygrasp_path = Path(anygrasp_path)
    
    if not anygrasp_path.is_dir():
        raise NotADirectoryError(f"The path '{anygrasp_path}' is not a valid directory.")
    
    for path in (anygrasp_path, anygrasp_path / 'grasp_detection'):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.append(path_str)
    


