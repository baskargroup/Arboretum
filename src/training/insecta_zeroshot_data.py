import numpy as np
import ast
from pathlib import Path

try:
    insecta_path = Path("./metadata/insecta_classes.txt")
    with open( insecta_path, 'r' ) as file:
        insecta_classnames = ast.literal_eval( file.read( ) )
except:
    insecta_path = Path("vlhub/metadata/insecta_classes.txt")
    with open( insecta_path, 'r' ) as file:
        insecta_classnames = ast.literal_eval( file.read( ) )

def get_insecta_classnames():
    return insecta_classnames