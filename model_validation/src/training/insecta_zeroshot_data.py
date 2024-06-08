import numpy as np
import ast
from pathlib import Path
import pandas as pd

try:
    insecta_path = Path("./metadata/insecta_classes.txt")
    with open( insecta_path, 'r' ) as file:
        insecta_classnames = ast.literal_eval( file.read( ) )
except:
    insecta_path = Path("vlhub/metadata/insecta_classes.txt")
    with open( insecta_path, 'r' ) as file:
        insecta_classnames = ast.literal_eval( file.read( ) )

try:
    insecta_id_path = Path("./metadata/insecta_id_map.txt")
    with open( insecta_id_path, 'r' ) as file:
        insecta_id_dict = ast.literal_eval( file.read( ) )
except:
    insecta_id_path = Path("vlhub/metadata/insecta_id_map.txt")
    with open( insecta_id_path, 'r' ) as file:
        insecta_id_dict = ast.literal_eval( file.read( ) )

def get_insecta_classnames():
    return insecta_classnames

def get_insecta_id_dict():
    return insecta_id_dict

try:
    arbor_rare_path = Path("./metadata/arboretum_rare_combined_metadata.csv")
    arbor_rare_df = pd.read_csv(arbor_rare_path)
except:
    arbor_rare_path = Path("vlhub/metadata/arboretum_rare_combined_metadata.csv")
    arbor_rare_df = pd.read_csv(arbor_rare_path)

def get_arboretum_rare_classes(taxon):
    return list(arbor_rare_df[taxon].unique())

try:
    arbor_test_path = Path("./metadata/arboretum_test_metadata.csv")
    arbor_test_df = pd.read_csv(arbor_test_path)
except:
    arbor_test_path = Path("vlhub/metadata/arboretum_test_metadata.csv")
    arbor_test_df = pd.read_csv(arbor_test_path)

def get_arboretum_test_classes(taxon):
    return list(arbor_test_df[taxon].unique())

try:
    bioclip_rare_path = Path("./metadata/bioclip_rare_metadata_n.csv")
    bioclip_rare_df = pd.read_csv(bioclip_rare_path)
except:
    bioclip_rare_path = Path("vlhub/metadata/bioclip_rare_metadata_n.csv")
    bioclip_rare_df = pd.read_csv(bioclip_rare_path)

def get_bioclip_rare_classes(taxon):
    return list(bioclip_rare_df[taxon].unique())

try:
    fungi_path = Path("./metadata/fungi_metadata_n.csv")
    fungi_df = pd.read_csv(fungi_path)
except:
    fungi_path = Path("vlhub/metadata/fungi_metadata_n.csv")
    fungi_df = pd.read_csv(fungi_path)

def get_fungi_classes():
    return list(fungi_df["class"].unique())

#/scratch/bf996/ArborCLIP_Benchmarking/metadata/ins2_metadata_n.csv

try:
    ins2_path = Path("./metadata/ins2_metadata_n.csv")
    ins2_df = pd.read_csv(ins2_path)
except:
    ins2_path = Path("vlhub/metadata/ins2_metadata_n.csv")
    ins2_df = pd.read_csv(ins2_path)

def get_ins2_classes():
    return list(ins2_df["class"].unique())