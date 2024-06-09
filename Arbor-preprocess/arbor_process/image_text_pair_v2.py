import pandas as pd
import os
import json
import glob
import time
import tarfile
from concurrent.futures import ThreadPoolExecutor

def create_files_and_json(output_folder, **kwargs):
    keys = ['photo_id', 'scientificName', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'common_name']
    data = {key: kwargs.get(key, '') for key in keys}

    if not data['photo_id']:
        raise ValueError("photo_id is required to create files")
        
        
    suffix_to_content = {
    '.com.txt': f"a photo of {data['common_name']}.",
    '.common_name.txt': f"{data['common_name']}.",
    '.sci.txt': f"a photo of {data['scientificName']}.",
    '.sci_com.txt': f"a photo of {data['scientificName']} with common name {data['common_name']}.",
    '.scientific_name.txt': f"{data['scientificName']}.",
    '.taxon.txt': f"a photo of {data['kingdom']} {data['phylum']} {data['class']} {data['order']} {data['family']} {data['genus']} {data['species']}.",
    '.taxonTag.txt': f"a photo of kingdom {data['kingdom']} phylum {data['phylum']} class {data['class']} order {data['order']} family {data['family']} genus {data['genus']} species {data['species']}.",
    '.taxonTag_com.txt': f"a photo of kingdom {data['kingdom']} phylum {data['phylum']} class {data['class']} order {data['order']} family {data['family']} genus {data['genus']} species {data['species']} with common name {data['common_name']}.",
    '.taxon_com.txt': f"a photo of {data['kingdom']} {data['phylum']} {data['class']} {data['order']} {data['family']} {data['genus']} {data['species']} with common name {data['common_name']}.",
    '.taxonomic_name.txt': f"{data['kingdom']} {data['phylum']} {data['class']} {data['order']} {data['family']} {data['genus']} {data['species']}."
}

    # Write all text files
    for suffix, content in suffix_to_content.items():
        file_path = os.path.join(output_folder, f"{data['photo_id']}{suffix}")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

    content = {
        'photo_id': data['photo_id'],
        'text': f"This is a photo of {data['scientificName']}.",
        'photo_of_taxon': f"a photo of {data['kingdom']} {data['phylum']} {data['class']} {data['order']} {data['family']} {data['genus']} {data['species']}.",
        'photo_of_detailed_taxon': f"a photo of kingdom {data['kingdom']} phylum {data['phylum']} class {data['class']} order {data['order']} family {data['family']} genus {data['genus']} species {data['species']}.",
        'photo_of_detailed_taxon_with_common': f"a photo of kingdom {data['kingdom']} phylum {data['phylum']} class {data['class']} order {data['order']} family {data['family']} genus {data['genus']} species {data['species']} with common name {data['common_name']}.",
        'photo_of_taxon_with_common': f"a photo of {data['kingdom']} {data['phylum']} {data['class']} {data['order']} {data['family']} {data['genus']} {data['species']} with common name {data['common_name']}.",
        'taxonomic_name': f"{data['kingdom']} {data['phylum']} {data['class']} {data['order']} {data['family']} {data['genus']} {data['species']}."
    }

    json_path = os.path.join(output_folder, f"{data['photo_id']}.json")
    os.makedirs(output_folder, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(content, f, indent=4)

def process_subfolder(metadata_path,subfolder_path):
    subfolder_name = os.path.basename(subfolder_path)
    parquet_file_path = os.path.join(metadata_path, f"{subfolder_name}.parquet")
    
    if not os.path.exists(parquet_file_path):
        print(f"Parquet file {parquet_file_path} does not exist.")
        return
    
    df_s = pd.read_parquet(parquet_file_path)
    start_time = time.time()
    
    imagelist = [int(os.path.basename(i).split('.')[0]) for i in glob.glob(f'{subfolder_path}/*.jpg')]
    df_s['photo_id'] = df_s['photo_id'].astype(int)
    df = df_s[df_s['photo_id'].isin(imagelist)]
    elapsed_time = time.time() - start_time
    print(f"Metadata filtering for {subfolder_name} completed in {elapsed_time:.2f} seconds.")

    columns = ['photo_id', 'scientificName', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'common_name']
    values = df[columns]
    start_time = time.time()

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(create_files_and_json, subfolder_path, **row.to_dict()) for _, row in values.iterrows()]

    for future in futures:
        future.result()

    elapsed_time = time.time() - start_time
    print(f"Metadata files for {subfolder_name} created in {elapsed_time:.2f} seconds.")

def create_image_text_pairs(metadata = 'metadata', img_folder='img_folder', output_base_folder='output_folder'):
    subfolders = [os.path.join(img_folder, subfolder) for subfolder in os.listdir(img_folder) if os.path.isdir(os.path.join(img_folder, subfolder))]
    
    for subfolder in subfolders:
        subfolder_name = os.path.basename(subfolder)
        output_folder = os.path.join(output_base_folder, subfolder_name)
        process_subfolder(metadata, subfolder)
        
        start_time = time.time()
        tar_path = f'{output_folder}.tar.gz'
        with tarfile.open(tar_path, 'w:gz') as tar:
            tar.add(subfolder, arcname=os.path.basename(output_folder))
        elapsed_time = time.time() - start_time
        print(f"Tar file for {subfolder_name} created: {tar_path} in {elapsed_time:.2f} seconds.")