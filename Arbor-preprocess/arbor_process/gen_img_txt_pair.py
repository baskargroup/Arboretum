import pandas as pd
import os
import json
import glob
import time
import tarfile
import argparse
from concurrent.futures import ThreadPoolExecutor

class GenImgTxtPair:
    """
    Class to generate text labels for the downloaded images.

    Inputs:
        - metadata: Path to the directory containing processed parquet files.
        - img_folder: Path to the directory containing downloaded images in subfolders.
        - output_base_folder: Path to the directory saving the img-text pair data in tar files.

    Outputs:
        - Generate 10 text labels in .txt and .json format for each image and save with each image.
        - Make tar files from each img-text subfolder.

    Example usage:
        config = load_config(args.config)
        params = config.get('img_text_gen_info', {})
        textgen = GenImgTxtPair(**params)
        textgen.create_image_text_pairs()
    """

    def __init__(self, processed_metadata_folder, img_folder, generate_tar = False):
        """
        Initialize the GenImgTxtPair class with metadata, image folder, and output base folder.

        Args:
            metadata (str): Path to the directory containing processed parquet files.
            img_folder (str): Path to the directory containing downloaded images in subfolders.
            generate_tar (bool): Flag to determine whether to generate tar files. Defaults to False
        """
        self.metadata = processed_metadata_folder
        self.img_folder = img_folder
        self.output_base_folder = f'{img_folder}_tar'
        self.generate_tar = generate_tar

    @staticmethod
    def create_files_and_json(output_folder, **kwargs):
        """
        Create text files and a JSON file with taxonomic metadata for each image.

        Args:
            output_folder (str): Path to the folder where the files will be saved.
            kwargs: Dictionary containing the metadata for each image.

        Raises:
            ValueError: If photo_id is not provided in the metadata.
        """
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

    def process_subfolder(self, subfolder_path):
        """
        Process a subfolder to generate text labels and JSON files for each image.

        Args:
            subfolder_path (str): Path to the subfolder containing the images.
        """
        subfolder_name = os.path.basename(subfolder_path)
        parquet_file_path = os.path.join(self.metadata, f"{subfolder_name}.parquet")

        if not os.path.exists(parquet_file_path):
            print(f"Parquet file {parquet_file_path} does not exist.")
            return

        df_s = pd.read_parquet(parquet_file_path)
        start_time = time.time()

        imagelist = [int(os.path.basename(i).split('.')[0]) for i in glob.glob(f'{subfolder_path}/*.jpg')]
        df_s['photo_id'] = df_s['photo_id'].astype(int)
        df = df_s[df_s['photo_id'].isin(imagelist)]
        
        columns = ['photo_id', 'scientificName', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'common_name']
        values = df[columns]
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.create_files_and_json, subfolder_path, **row.to_dict()) for _, row in values.iterrows()]

        for future in futures:
            future.result()

        elapsed_time = time.time() - start_time
        print(f"Text files for {subfolder_name} created in {elapsed_time:.2f} seconds.")

    def create_image_text_pairs(self):
        """
        Generate text labels and JSON files for all images in the specified folder structure.
        """
        subfolders = [os.path.join(self.img_folder, subfolder) for subfolder in os.listdir(self.img_folder) if os.path.isdir(os.path.join(self.img_folder, subfolder))]

        if self.generate_tar:
                os.makedirs(self.output_base_folder, exist_ok=True)

        for subfolder in subfolders:
            subfolder_name = os.path.basename(subfolder)
            self.process_subfolder(subfolder)
            
            if self.generate_tar:
                start_time = time.time()
                output_tar_file = os.path.join(self.output_base_folder, subfolder_name)
                tar_path = f'{output_tar_file}.tar'
                with tarfile.open(tar_path, 'w') as tar:
                    tar.add(subfolder, arcname=os.path.basename(output_tar_file))
                elapsed_time = time.time() - start_time
                print(f"Tar file for {subfolder_name} created: {tar_path} in {elapsed_time:.2f} seconds.")

def load_config(config_path):
    """
    Load configuration from a JSON file.

    Args:
        config_path (str): Path to the configuration JSON file.

    Returns:
        dict: Dictionary containing configuration parameters.
    """
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

def main():
    """
    Main function to load configuration and run the GenImgTxtPair.
    """
    parser = argparse.ArgumentParser(description="Generate text labels.")
    parser.add_argument('--config', type=str, default='../config.json',
                        help='Path to the config file (default: config.json in current directory).')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file '{args.config}' does not exist.")

    config = load_config(args.config)

    params = config.get('img_text_gen_info', {})
    textgen = GenImgTxtPair(**params)
    textgen.create_image_text_pairs()

# Example usage:
# python gen_img_txt_pair.py --config ../config.json

if __name__ == "__main__":
    main()