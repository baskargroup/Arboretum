import pandas as pd
import argparse
import json
import os
import glob
from concurrent.futures import ThreadPoolExecutor
from .plotting_func import generate_plots 

"""
    MetadataProcessor is a class that processes metadata files in parquet format.
    It filters the metadata based on categories and counts the number of species and categories.
    The results are saved in CSV files.
    
    inputs:
    - source_folder: the folder containing the parquet files
    - destination_folder: the folder where the results will be saved
    - categories: a list of categories to filter the metadata
    
    outputs:
    - CSV files containing the counts of species and categories
    
    in config.json:
    "metadata_processor_info": {
        "source_folder": "Dev_Folders/metadataChunks_w_common/",
        "destination_folder": "Dev_Folders/data_v0/",
        "categories": ["Aves", "Arachnida", "Insecta", "Plantae", "Fungi", "Mollusca", "Reptilia"]
    }
    Example usage:
    
    config = load_config(args.config)
    params = config.get('metadata_processor_info', {})
    processor = MetadataProcessor(**params)
    processor.process_all_files()

    The source_folder is the folder containing the parquet files.
    The destination_folder is the folder where the results will be saved.
    The categories are the list of categories to filter the metadata.
    
    The process_all_files method processes all parquet files in the source folder.
    It uses multithreading to process multiple files concurrently.
    The results are saved in CSV files in the destination folder.
"""


class MetadataProcessor:
    def __init__(self,source_folder,destination_folder,categories = None):
        """
        Initialize the MetadataProcessor with source and destination folders and categories.
        """
        self.source_folder = source_folder
        self.destination_folder = destination_folder
        self.categories = categories
        if self.categories is None:
            self.categories = ['Aves', 'Arachnida', 'Insecta', 'Plantae', 'Fungi', 'Mollusca', 'Reptilia']
        

    @property
    def source_folder(self):
        return self._source_folder

    @source_folder.setter
    def source_folder(self, value):
        '''
        Check if the source folder exists and has parquet files.
        '''
        if not isinstance(value, str):
            raise ValueError("Source folder path must be a string.")
        if not os.path.exists(value):
            raise FileNotFoundError(f"Source folder '{value}' does not exist.")
        if not os.path.isdir(value):
            raise NotADirectoryError(f"Source folder '{value}' is not a directory.")
        if not glob.glob(os.path.join(value, '*.parquet')):
            raise FileNotFoundError(f"No parquet files found in the source folder '{value}'.")
        self._source_folder = value

    @property
    def destination_folder(self):
        return self._destination_folder

    @destination_folder.setter
    def destination_folder(self, value):
        '''
        Check if the destination folder exists and create it if it does not exist.
        '''
        if not os.path.exists(value):
            print(f"Destination folder '{value}' does not exist. Creating it now.")
            os.makedirs(value, exist_ok=True)

        if not isinstance(value, str):
            raise ValueError("Destination folder path must be a string.")
        self._destination_folder = value

    @property
    def categories(self):
        return self._categories

    @categories.setter
    def categories(self, value):
        '''
        Categories must be a list of strings.
        '''
        if value is None:
            value = ['Aves', 'Arachnida', 'Insecta', 'Plantae', 'Fungi', 'Mollusca', 'Reptilia']
        if not isinstance(value, list):
            raise ValueError("Categories must be a list of strings.")
        if not all(isinstance(category, str) for category in value):
            raise ValueError("Categories must be a list of strings.")
        self._categories = value


    def _filter_and_count(self, df_parquet):
        """
        Filter the dataframe and count species and categories.
        """
        category_filter = (
                (df_parquet['class'].isin(self.categories)) |
                (df_parquet['kingdom'].isin(self.categories)) |
                (df_parquet['phylum'].isin(self.categories))
        )
        df_filtered = df_parquet[category_filter]
        
        species_count = df_filtered['species'].value_counts().reset_index()
        species_count.columns = ['species', 'count']

        
        counts = {category: len(df_filtered[df_filtered[category_type] == category])
                  for category, category_type in [('Aves', 'class'), ('Arachnida', 'class'),
                                                  ('Insecta', 'class'), ('Plantae', 'kingdom'),
                                                  ('Fungi', 'kingdom'), ('Mollusca', 'phylum'),
                                                  ('Reptilia', 'class')] if category in self.categories}
        counts_df = pd.DataFrame(counts.items(), columns=['category', 'count'])

        df_filtered['category'] = df_filtered.apply(
            lambda row: next((category for category, category_type in [('Aves', 'class'), ('Arachnida', 'class'),
                                                                        ('Insecta', 'class'), ('Plantae', 'kingdom'),
                                                                        ('Fungi', 'kingdom'), ('Mollusca', 'phylum'),
                                                                        ('Reptilia', 'class')] if row[category_type] == category), 'Other'),
            axis=1)
        
        # Get the species count within each category
        species_count_within_category = (
            df_filtered.groupby(['category', 'species'])
            .size()
            .reset_index(name='count')
        )
             
        return species_count_within_category

    def process_single_file(self, file):
        """
        Process a single parquet file and save the results.
        """
        df_parquet = pd.read_parquet(file)
        species_count = self._filter_and_count(df_parquet)

        base_name = os.path.basename(file)
        species_count.to_csv(f"{self.destination_folder}/{base_name}_sample_counts_per_species.csv", index=False)
        print(f"Processed file: {file}")

    def process_all_files(self):
        """
        Process all parquet files in the source folder.
        """
        parquet_files = glob.glob(os.path.join(self.source_folder, '*.parquet'))

        with ThreadPoolExecutor() as executor:
            executor.map(self.process_single_file, parquet_files)

        self._combine_results()

    def _combine_results(self):
        """
        Combine results from individual files into summary files.
        """
        species_counts_files = glob.glob(os.path.join(self.destination_folder, '*_sample_counts_per_species.csv'))
        species_counts = pd.concat([pd.read_csv(file) for file in species_counts_files])
        species_group_counts = species_counts.groupby(['category','species'])['count'].sum().reset_index()
        species_group_counts.to_csv(f"{self.destination_folder}/combined_sample_counts_per_species.csv", index=False)
        
        generate_plots(f"{self.destination_folder}/combined_sample_counts_per_species.csv"
                      ,f"{self.destination_folder}", min_threshold = 30,max_threshold = 1000)
                      
        print("Metadata counts have been processed and saved successfully.")

def load_config(config_path):
    """
    Load configuration from a JSON file.

    Args:
        - config_path: Path to the configuration JSON file.

    Returns:
        - config: Dictionary containing configuration parameters.
    """
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

def main():
    parser = argparse.ArgumentParser(description="Process metadata files.")
    parser.add_argument('--config', type=str, default='../config.json',
                        help='Path to the config file (default: config.json in current directory).')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file '{args.config}' does not exist.")

    config = load_config(args.config)
    
    params = config.get('metadata_processor_info', {})

    processor = MetadataProcessor(**params)
    processor.process_all_files()

# Example usage:
# python metadata_processor.py --config config.json

if __name__ == "__main__":
    main()