import pandas as pd
import argparse
import json
import os
import glob
from concurrent.futures import ThreadPoolExecutor

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
    
    Example usage:
    processor = MetadataProcessor(source_folder='source_folder', destination_folder='destination_folder', categories=['Aves', 'Arachnida', 'Insecta', 'Plantae', 'Fungi', 'Mollusca', 'Reptilia'])
    processor.process_all_files()
    
    The source_folder is the folder containing the parquet files.
    The destination_folder is the folder where the results will be saved.
    The categories are the list of categories to filter the metadata.
    
    The process_all_files method processes all parquet files in the source folder.
    It uses multithreading to process multiple files concurrently.
    The results are saved in CSV files in the destination folder.
"""


class MetadataProcessor:
    def __init__(self, source_folder='source_folder', destination_folder='destination_folder', categories=None):
        """
        Initialize the MetadataProcessor with source and destination folders and categories.
        """
        if categories is None:
            categories = ['Aves', 'Arachnida', 'Insecta', 'Plantae', 'Fungi', 'Mollusca', 'Reptilia']
        self.source_folder = source_folder
        self.destination_folder = destination_folder
        self.categories = categories

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
        return species_count, counts_df

    def process_single_file(self, file):
        """
        Process a single parquet file and save the results.
        """
        df_parquet = pd.read_parquet(file)
        species_count, counts_df = self._filter_and_count(df_parquet)

        base_name = os.path.basename(file)
        counts_df.to_csv(f"{self.destination_folder}/counts_{base_name}.csv", index=False)
        species_count.to_csv(f"{self.destination_folder}/species_counts_{base_name}.csv", index=False)
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
        counts_files = glob.glob(os.path.join(self.destination_folder, 'counts_*.csv'))
        counts = pd.concat([pd.read_csv(file) for file in counts_files])
        group_counts = counts.groupby('category')['count'].sum().reset_index()
        group_counts.to_csv(f"{self.destination_folder}/group_counts.csv", index=False)

        species_counts_files = glob.glob(os.path.join(self.destination_folder, 'species_counts_*.csv'))
        species_counts = pd.concat([pd.read_csv(file) for file in species_counts_files])
        species_group_counts = species_counts.groupby('species')['count'].sum().reset_index()
        species_group_counts.to_csv(f"{self.destination_folder}/species_group_counts.csv", index=False)
        print("Metadata counts have been processed and saved successfully.")

def main():
    parser = argparse.ArgumentParser(description="Process metadata files.")
    parser.add_argument('--config', type=str, default='../config.json',
                        help='Path to the config file (default: config.json in current directory).')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file '{args.config}' does not exist.")

    # Load configuration from JSON file
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)

    # Access the nested metadata_processor_info
    processor_info = config.get('metadata_processor_info', {})

    source_folder = processor_info.get('source_folder', 'source_directory')
    destination_folder = processor_info.get('destination_folder', 'destination_directory')
    categories = processor_info.get('categories', None)

    processor = MetadataProcessor(
        source_folder=source_folder,
        destination_folder=destination_folder,
        categories=categories
    )
    processor.process_all_files()

# Example usage:
# python metadata_processor.py --config config.json

if __name__ == "__main__":
    main()
