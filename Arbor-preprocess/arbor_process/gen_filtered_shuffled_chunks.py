import pandas as pd
import os
import shutil
import glob
import random
import time
import json
import argparse
from tqdm import tqdm

class GenShuffledChunks:
    """
    Class to process data files. Filters rare cases, caps frequent cases, and shuffles the data into specified parts.

    inputs:
        - species_count_data: Path to the species count data file.
        - directory: Path to the directory containing the original parquet files.
        - rare_threshold: Threshold for rare cases (default: 10). If any species has less than this count, it will be considered rare.
        - cap_threshold: Threshold for frequent cases (default: 1000). If any species has more than this count, it will be capped.
        - part_size: Size of each part after shuffling (default: 500). The data will be shuffled and split into parts of this size.
        - rare_dir: Directory to save rare cases (default: 'rare_cases'). 
        - cap_filtered_dir_train: Directory to save capped and filtered cases (default: 'cap_filtered_train').
        - capped_dir: Directory to save capped cases (default: 'capped_cases').
        - merged_dir: Directory to save merged shuffled files (default: 'merged_cases').
        - files_per_chunk: Number of files to merge into a single chunk (default: 5).
        - random_seed: Random seed for shuffling (default: 42).
    
    outputs:
        - Saves rare cases, capped cases, and shuffled parts in specified directories.

    in config.json:
        "metadata_filter_and_shuffle_info": {
        "species_count_data": "path/to/species_counts.csv",
        "directory": "path/to/data",
        "rare_threshold": 10,
        "cap_threshold": 12,
        "part_size": 50,
        "rare_dir": "path/to/rare_cases",
        "cap_filtered_dir_train": "path/to/cap_filtered_train",
        "capped_dir": "path/to/capped_cases",
        "merged_dir": "path/to/merged_cases",
        "files_per_chunk": 10,
        "random_seed": 42
    }
        
    Example usage:
    config = load_config('config.json')
    params = config.get('metadata_filter_and_shuffle_info', {})
    gen_shuffled_chunks = GenShuffledChunks(**params)
    gen_shuffled_chunks.process_files()

    """
    
    def __init__(self, **kwargs):
        self.species_count_data = kwargs.get('species_count_data')
        self.directory = kwargs.get('directory')
        self.rare_threshold = kwargs.get('rare_threshold', 10)
        self.cap_threshold = kwargs.get('cap_threshold', 1000)
        self.part_size = kwargs.get('part_size', 500)
        self.rare_dir = kwargs.get('rare_dir', 'rare_cases')
        self.cap_filtered_dir_train = kwargs.get('cap_filtered_dir_train', 'cap_filtered_train')
        self.capped_dir = kwargs.get('capped_dir', 'capped_cases')
        self.merged_dir = kwargs.get('merged_dir', 'merged_cases')
        self.files_per_chunk = kwargs.get('files_per_chunk', 5)
        self.random_seed = kwargs.get('random_seed', 42)

    def process_files(self):
        """
        Process files based on configuration parameters. Filters rare cases,
        caps frequent cases, and shuffles the data into specified parts.
        """
        start_time = time.time()
        
        final_counts = pd.read_csv(self.species_count_data)
        rare_case = set(final_counts[final_counts['count'] < self.rare_threshold]['species'])
        frequent_case = set(final_counts[final_counts['count'] > self.cap_threshold]['species'])

        for dir_path in [self.rare_dir, self.cap_filtered_dir_train, self.capped_dir, self.merged_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path, exist_ok=True)

        frequent_counts = {}
        capped_cases = []
        files = [f for f in os.listdir(self.directory) if f.endswith(".parquet")]

        for filename in tqdm(files, desc="Processing files"):
            filepath = os.path.join(self.directory, filename)
            df = pd.read_parquet(filepath)

            rare_df = df[df['species'].isin(rare_case)]
            capped_filtered_df = df[~df['species'].isin(rare_case)]

            rare_df.to_parquet(os.path.join(self.rare_dir, filename), index=False)

            frequent_df = capped_filtered_df[capped_filtered_df['species'].isin(frequent_case)]
            frequent_case_counts = frequent_df['species'].value_counts().to_dict()

            for case, count in frequent_case_counts.items():
                frequent_counts[case] = frequent_counts.get(case, 0) + count
                if frequent_counts[case] > self.cap_threshold and case not in capped_cases:
                    capped_cases.append(case)
                    cap_case_df = frequent_df[frequent_df['species'] == case]
                    cap_case_df.to_parquet(os.path.join(self.capped_dir, f'capped_{case}.parquet'), index=False)

            capped_df = capped_filtered_df[~capped_filtered_df['species'].isin(capped_cases)]
            df_shuffled = capped_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
            num_parts = max(1, round(len(df_shuffled) / self.part_size))
            rows_per_part = len(df_shuffled) // num_parts

            df_parts = [df_shuffled.iloc[i * rows_per_part: (i + 1) * rows_per_part] for i in range(num_parts)]

            if len(df_shuffled) % num_parts != 0:
                df_parts[-1] = pd.concat([df_parts[-1], df_shuffled.iloc[num_parts * rows_per_part:]], ignore_index=True)

            base_filename, _ = os.path.splitext(filename)
            for i, part in enumerate(df_parts):
                cap_filtered_filepath = os.path.join(self.cap_filtered_dir_train, f'{base_filename}_part{i+1}.parquet')
                part.to_parquet(cap_filtered_filepath, index=False)

        self.merge_shuffled_files()
        elapsed_time = time.time() - start_time
        print(f"Processing completed in {elapsed_time:.2f} seconds.")

    def merge_shuffled_files(self):
        """
        Merge the shuffled files into larger chunks.
        """
        start_time = time.time()
        all_files = sorted(glob.glob(os.path.join(self.cap_filtered_dir_train, "*.parquet")))
        random.seed(self.random_seed)
        random.shuffle(all_files)

        for i in tqdm(range(0, len(all_files), self.files_per_chunk), desc="Merging files"):
            chunk_files = all_files[i:i + self.files_per_chunk]
            merged_df = pd.concat([pd.read_parquet(file) for file in chunk_files], ignore_index=True)
            chunk_filename = os.path.join(self.merged_dir, f'processed_chunks_{i // self.files_per_chunk + 1:04d}.parquet')
            merged_df.to_parquet(chunk_filename, index=False)
            print(f'Saved merged chunk: {chunk_filename}')

        elapsed_time = time.time() - start_time
        print(f"Merging completed in {elapsed_time:.2f} seconds.")

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
    """
    Main function to load configuration and run the FileProcessor.
    """
    parser = argparse.ArgumentParser(description="Process and shuffle data files.")
    parser.add_argument('--config', type=str, default='../config.json',
                        help='Path to the config file (default: config.json in current directory).')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file '{args.config}' does not exist.")
    
    config = load_config(args.config)
    
    params = config.get('metadata_filter_and_shuffle_info', {})
    gen_shuffled_chunks = GenShuffledChunks(**params)
    gen_shuffled_chunks.process_files()

# Example usage:
# python gen_filtered_shuffled_chunks.py --config ../config.json

if __name__ == "__main__":
    main()
