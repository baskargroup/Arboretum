import pandas as pd
import os
import shutil
import glob
import random
import time
from tqdm import tqdm
import yaml


def save_config_to_yaml(config, filename):
    with open(filename, 'w') as file:
        yaml.dump(config, file)

def process_files(directory, species_count_data, rare_threshold=10, cap_threshold=1000, part_size=500, rare_dir='rare_cases', cap_filtered_dir_train='cap_filtered_train', capped_dir='capped_cases', merged_dir='merged_cases', files_per_chunk=5, random_seed=42, save_config = 'cap_th_chunk_config.yml'):
    
    config = {
        'directory': directory,
        'species_count_data': species_count_data,
        'rare_threshold': rare_threshold,
        'cap_threshold': cap_threshold,
        'part_size': part_size,
        'rare_dir': rare_dir,
        'cap_filtered_dir_train': cap_filtered_dir_train,
        'capped_dir': capped_dir,
        'merged_dir': merged_dir,
        'files_per_chunk': files_per_chunk,
        'random_seed': random_seed,
        'save_config': save_config
    }
    
    save_config_to_yaml(config, save_config)
    
    start_time = time.time()
    final_counts = pd.read_csv(species_count_data)
    rare_case = set(final_counts[final_counts['count'] < rare_threshold]['species'])
    frequent_case = set(final_counts[final_counts['count'] > cap_threshold]['species'])

    for dir_path in [rare_dir, cap_filtered_dir_train, capped_dir, merged_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)

    frequent_counts = {}
    capped_cases = []
    files = [f for f in os.listdir(directory) if f.endswith(".parquet")]
    
    for filename in tqdm(files, desc="Processing files"):
        filepath = os.path.join(directory, filename)
        df = pd.read_parquet(filepath)

        rare_df = df[df['species'].isin(rare_case)]
        capped_filtered_df = df[~df['species'].isin(rare_case)]

        rare_df.to_parquet(os.path.join(rare_dir, filename), index=False)
        
        
        frequent_df = capped_filtered_df[capped_filtered_df['species'].isin(frequent_case)]
        frequent_case_counts = frequent_df['species'].value_counts().to_dict()
        
        for case, count in frequent_case_counts.items():
            frequent_counts[case] = frequent_counts.get(case, 0) + count
            if frequent_counts[case] > cap_threshold and case not in capped_cases:
                capped_cases.append(case)
                cap_case_df = frequent_df[frequent_df['species'] == case]
                cap_case_df.to_parquet(os.path.join(capped_dir, f'capped_{case}.parquet'), index=False)
                
                
        # Filter out capped cases
        capped_df = capped_filtered_df[~capped_filtered_df['species'].isin(capped_cases)]

        # Shuffle the DataFrame randomly
        df_shuffled = capped_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # Calculate the number of rows for each part
        num_parts = max(1, round(len(df_shuffled) / part_size))
        rows_per_part = len(df_shuffled) // num_parts

        # Split the DataFrame into equal parts
        df_parts = [df_shuffled.iloc[i * rows_per_part: (i + 1) * rows_per_part] for i in range(num_parts)]

        # If there are any remaining rows due to rounding, add them to the last part
        if len(df_shuffled) % num_parts != 0:
            df_parts[-1] = pd.concat([df_parts[-1], df_shuffled.iloc[num_parts * rows_per_part:]], ignore_index=True)

        # Remove the .parquet extension from the filename
        base_filename, _ = os.path.splitext(filename)

        # Save each part to a separate Parquet file
        for i, part in enumerate(df_parts):
            cap_filtered_filepath = os.path.join(cap_filtered_dir_train, f'{base_filename}_part{i+1}.parquet')
            part.to_parquet(cap_filtered_filepath, index=False)
        

    merge_shuffled_files(cap_filtered_dir_train, files_per_chunk, merged_dir, random_seed)
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds.")

def merge_shuffled_files(directory, files_per_chunk, merged_dir, random_seed):
    start_time = time.time()
    all_files = sorted(glob.glob(os.path.join(directory, "*.parquet")))
    random.seed(random_seed)
    random.shuffle(all_files)

    for i in tqdm(range(0, len(all_files), files_per_chunk), desc="Merging files"):
        chunk_files = all_files[i:i + files_per_chunk]
        merged_df = pd.concat([pd.read_parquet(file) for file in chunk_files], ignore_index=True)
        chunk_filename = os.path.join(merged_dir, f'chunks_{i // files_per_chunk + 1:04d}.parquet')
        merged_df.to_parquet(chunk_filename, index=False)
        print(f'Saved merged chunk: {chunk_filename}')
    
    elapsed_time = time.time() - start_time
    print(f"Merging completed in {elapsed_time:.2f} seconds.")
