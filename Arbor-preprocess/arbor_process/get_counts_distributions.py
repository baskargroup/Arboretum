import pandas as pd
import os
import glob
from concurrent.futures import ThreadPoolExecutor

def process_file(file, destination='destination_folder', categories=None):
    if categories is None:
        categories = ['Aves', 'Arachnida', 'Insecta', 'Plantae', 'Fungi', 'Mollusca', 'Reptilia']
    
    df_parquet = pd.read_parquet(file)
    category_filter = (
        (df_parquet['class'].isin(categories)) |
        (df_parquet['kingdom'].isin(categories)) |
        (df_parquet['phylum'].isin(categories))
    )
    df_filtered = df_parquet[category_filter]
    species_count = df_filtered['species'].value_counts().reset_index()
    species_count.columns = ['species', 'count']
    
    counts = {category: len(df_filtered[df_filtered[category_type] == category])
              for category, category_type in [('Aves', 'class'), ('Arachnida', 'class'), 
                                              ('Insecta', 'class'), ('Plantae', 'kingdom'), 
                                              ('Fungi', 'kingdom'), ('Mollusca', 'phylum'), 
                                              ('Reptilia', 'class')] if category in categories}
    counts_df = pd.DataFrame(counts.items(), columns=['category', 'count'])

    base_name = os.path.basename(file)
    counts_df.to_csv(f"{destination}/counts_{base_name}.csv", index=False)
    species_count.to_csv(f"{destination}/species_counts_{base_name}.csv", index=False)

def process_metadata_files(source='source_folder', destination='destination_folder', categories=None):
    if categories is None:
        categories = ['Aves', 'Arachnida', 'Insecta', 'Plantae', 'Fungi', 'Mollusca', 'Reptilia']
    
    os.makedirs(destination, exist_ok=True)
    parquet_files = glob.glob(os.path.join(source, '*.parquet'))

    with ThreadPoolExecutor() as executor:
        executor.map(lambda file: process_file(file, destination, categories), parquet_files)

    counts_files = glob.glob(os.path.join(destination, 'counts_*.csv'))
    counts = pd.concat([pd.read_csv(file) for file in counts_files])
    group_counts = counts.groupby('category')['count'].sum().reset_index()
    group_counts.to_csv(f"{destination}/group_counts.csv", index=False)

    species_counts_files = glob.glob(os.path.join(destination, 'species_counts_*.csv'))
    species_counts = pd.concat([pd.read_csv(file) for file in species_counts_files])
    species_group_counts = species_counts.groupby('species')['count'].sum().reset_index()
    species_group_counts.to_csv(f"{destination}/species_group_counts.csv", index=False)

    print("Metadata counts have been processed and saved successfully.")
