Metadata Processor:
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
