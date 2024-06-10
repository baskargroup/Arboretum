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

File Processor:
    FileProcessor is a class to process data files. Filters rare cases, caps frequent cases, and shuffles the data into specified parts.

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
    processor = FileProcessor(**config)
    params = config.get('metadata_filter_and_shuffle_info', {})
    processor = FileProcessor(**params)
    processor.process_files()


GetImage:
    GetImage is a class to download images from URLs stored in parquet files asynchronously.

    Inputs:
        - input_folder: Path to the folder containing parquet files.
        - output_folder: Path to the folder where images will be saved.
        - start_index: Index of the first parquet file to process (default: 0).
        - end_index: Index of the last parquet file to process (default: None).
        - concurrent_downloads: Number of concurrent downloads (default: 1000).

    Example usage:
        config = load_config(args.config)
        params = config.get('image_download_info', {})
        image_downloader = GetImages(**params)
        asyncio.run(image_downloader.download_images())

GenImgTxtPair:
    GenImgTxtPair is a class to generate text labels for the downloaded images.

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
    


