# README

## Overview
**Before using this script, please download the metadata from Hugging Face.**

This repository contains scripts to generate machine learning-ready image-text pairs in four steps:

1. Processing metadata files to get category and species distribution.
2. Filtering metadata based on user-defined thresholds and generating shuffled chunks.
3. Downloading images based on URLs in the metadata.
4. Generating text labels for the images.

The scripts can be run as Python scripts or from the command line.

## Classes and Their Descriptions

### 1. MetadataProcessor
- **Description**: Processes metadata files in parquet format. Filters the metadata based on categories and counts the number of species and categories. Saves the results in CSV files.
- **Inputs**:
  - `source_folder`: The folder containing the parquet files.
  - `destination_folder`: The folder where the results will be saved.
  - `categories`: A list of categories to filter the metadata.
- **Outputs**:
  - CSV files containing the counts of species and categories.

### 2. GenShuffledChunks
- **Description**: Processes data files by filtering rare cases, capping frequent cases, and shuffling the data into specified parts.
- **Inputs**:
  - `species_count_data`: Path to the species count data file.
  - `directory`: Path to the directory containing the original parquet files.
  - `rare_threshold`: Threshold for rare cases (default: 10).
  - `cap_threshold`: Threshold for frequent cases (default: 1000).
  - `part_size`: Size of each part after shuffling (default: 500).
  - `rare_dir`: Directory to save rare cases (default: 'rare_cases').
  - `cap_filtered_dir_train`: Directory to save capped and filtered cases (default: 'cap_filtered_train').
  - `capped_dir`: Directory to save capped cases (default: 'capped_cases').
  - `merged_dir`: Directory to save merged shuffled files (default: 'merged_cases').
  - `files_per_chunk`: Number of files to merge into a single chunk (default: 5).
  - `random_seed`: Random seed for shuffling (default: 42).
- **Outputs**:
  - Saves rare cases, capped cases, and shuffled parts in specified directories.

### 3. GetImages
- **Description**: Downloads images from URLs stored in parquet files asynchronously.
- **Inputs**:
  - `input_folder`: Path to the folder containing parquet files.
  - `output_folder`: Path to the folder where images will be saved.
  - `start_index`: Index of the first parquet file to process (default: 0).
  - `end_index`: Index of the last parquet file to process (default: None).
  - `concurrent_downloads`: Number of concurrent downloads (default: 1000).
- **Outputs**:
  - Downloads images to `output_folder`.

### 4. GenImgTxtPair
- **Description**: Generates text labels for downloaded images.
- **Inputs**:
  - `metadata`: Path to the directory containing processed parquet files.
  - `img_folder`: Path to the directory containing downloaded images in subfolders.
  - `output_base_folder`: Path to the directory saving the image-text pair data in tar files.
- **Outputs**:
  - Generates 10 text labels in .txt and .json format for each image and saves them with each image.
  - Makes tar files from each image-text subfolder.

## To Use

### Installation
Use any of the following methods:
- Clone the repository and run the script.
- Clone the repository and `pip install .` to use as a package.
- Or `pip install arbor-process` or `conda install arbor-process`.

### 1. Adjust the `config.json` File
- The `config.json` file contains the arguments for different classes. Ensure it is updated with the correct paths and parameters before running the script.

### 2. Running as a Python Script
To run the entire script sequentially, use the provided example script. Comment out any step you do not want to run.

```python
from arbor_process import *
import asyncio
import json

# Load configuration
config = load_config('config.json')

# Step 1: Process metadata
params = config.get('metadata_processor_info', {})
mp = MetadataProcessor(**params)
mp.process_all_files()

# Step 2: Generate shuffled chunks of metadata
params = config.get('metadata_filter_and_shuffle_info', {})
gen_shuffled_chunks = GenShuffledChunks(**params)
gen_shuffled_chunks.process_files()

# Step 3: Download images
params = config.get('image_download_info', {})
gi = GetImages(**params)
asyncio.run(gi.download_images())

# Step 4: Generate text pairs and create tar files (optional)
params = config.get('img_text_gen_info', {})
textgen = GenImgTxtPair(**params)
textgen.create_image_text_pairs()
```

### 3. Running from the Command Line
Update the `config.json` file with the path to the file and use the following commands to run each step individually.

```bash
# Step 1: Process metadata
python arbor_process/metadata_processor.py --config config.json

# Step 2: Generate shuffled chunks of metadata
python arbor_process/gen_filtered_shuffled_chunks.py --config config.json

# Step 3: Download images
python arbor_process/get_imgs.py --config config.json

# Step 4: Generate text pairs and create tar files (optional)
python arbor_process/gen_img_txt_pair.py --config config.json
```

By following these instructions, you can effectively use the script to process metadata, filter and shuffle data files, download images, and generate text labels.