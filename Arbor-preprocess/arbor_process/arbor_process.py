from generate_capped_small_chunks import process_files, merge_shuffled_files
from metadata_processor import MetadataProcessor
from get_imgs import download_images
from image_text_pair_v2 import create_image_text_pairs
import asyncio

def main():
    # Example calls to the functions; modify these calls based on actual use case
    process_files(
        directory='data_directory',
        species_count_data='species_counts.csv'
    )

    processor = MetadataProcessor(
        source_folder='path_to_source_directory',
        destination_folder='path_to_destination_directory'
    )
    # processor.process_single_file('path_to_single_file.parquet')
    processor.process_all_files()

    asyncio.run(download_images(
        input_folder='input_directory',
        output_folder='output_directory'
    ))

    create_image_text_pairs(
        metadata='metadata.parquet',
        img_folder='image_folder',
        output_folder='output_directory'
    )

if __name__ == "__main__":
    main()
