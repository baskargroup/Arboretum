from generate_capped_small_chunks import process_files, merge_shuffled_files
from get_counts_distributions import process_metadata_files
from get_imgs import download_images
from image_text_pair_v2 import create_image_text_pairs

def main():
    # Example calls to the functions; modify these calls based on actual use case
    process_files(
        directory='data_directory',
        species_count_data='species_counts.csv'
    )

    process_metadata_files(
        source='source_directory',
        destination='destination_directory'
    )

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
