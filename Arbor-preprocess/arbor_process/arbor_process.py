from gen_filtered_shuffled_chunks import FileProcessor, load_config
from metadata_processor import MetadataProcessor
from get_imgs import GetImages
from gen_img_txt_pair import GenImgTxtPair
from plotting_func import generate_plots
import asyncio

def main():
    # Example calls to the functions; modify these calls based on actual use case
    # get data distribution
    config = load_config('config.json')
    params = config.get('metadata_processor_info', {})
    mp = MetadataProcessor(**params)
    processor.process_all_files()


    # processor.process_single_file('path_to_single_file.parquet')
    config = load_config('config.json')
    params = config.get('metadata_filter_and_shuffle_info', {})
    processor = FileProcessor(**params)
    processor.process_files()
    mp.process_all_files()

    # download images 
    config = load_config('config.json')
    params = config.get('image_download_info', {})
    gi = GetImages(**params)
    asyncio.run(gi.download_images())
    
    # generate text pair and make tar
    config = load_config('config.json')
    params = config.get('img_text_gen_info', {})
    textgen = GenImgTxtPair(**params)
    textgen.create_image_text_pairs()

if __name__ == "__main__":
    main()
