from arbor_process import *
import json

# to do : download metadata from hugging face

# get distribitions from metadata (source full path : /work/mech-ai-scratch/znjubery/2024/Arboretum/Arbor-preprocess/Dev_Folders/metadataChunks_w_common/')
'''
source = '/work/mech-ai-scratch/nirmal/bio_clip/git/Arboretum/Arbor-preprocess/Dev_Folders/metadataChunks_w_common/'
destination = '/work/mech-ai-scratch/nirmal/bio_clip/git/Arboretum/Arbor-preprocess/Dev_Folders/outputs/'
processor = MetadataProcessor(source, destination)
processor.process_all_files()
'''
 
# to do : notebook or gradio or plotly to interactively investigate the data
   
# capped filtered metadata
config = load_config('config.json')
processor = FileProcessor(**config)
params = config.get('metadata_filter_and_shuffle_info', {})
processor = FileProcessor(**params)
processor.process_files()

'''

# to do:  add # of species filter
'''

''' 
# get images

import asyncio
asyncio.run(download_images(
input_folder='Dev_Folders/data_v0/merged_cases', 
output_folder='Dev_Folders/data_v0/img_txt',
 start_index=0, end_index=2, 
 concurrent_downloads=1000)) 
'''

# generate image_text 
'''
create_image_text_pairs(metadata='Dev_Folders/data_v0/merged_cases', 
img_folder='Dev_Folders/data_v0/img_txt',
output_base_folder='Dev_Folders/data_v0/img_txt_tar')
'''
