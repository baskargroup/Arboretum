from arbor_process import *

# to do : download metadata from hugging face

# get distribitions from metadata (source full path : /work/mech-ai-scratch/znjubery/Arboretum/metadataChunks_w_common/')
'''
process_metadata_files(source='../metadataChunks_w_common/',
destination='v1/meta_aves_fungi_counts', 
categories= ['Aves','Fungi'])
'''
 
# to do : notebook or gradio or plotly to interactively investigate the data
   
'''
# capped filtered metadata
source = '../metadataChunks_w_common/'
species_count_data = 'v1/meta_aves_fungi_counts/species_group_counts.csv'

process_files(source, species_count_data, rare_threshold=1000, cap_threshold=5000, part_size=500, 
rare_dir='v1/rare_cases', cap_filtered_dir_train='v1/tmp_cap_filtered', 
capped_dir='v1/overthecap_cases', merged_dir='v1/merged_cases', save_config = 'v1/cap_th_chunk_config.yml',
files_per_chunk=10, random_seed=42)
'''
# to do:  add # of species filter
 
# get images
'''
import asyncio
asyncio.run(download_images(
input_folder='v1/merged_cases', 
output_folder='v1/img_txt',
 start_index=0, end_index=2, 
 concurrent_downloads=1000)) 
'''

# generate image_text pair
create_image_text_pairs(metadata='v1/merged_cases', 
img_folder='v1/img_txt',
output_base_folder='v1/img_txt_tar')