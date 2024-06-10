from arbor_process import *
import asyncio
import json

# download metadata from hugging face
# update config.json with the path to the file

# get data distribution
config = load_config('config.json')
params = config.get('metadata_processor_info', {})
mp = MetadataProcessor(**params)
mp.process_all_files()

# capped filtered metadata
config = load_config('config.json')
params = config.get('metadata_filter_and_shuffle_info', {})
processor = FileProcessor(**params)
processor.process_files()

# download images 
config = load_config('config.json')
params = config.get('image_download_info', {})
gi = GetImages(**params)
asyncio.run(gi.download_images())

# generate text pair and make tar (optional)
config = load_config('config.json')
params = config.get('img_text_gen_info', {})
textgen = GenImgTxtPair(**params)
textgen.create_image_text_pairs()