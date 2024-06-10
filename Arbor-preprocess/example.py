from arbor_process import *
import asyncio
import json

# download metadata from hugging face
# update config.json with the path to the file

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