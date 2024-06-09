import pandas as pd
import os
import aiohttp
import asyncio
from tqdm.asyncio import tqdm

def list_parquet_files(input_folder: str):
    """List all parquet files in the input folder."""
    return [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.parquet')]

async def download_image(session, semaphore, row, output):
    """Download an image asynchronously using aiohttp with a semaphore to limit concurrency"""
    async with semaphore:
        key, image_url = row.name, row['photo_url']
        file_name = f"{image_url.split('/')[-2]}.jpg"
        file_path = os.path.join(output, file_name)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        try:
            async with session.get(image_url) as response:
                if response.status == 200:
                    content = await response.read()
                    with open(file_path, 'wb') as f:
                        f.write(content)
                    return key, file_name, None
                else:
                    return key, None, f"HTTP error: {response.status}"
        except Exception as err:
            return key, None, str(err)

async def process_parquet_file(session, semaphore, file_path, output_folder):
    """Process a single parquet file and download images."""
    df = pd.read_parquet(file_path)
    subfolder_name = os.path.splitext(os.path.basename(file_path))[0]
    subfolder_path = os.path.join(output_folder, subfolder_name)
    
    tasks = []
    for _, row in df.iterrows():
        task = download_image(session, semaphore, row, subfolder_path)
        tasks.append(asyncio.create_task(task))

    errors = 0
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        key, file_name, error = await future
        if error:
            errors += 1
            print(f"Error downloading image {file_name}: {error}")
    print(f"Completed {file_path} with {errors} errors.")

async def download_images(input_folder: str, output_folder: str, start_index: int = 0, end_index: int = None, concurrent_downloads: int = 1000):
    parquet_files = list_parquet_files(input_folder)
    
    if end_index is None:
        end_index = len(parquet_files)

    files_to_process = parquet_files[start_index:end_index]

    semaphore = asyncio.Semaphore(concurrent_downloads)

    async with aiohttp.ClientSession() as session:
        for file_path in files_to_process:
            await process_parquet_file(session, semaphore, file_path, output_folder)
