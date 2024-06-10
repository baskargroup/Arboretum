import pandas as pd
import os
import aiohttp
import asyncio
import json
from tqdm.asyncio import tqdm
import argparse

class GetImages:
    """
    Class to download images from URLs stored in parquet files asynchronously.

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
    """

    def __init__(self, processed_metadata_folder, output_folder, start_index=0, end_index=None, concurrent_downloads=1000):
        self.input_folder = processed_metadata_folder
        self.output_folder = output_folder
        self.start_index = start_index
        self.end_index = end_index
        self.concurrent_downloads = concurrent_downloads

    @staticmethod
    def list_parquet_files(input_folder: str):
        """
        List all parquet files in the input folder.

        Args:
            input_folder (str): Path to the folder containing parquet files.

        Returns:
            list: List of parquet file paths.
        """
        return [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.parquet')]

    async def download_image(self, session, semaphore, row, output):
        """
        Download an image asynchronously using aiohttp with a semaphore to limit concurrency.

        Args:
            session (aiohttp.ClientSession): The aiohttp client session.
            semaphore (asyncio.Semaphore): Semaphore to limit concurrent downloads.
            row (pd.Series): Row of the DataFrame containing image URL.
            output (str): Path to the folder where images will be saved.

        Returns:
            tuple: Key, file name, and error message if any.
        """
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

    async def process_parquet_file(self, session, semaphore, file_path, output_folder):
        """
        Process a single parquet file and download images.

        Args:
            session (aiohttp.ClientSession): The aiohttp client session.
            semaphore (asyncio.Semaphore): Semaphore to limit concurrent downloads.
            file_path (str): Path to the parquet file.
            output_folder (str): Path to the folder where images will be saved.
        """
        df = pd.read_parquet(file_path)
        subfolder_name = os.path.splitext(os.path.basename(file_path))[0]
        subfolder_path = os.path.join(output_folder, subfolder_name)
        
        tasks = []
        for _, row in df.iterrows():
            task = self.download_image(session, semaphore, row, subfolder_path)
            tasks.append(asyncio.create_task(task))

        errors = 0
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            key, file_name, error = await future
            if error:
                errors += 1
                print(f"Error downloading image {file_name}: {error}")
        print(f"Completed {file_path} with {errors} errors.")

    async def download_images(self):
        """
        Download images from URLs stored in parquet files asynchronously.
        """
        parquet_files = self.list_parquet_files(self.input_folder)
        
        if self.end_index is None:
            self.end_index = len(parquet_files)

        files_to_process = parquet_files[self.start_index:self.end_index]

        semaphore = asyncio.Semaphore(self.concurrent_downloads)

        async with aiohttp.ClientSession() as session:
            for file_path in files_to_process:
                await self.process_parquet_file(session, semaphore, file_path, self.output_folder)

def load_config(config_path):
    """
    Load configuration from a JSON file.

    Args:
        - config_path: Path to the configuration JSON file.

    Returns:
        - config: Dictionary containing configuration parameters.
    """
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

def main():
    """
    Main function to load configuration and run the GetImages class.
    """
    parser = argparse.ArgumentParser(description="Download images from parquet files.")
    parser.add_argument('--config', type=str, default='../config.json',
                        help='Path to the config file (default: ../config.json).')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file '{args.config}' does not exist.")

    config = load_config(args.config)

    params = config.get('image_download_info', {})
    image_downloader = GetImages(**params)
    asyncio.run(image_downloader.download_images())

# Example usage:
# python get_images.py --config ../config.json

if __name__ == "__main__":
    main()