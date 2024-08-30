import os
import requests
import zipfile
from tqdm import tqdm

def download_plantvillage_dataset():
    url = "https://data.mendeley.com/public-files/datasets/tywbt9jljs/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded"
    filename = "plantvillage_dataset.zip"
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)
    
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall("plantvillage_dataset")
    
    os.remove(filename)
    print("Dataset downloaded and extracted successfully.")

if __name__ == "__main__":
    download_plantvillage_dataset()