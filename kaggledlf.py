import os
import requests
import zipfile
from tqdm import tqdm

def download_kaggle_dataset():
    kaggle.api.dataset_download_files('mexwell/crop-diseases-classification', path='.', unzip=True)
    print("Dataset downloaded and extracted successfully.")
    get_class_distribution('data')
    
def get_class_distribution(data_dir):
    class_counts = {}
    total_images = 0
    
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            num_images = len([f for f in os.listdir(class_path) if f.endswith('.jpg')])
            class_counts[class_name] = num_images
            total_images += num_images
            
    print("Class distribution:")