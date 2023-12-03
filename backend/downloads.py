import os
import shutil
import getpass
from tqdm import tqdm

# Constants
KAGGLE_PATH = "backend/kaggle.json"
USERNAME = getpass.getuser()
OS_KAGGLE_DIR = "/Users/" + USERNAME + "/.kaggle"
OS_KAGGLE_FILE = os.path.join(OS_KAGGLE_DIR, "kaggle.json")
DOWNLOAD_PATH = "backend/download"

def setup_kaggle_directory():
    """Set up .kaggle directory and copy the API key."""
    if not os.path.exists(OS_KAGGLE_DIR):
        os.makedirs(OS_KAGGLE_DIR)
    print(f"Copying {KAGGLE_PATH} to {OS_KAGGLE_FILE}")
    shutil.copy2(KAGGLE_PATH, OS_KAGGLE_FILE)

def initialize_kaggle_api():
    """Initialize and authenticate the Kaggle API."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api

def ensure_download_directory():
    """Ensure the download directory exists."""
    if not os.path.exists(DOWNLOAD_PATH):
        os.makedirs(DOWNLOAD_PATH)

def download_datasets(api):
    """Download datasets listed in datasets.txt."""
    with open("backend/datasets.txt", "r") as file:
        datasets = file.readlines()

    for dataset in tqdm(datasets, desc="Downloading datasets", unit="dataset"):
        dataset = dataset.strip()  
        if dataset:
            print(f"\nDownloading {dataset}...")
            api.dataset_download_files(dataset, path=DOWNLOAD_PATH, unzip=True)
            print(f"Downloaded {dataset} successfully!")

def cleanup():
    """Remove kaggle.json after downloading."""
    if os.path.exists(OS_KAGGLE_FILE):
        os.remove(OS_KAGGLE_FILE)

if __name__ == "__main__":
    setup_kaggle_directory()
    api = initialize_kaggle_api()
    ensure_download_directory()
    download_datasets(api)
    cleanup()
    print("\nAll datasets downloaded successfully!")
