import os
import shutil
import getpass
from tqdm import tqdm

# Constants
KAGGLE_PATH = "backend/kaggle.json"  # Path to the local Kaggle API key
USERNAME = getpass.getuser()  # Fetch the current username of the system user
OS_KAGGLE_DIR = "/Users/" + USERNAME + "/.kaggle"  # Path to store Kaggle API key
OS_KAGGLE_FILE = os.path.join(OS_KAGGLE_DIR, "kaggle.json")  # Full path for Kaggle API key
DOWNLOAD_PATH = "backend/download"  # Directory to download Kaggle datasets

def setup_kaggle_directory():
    """
    Set up the .kaggle directory in the user's home directory and copy the API key there.
    This step is necessary for Kaggle API authentication.
    """
    if not os.path.exists(OS_KAGGLE_DIR):
        os.makedirs(OS_KAGGLE_DIR)  # Create the directory if it doesn't exist
    print(f"Copying {KAGGLE_PATH} to {OS_KAGGLE_FILE}")
    shutil.copy2(KAGGLE_PATH, OS_KAGGLE_FILE)  # Copy the Kaggle API key

def initialize_kaggle_api():
    """
    Initialize and authenticate the Kaggle API.
    This function imports the Kaggle API, creates an instance, and authenticates using the API key.

    Returns:
        KaggleApi: Authenticated Kaggle API object.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()  # Create an instance of the Kaggle API
    api.authenticate()  # Authenticate with the API key
    return api

def ensure_download_directory():
    """
    Ensure the download directory exists.
    Creates the directory if it doesn't already exist.
    """
    if not os.path.exists(DOWNLOAD_PATH):
        os.makedirs(DOWNLOAD_PATH)  # Create the directory

def download_datasets(api):
    """
    Download datasets listed in datasets.txt using the Kaggle API.
    This function reads dataset names from a file and downloads each one.

    Args:
        api (KaggleApi): Authenticated Kaggle API object.
    """
    with open("backend/datasets.txt", "r") as file:
        datasets = file.readlines()  # Read dataset names from the file

    for dataset in tqdm(datasets, desc="Downloading datasets", unit="dataset"):
        dataset = dataset.strip()  # Remove any leading/trailing whitespace
        if dataset:
            print(f"\nDownloading {dataset}...")
            api.dataset_download_files(dataset, path=DOWNLOAD_PATH, unzip=True)
            print(f"Downloaded {dataset} successfully!")

def cleanup():
    """
    Remove the Kaggle API key file after downloading.
    This is a security measure to ensure the API key isn't left exposed.
    """
    if os.path.exists(OS_KAGGLE_FILE):
        os.remove(OS_KAGGLE_FILE)  # Delete the Kaggle API key file

if __name__ == "__main__":
    # Main execution flow
    setup_kaggle_directory()  # Set up the Kaggle directory
    api = initialize_kaggle_api()  # Initialize the Kaggle API
    ensure_download_directory()  # Ensure download directory exists
    download_datasets(api)  # Download datasets from Kaggle
    cleanup()  # Cleanup API key file
    print("\nAll datasets downloaded successfully!")
