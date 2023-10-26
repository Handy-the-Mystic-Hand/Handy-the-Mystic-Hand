import os
import shutil
import getpass
from tqdm import tqdm

# Paths
kagglePath = "backend/kaggle.json"
username = getpass.getuser()
osKaggleDir = "/Users/" + username + "/.kaggle"
osKaggleFile = os.path.join(osKaggleDir, "kaggle.json")

# Check if .kaggle directory exists, if not, create it
if not os.path.exists(osKaggleDir):
    os.makedirs(osKaggleDir)

# Copy kaggle.json to the .kaggle directory
print(f"Copying {kagglePath} to {osKaggleFile}")
shutil.copy2(kagglePath, osKaggleFile)

from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Set the path where you want to download the dataset
download_path = "backend/download"

# Check if directory exists, create if it doesn't
if not os.path.exists(download_path):
    os.makedirs(download_path)

sets = []
# Download dataset
with open("backend/datasets.txt", "r") as file:
    datasets = file.readlines()
    sets.append(datasets)

# Wrap the datasets list with tqdm for progress bar
for dataset in tqdm(datasets, desc="Downloading datasets", unit="dataset"):
    dataset = dataset.strip()  # remove newlines or extra spaces
    if dataset:  # ensure it's not an empty line
        print(f"\nDownloading {dataset}...")
        api.dataset_download_files(dataset, path=download_path, unzip=True)
        print(f"Downloaded {dataset} successfully!")

# Remove kaggle.json after downloading
if os.path.exists(osKaggleFile):
    os.remove(osKaggleFile)

print("\nAll datasets downloaded successfully!")
