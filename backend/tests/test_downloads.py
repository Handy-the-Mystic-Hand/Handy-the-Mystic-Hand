import unittest
import os
import downloads

class DownloadsTester(unittest.TestCase):

    def setUp(self):
        self.kaggle_dir = downloads.OS_KAGGLE_DIR
        self.kaggle_file = downloads.OS_KAGGLE_FILE
        downloads.setup_kaggle_directory()


    def test_kaggle_directory_creation(self):
        self.assertTrue(os.path.exists(self.kaggle_dir))

    # def test_kaggle_file_copy(self):
    #     if os.path.exists(self.kaggle_file):
    #         os.remove(self.kaggle_file)
    #     downloads.setup_kaggle_directory()
    #     self.assertTrue(os.path.exists(self.kaggle_file))

    # def test_download_datasets_functional(self):
    #     # Setup Kaggle directory
    #     downloads.setup_kaggle_directory()

    #     # Initialize the Kaggle API
    #     api = downloads.initialize_kaggle_api()

    #     # Ensure download directory exists
    #     downloads.ensure_download_directory()

    #     # Download the datasets using the initialized API
    #     downloads.download_datasets(api)

    #     # Cleanup actions
    #     downloads.cleanup()

    #     self.assertTrue(os.path.exists(downloads.DOWNLOAD_PATH))

    def tearDown(self):
        # Cleanup logic, for instance:
        if os.path.exists(self.kaggle_file):
            os.remove(self.kaggle_file)

if __name__ == '__main__':
    unittest.main()
