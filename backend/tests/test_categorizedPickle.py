import unittest
import os
import categorizedPickle
import mediapipe as mp
import cv2
import os
import numpy as np
import pickle
import shutil
from tqdm import tqdm
import imghdr
from datetime import datetime, timedelta

class CategorizedPickleTester(unittest.TestCase):
    def setUp(self):
        # Sample image for testing
        self.bones_path = "backend/tests/test_bones"
        self.database_path = "backend/tests/test_database"
        self.pickle_path = "backend/tests/test_pickles"
        self.sample_image_path = "backend/tests/sample.jpg"
        shutil.copy2(self.sample_image_path, self.bones_path)

        # Set up HandDatabase instance
        self.mp_hands = mp.solutions.hands
        self.hands_processor = self.mp_hands.Hands(static_image_mode=True)
        self.database = categorizedPickle.HandDatabase(self.bones_path, self.database_path, self.pickle_path, self.hands_processor)

    def tearDown(self):
    # Cleanup contents of created directories
        self._clear_directory_contents(self.bones_path)
        self._clear_directory_contents(self.database_path)
        self._clear_directory_contents(self.pickle_path)

    def _clear_directory_contents(self, dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    
    def test_boneStructure_new_image(self):
        self.database.boneStructure()
        with open(os.path.join(self.pickle_path, "bonesPickle.pickle"), "rb") as f:
            data = f.read()
        self.assertNotEqual(data, b"")
    
    def test_generateDatabase(self):
        self.database.boneStructure()
        self.database.generateDatabase()

        # Check if directories were created
        img_dir = os.path.join(self.database_path, "sample")
        self.assertTrue(os.path.exists(img_dir))
        # Check if image was copied
        self.assertTrue(os.path.exists(os.path.join(img_dir, "sample.jpg")))
    
    def test_image_was_modified(self):
        # Process the image for the first time
        self.database.boneStructure()

        # Modify the sample image by overwriting it or making actual changes (here we're simply copying the same image over to simulate a file change)
        shutil.copy2(self.sample_image_path, os.path.join(self.bones_path, "sample.jpg"))

        # Process again
        self.database.boneStructure()

        # Check if the image was processed again by looking for its updated timestamp in processed_images
        modified = False
        for image in self.database.processed_images:
            if image[0] == os.path.join(self.bones_path, "sample.jpg"):
                # Check if the image's timestamp was updated in the last 5 seconds
                if image[2] > datetime.now() - timedelta(seconds=5):  # Assuming tests run fast
                    modified = True
        self.assertTrue(modified, "The image was not reprocessed after being modified.")

    def test_bonePickle_structure_and_values(self):
        # Process the images to update the bonesPickle
        self.database.boneStructure()

        # Load the data from bonesPickle
        with open(os.path.join(self.pickle_path, "bonesPickle.pickle"), "rb") as f:
            data = pickle.load(f)

        # Check if the data is a list (assuming it should be a list)
        self.assertTrue(isinstance(data, list), "The bonesPickle data is not a list.")

        last_order = 0
        for entry in data:
            # Check if each entry is a tuple of length 4 (or whatever the expected length is)
            self.assertTrue(isinstance(entry, tuple) and len(entry) == 4, "Invalid entry structure in bonesPickle.")

            # Check the structure of each tuple's elements
            img_path = entry[0]
            self.assertTrue(os.path.isfile(img_path), f"The image path {img_path} does not exist.")
            
            landmarks = entry[1]
        for landmark in landmarks:
            x, y = landmark['x'], landmark['y']
            # Check if x and y are between -1 and 1, as they should be after normalization
            self.assertTrue(-1 <= x <= 1)
            self.assertTrue(-1 <= y <= 1)
            
            timestamp = entry[2]
            self.assertTrue(timestamp <= datetime.now(), "The timestamp is in the future.")
            
            order = entry[3]
            self.assertTrue(order == last_order, "The order is not sequential.")
            last_order = order