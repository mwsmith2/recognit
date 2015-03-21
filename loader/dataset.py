"""dataset.py: Contains a class to create datasets."""

__author__ = "Durmus U. Karatay, Matthias W. Smith"
__email__ = "ukaratay@uw.edu, mwsmith2@uw.edu"
__maintainer__ = "Durmus U. Karatay"
__status__ = "Development"

import os
import json
import random
import numpy as np
from skimage import io

class Dataset(object):
    """
    Creates a dataset from a given path that contains images and image metadata
    in JSON format.

    Parameters
    ----------
    path : string
        Path to folder that contain images and image metadata.

    Attributes
    ----------
        

    """

    def __init__(self, path):

        self.seed = 1234

        # Set the path and parse from JSON files.
        self.path = path
        self.json_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('json')]
        
        if len(self.json_files) == 1:
            
            with open(self.json_files[0], 'r') as f:
                
                self.metadata = json.load(f)
        
        elif len(self.json_files) > 1:
            
            self.metadata = {}

            for json_file in self.json_files:
                
                with open(json_file, 'r') as f:
                    
                    self.metadata.update(json.load(f))
                    
        else:
            
            raise IOError('No JSON file found!')
        
        return
        
    def create(self, label='name', library='cmu_faces_large'):
        
        if not library:
            
            image_dict = {key:value for key, value in self.metadata.items() if value['dataset'] == library}            

        else:
            
            image_dict = self.metadata

        labels = []
        images = []
        random.seed(self.seed)
        
        
        for key, value in sorted(image_dict.items(), key=lambda x: random.random()):
            
            image_file = os.path.join(self.path, key)
            image = io.imread(image_file, as_grey=True)
            
            images.append(image.flatten())
            labels.append(value['labels'][label])
            
        X = np.array(images)
        y = np.array(labels)
        
        return (X, y)