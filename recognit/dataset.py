"""dataset.py: Contains a class to create datasets."""

__author__ = "Durmus U. Karatay, Matthias W. Smith"
__email__ = "ukaratay@uw.edu, mwsmith2@uw.edu"
__maintainer__ = "Durmus U. Karatay"
__status__ = "Development"

import os
import json
import random
import numpy as np
from PIL import Image

class Dataset(object):
    """
    Creates a dataset from a given path that contains images and image metadata
    in JSON format.

    Parameters
    ----------
    path : string
        Path to folder that contain images and image metadata.

    training : float
        Ratio of training set to whole dataset.

    validation : float
        Ratio of validation set to whole dataset.

    Attributes
    ----------


    """

    def __init__(self, path, training=0.6, validation=0.4):

        # Set the attributes.
        self.path = path
        self.training_ratio = training
        self.validation_ratio = validation
        self.test_ratio = 1 - training - validation

        self.label = None
        self.library = None

        # Set the properties.
        self._seed = 1234

        # Initialize the outputs.
        self._xtrain = None
        self._ytrain = None
        self._xvalid = None
        self._yvalid = None
        self._xtest = None
        self._ytest = None

        # Check if ratios are correct.
        if self.test_ratio < 0:

            raise ValueError('Sum of training and validation ratios'
                             ' should be smaller than or equal to 1!')

        elif self.training_ratio < 0 or self.validation_ratio < 0:

            raise ValueError('Ratios can not be negative!')

        # Look for json files in the given path.
        self.json_files = [os.path.join(path, f) for f in os.listdir(path)
                           if f.endswith('json')]

        # Check if there a single concatenated json file or a json file for
        # every image file.
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

    @property
    def seed(self):
        """Seed for randomization"""

        return self._seed

    @property
    def xtrain(self):

        if not self._xtrain:

            raise AttributeError('Training dataset is not created yet!')

        else:

            return self._xtrain

    @property
    def ytrain(self):

        if not self._ytrain:

            raise AttributeError('Training dataset is not created yet!')

        else:

            return self._ytrain

    @property
    def xvalid(self):

        if not self._xvalid:

            if self.validation_ratio == 0:

                raise AttributeError('There is no validation dataset!')

            else:

                raise AttributeError('Validation dataset is not created yet!')

        else:

            return self._xvalid

    @property
    def yvalid(self):

        if not self._yvalid:

            if self.validation_ratio == 0:

                raise AttributeError('There is no validation dataset!')

            else:

                raise AttributeError('Validation dataset is not created yet!')

        else:

            return self._yvalid

    @property
    def xtest(self):

        if not self._xtest:

            if self.test_ratio == 0:

                raise AttributeError('There is no test dataset!')

            else:

                raise AttributeError('Test dataset is not created yet!')

        else:

            return self._xest

    @property
    def ytest(self):

        if not self._ytest:

            if self.test_ratio == 0:

                raise AttributeError('There is no test dataset!')

            else:

                raise AttributeError('Test dataset is not created yet!')

        else:

            return self._ytest

    @seed.setter
    def set_seed(self, value):
        """Setter for seed."""

        self._seed = value

    def create(self, label, library):
        """Create datasets for given label and library."""

        # Set label and library.
        self.label = label
        self.library = library

        # If no library is given, get all images in metadata.
        if library:

            image_dict = {key: value for key, value in self.metadata.items()
                          if value['dataset'] == self.library}

        else:

            image_dict = self.metadata

        # Initialize lists.
        labels = []
        images = []

        # Set the seed for randomization and shuffle the dataset.
        random.seed(self.seed)
        iterable = sorted(image_dict.items(), key=lambda x: random.random())

        # Iterate over images and load them.
        for key, value in iterable:

            im_file = os.path.join(self.path, key)
            im = Image.open(im_file).convert('L')

            images.append(np.asarray(im, dtype=np.float_).flatten())
            labels.append(value['labels'][self.label])

        x = np.array(images)
        y = np.array(labels)

        idx_train = int(len(images) * self.training_ratio)
        idx_valid = idx_train + int(len(images) * self.validation_ratio)

        self._xtrain, rest = np.hsplit(x, idx_train)
        self._xvalid, self._xtest = np.hsplit(rest, idx_valid)

        self._ytrain, rest = np.hsplit(y, idx_train)
        self._yvalid, self._ytest = np.hsplit(rest, idx_valid)
