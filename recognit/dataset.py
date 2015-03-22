"""dataset.py: Contains a class to create datasets."""

import os
import json
import random
import numpy as np
from PIL import Image

__author__ = "Durmus U. Karatay, Matthias W. Smith"
__email__ = "ukaratay@uw.edu, mwsmith2@uw.edu"
__maintainer__ = "Durmus U. Karatay"
__status__ = "Development"


class Dataset(object):
    """
    Creates a dataset from a given path that contains images and image metadata
    in JSON format.

    If images are different sizes, it resizes all of them to the biggest size
    in the dataset.

    Test set is created from what is left after creating training and
    validation datasets.

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
    label : string
        Label for y values of dataset, gets its data from label property in
        metadata file.

    library : string
        Library for the dataset if there is more than one libraries included
        in the metadata file, get its data from library property in metadata
        file.

    Properties
    ----------
    seed : integer
        Seed for random shuffling the dataset. It can be set to same number for
        repeatability.

    xtrain : array_like (n_samples, n_features)
        Training images flattened into an array-like structure.

    xvalid : array_like (n_samples, n_features)
        Validation images flattened into an array-like structure.

    xtest : array_like (n_samples, n_features)
        Test images flattened into an array-like structure.

    ytrain : array_like (n_samples, )
        Training labels in an array-like structure.

    xvalid : array_like (n_samples, )
        Validation labels in an array-like structure.

    xtest : array_like (n_samples, )
        Test labels in an array-like structure.

    """

    def __init__(self, path, training=0.6, validation=0.4):

        # Set the attributes.
        self.path = path
        self.training_ratio = training
        self.validation_ratio = validation
        self.test_ratio = 1 - training - validation

        self.label = ''
        self.library = ''

        # Set the properties.
        self._seed = 1234

        # Initialize the outputs.
        self._xtrain = np.array([])
        self._ytrain = np.array([])
        self._xvalid = np.array([])
        self._yvalid = np.array([])
        self._xtest = np.array([])
        self._ytest = np.array([])

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

        return

    @property
    def seed(self):
        """Seed for randomization"""

        return self._seed

    @seed.setter
    def set_seed(self, value):
        """Setter for seed."""

        self._seed = value

        return

    @property
    def xtrain(self):
        """Training images flattened into an array-like structure."""

        if not len(self._xtrain):

            raise AttributeError('Training dataset is not created yet!')

        else:

            return self._xtrain

    @property
    def ytrain(self):
        """Training labels in an array-like structure."""

        if not len(self._ytrain):

            raise AttributeError('Training dataset is not created yet!')

        else:

            return self._ytrain

    @property
    def xvalid(self):
        """Validation images flattened into an array-like structure."""

        if not len(self._xvalid):

            if self.validation_ratio == 0:

                raise AttributeError('There is no validation dataset!')

            else:

                raise AttributeError('Validation dataset is not created yet!')

        else:

            return self._xvalid

    @property
    def yvalid(self):
        """Validation labels in an array-like structure."""

        if not len(self._yvalid):

            if self.validation_ratio == 0:

                raise AttributeError('There is no validation dataset!')

            else:

                raise AttributeError('Validation dataset is not created yet!')

        else:

            return self._yvalid

    @property
    def xtest(self):
        """Test images flattened into an array-like structure."""

        if not len(self._xtest):

            if self.test_ratio == 0:

                raise AttributeError('There is no test dataset!')

            else:

                raise AttributeError('Test dataset is not created yet!')

        else:

            return self._xtest

    @property
    def ytest(self):
        """Test labels in an array-like structure."""

        if not len(self._ytest):

            if self.test_ratio == 0:

                raise AttributeError('There is no test dataset!')

            else:

                raise AttributeError('Test dataset is not created yet!')

        else:

            return self._ytest

    def create(self, label, library=None):
        """Create datasets for given label and library."""

        # Set label and library.
        self.label = label
        self.library = library

        # If no library is given, get all images in metadata.
        if library:

            image_dict = {key: value for key, value in self.metadata.items()
                          if value['library'] == self.library}

        else:

            image_dict = self.metadata

        # Initialize variables.
        labels = []
        images = []
        size = (0, 0)

        # Set the seed for randomization and shuffle the dataset.
        random.seed(self.seed)
        iterable = sorted(image_dict.items(), key=lambda x: random.random())

        # Iterate over images and load them.
        for key, value in iterable:

            img_file = os.path.join(self.path, key)
            img = Image.open(img_file).convert('L')

            size = max(size, img.size)

            images.append(img)
            labels.append(value['labels'][self.label])

        # Initialize the numpy array.
        xset = np.empty((len(images), size[0] * size[1]))

        # Iterate over images and resize them, then add them to the array.
        for i, img in enumerate(images):

            img = img.resize(size)
            xset[i, :] = np.asarray(img, dtype=np.float).flatten()

        yset = np.array(labels, ndmin=1)

        # Calculate where splits are going to happen.
        idx_train = int(len(images) * self.training_ratio) + 1
        idx_valid = idx_train + int(len(images) * self.validation_ratio)

        idx = [idx_train, idx_valid]

        # Split arrays into datasets.
        self._xtrain, self._xvalid, self._xtest = np.vsplit(xset, idx)
        self._ytrain, self._yvalid, self._ytest = np.split(yset, idx)

        return
