#!/bin/python

"""Process the labels on all these images and create a json metadata file
for the data_set."""

import json
import glob

img_ext = '.pgm'

# First large images
dataset = 'cmu_faces_large'

images = glob.glob('*_4' + img_ext)
metadata = []
traits = ['name', 'orientation', 'mood', 'eyewear']

for image in images:

	data = {}
	labels = image.split('.')[0].split('_')
	data['filename'] = image
	data['dataset'] = dataset
	data['labels'] = {}

	for trait, label in zip(traits, labels):

		data['labels'][trait] = label

	metadata.append(data)

# Now medium
dataset = 'cmu_faces_large'

images = glob.glob('*_2' + img_ext)
metadata = []
traits = ['name', 'orientation', 'mood', 'eyewear']

for image in images:

	data = {}
	labels = image.split('.')[0].split('_')
	data['filename'] = image
	data['dataset'] = dataset
	data['labels'] = {}

	for trait, label in zip(traits, labels):

		data['labels'][trait] = label

	metadata.append(data)

dataset = 'cmu_faces_large'

images = glob.glob('*_0' + img_ext)
metadata = []
traits = ['name', 'orientation', 'mood', 'eyewear']

for image in images:

	data = {}
	labels = image.split('.')[0].split('_')
	data['filename'] = image
	data['dataset'] = dataset
	data['labels'] = {}

	for trait, label in zip(traits, labels):

		data['labels'][trait] = label

	metadata.append(data)

# Write the json file
metafile = open('metadata.json', 'w')
metafile.write(json.dumps(metadata, 
						  sort_keys=True, 
						  separators=[',', ':'], 
						  indent=4))

