# import scientific libraries
import numpy as np
import scipy.linalg as la


# Need to write a function to open PGM files first and foremost
def read_pgm(filename):
	
	f = open(filename)
	pgm_id = f.readline() # This is the pgm identifier - P2

	if pgm_id[:2] == 'P2':

		img = decode_plain_pgm(f)
		f.close()
		return img

	elif pgm_id[:2] == 'P5':

		img = decode_raw_pgm(f)
		f.close()
		return img

	else:

		print "Not a valid PGM file!"
		return -1


# A function to decode plain pgms
def decode_plain_pgm(f):

	line = f.readline().split() # This line contains the width and height
	width  = int(line[0])
	height = int(line[1])

	line = f.readline().split() # It contains the maxval which we don't need.

	img = np.zeros([height, width], dtype='int16')
	n = 0

	while (n < height):

		vals = []

		while (len(vals) < width and len(line) != 0):

			line = f.readline().split()

			for x in line:
			
				vals.append(np.int(x))

		img[n] = np.array(vals)
		n += 1

	return img


# A function to decode raw pgms
def decode_raw_pgm(f):

	line = f.readline().split() # This line contains the width and height
	width  = int(line[0])
	height = int(line[1])

	line = f.readline().split() # This line contains the max value
	maxval = int(line[0])

	if maxval < 256:
		pixel_size = 1 # bytes
	else:
		pixel_size = 2

	img = np.zeros([height, width], dtype='int16')
	n = 0

	while (n < height):

		buff = bytearray(f.read(width * pixel_size))

		for i in range(width):
			
			img[n][i] = int(buff[pixel_size * i])
			
			if pixel_size == 2:
			
				img[n][i] += 2**8 * int(buff[pixel_size * i  + 1])

		n += 1

	return img


# The first analyzer function, principal component analysis
def PCA(X, Y, k=10):

	w, v = la.eig(np.dot(X, np.transpose(X))) # BUG - not sure of order

	big_eigs = np.empty([k, 2])
	eig_vecs = np.empty([k, v.shape[0]])

	for i in range(k):

		idx = np.argsort(w)[i]
		big_eigs[i, 0] = idx
		big_eigs[i, 1] = w[idx]
		eig_vecs[i] = v[:, idx]

	return big_eigs[:,0], eig_vecs




