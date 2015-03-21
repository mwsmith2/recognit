import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

m_dpi = 600

def create_font(fontname='Tahoma', fontsize=10):
	"""
	Create a font object to be used in matplotlib figures.

	Parameters
	----------
	fontname : string

		The desired font, i.e., Times New Roman, Verdana, etc.

	fontsize : integer

		The fontsize used for display on the figure.

	Returns
	-------
	mpl_font : dictionary

		A font dictionary used by matplotlib figures.

	"""

	return { 'fontname': fontname, 'fontsize': fontsize }

def plot_faces(title, images, rows=3, cols=6, sptitle="subplot", sptitles=[], colormap=cm.jet, filename=None):
	"""
	Plots a grid of faces, useful for plotting eigenfaces/fisherfaces.

	Parameters
	----------

	title : string

		Title for the whole set of plots.

	images : array-like

		The set of images to be plotted.

	rows : integer

		Number of rows in face image grid.

	cols : integer

		Number of columns in the face image grid.

	sptitle : string

		The title to be numbered and used on each subplot.

	sptitles : list of strings

		A set of different titles to be called for each sequential subplot.

	colormap : matplotlib cm

		The color gradient that represents intensity on the figure.

	filename : string

		Output filename to save the image to disk.

	Returns
	-------
	None

	"""


	plt.clf()

	fig = plt.figure(figsize=(15, 8), dpi=m_dpi)
	fig.text(.5, .95, title, horizontalalignment='center')

	for i in xrange(len(images)):

		ax0 = fig.add_subplot(rows,cols,(i+1)) 
		plt.setp(ax0.get_xticklabels(), visible=False) 
		plt.setp(ax0.get_yticklabels(), visible=False)

		if len(sptitles) == len(images):
			plt.title(r'%s %s' % (sptitle, str(sptitles[i])), create_font('Tahoma', 10))
		else:
			plt.title(r'%s %d' % (sptitle, (i+1)), create_font('Tahoma',10))
		plt.imshow(np.asarray(images[i]), cmap=colormap) 

	if filename is None:
		plt.show() 
	else:
		fig.savefig(filename)

	plt.close()

def normalize(a, lo, hi):
	"""Maps an array linear into a new range.

	Parameters
	----------

	a : array-like

		The data to be adjusted

	lo : float

		The lowest desired value

	hi : float

		The highest desired value

	Returns
	-------

	a : array-like

		The remapped input array.

	"""

	amin = np.min(a)
	amax = np.max(a)
	sf = float(hi - lo) / (amax - amin)

	a -= amin
	a *= sf
	return a

def scatter_face(title, faceweights, x1=1, x2=2, filename=None):
	"""Create a nice plot meant to show clustering in the PCA weights.

	Parameters
	----------

	title : string

		The title of the plot

	faceweights : array-like 

		The projected data going into the plot.

	x1 : integer

		The principal component index on the x-axis

	x2 : integer

		The principal component index on the y-axis

	filename : string

		If given, the plot will be saved as this file.

	Returns
	-------
	None

	"""

	plt.clf()

	fig = plt.figure(figsize=(12, 8), dpi=m_dpi)
	#fig.text(0.5, 0.95, title, horizontalalignment='center')

	colors = cm.gist_ncar(np.linspace(0, 1, len(faceweights)))

	# Need to find average distance, and I can't think of a better way
	davg = 0
	dcount = 0
	for name in faceweights:
		for vec in faceweights[name]:
			davg += np.sum(vec[2])
			dcount += 1

	davg /= dcount

	# Now loop and plot all the points
	for name, color in zip(faceweights, colors):

		x = []
		y = []
		s = []

		for vec in faceweights[name]:
			x.append(vec[0])
			y.append(vec[1])
			s.append(5 + 295 * (np.tanh(vec[2] / davg - 1) + 1.0))

		plt.scatter(x, y, s=s, c=color, alpha=0.5, label=name)

	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')

	plt.tick_params(axis='x',labelbottom='off')
	plt.tick_params(axis='y',labelleft='off')
	plt.xlim(plt.xlim()[0], plt.xlim()[1] + 0.2 * (plt.xlim()[1] - plt.xlim()[0]))
	plt.xlabel(r'$\omega_' + str(x1) + r'$', fontsize=16)
	plt.ylabel(r'$\omega_' + str(x2) + r'$', fontsize=16)
	plt.legend(loc=4)

	if filename is None:
		plt.show() 
	else:
		fig.savefig(filename)

	plt.close()
