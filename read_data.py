# set up matplotlib
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

import analyze

filename = 'data/cheyer/cheyer_up_neutral_open.pgm'

img1 = analyze.read_pgm(filename)
plt.contour(img1)
plt.savefig('fig/image1.pdf')
plt.close()

filename = 'data/cheyer/cheyer_left_angry_open_2.pgm'

img2 = analyze.read_pgm(filename)
plt.contour(img2)
plt.savefig('fig/image2.pdf')
plt.close()
