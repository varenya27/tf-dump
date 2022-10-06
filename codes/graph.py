import corner
import numpy as np

ndim, nsamples = 2, 10000
np.random.seed(42)
samples = np.random.randn(ndim * nsamples).reshape([nsamples, ndim])
figure = corner.corner(samples,range=[(-10,10),(-5,5)])
figure.savefig('codes/corner.pdf')