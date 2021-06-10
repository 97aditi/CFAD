import numpy as np 
import numpy.linalg as la
import scipy.linalg as linalg
from utils.utils import *

"""" This code was ported from LDR packaange in MATLAB """

def DR(y, X, d, ytype='disc', nslices=0):
	""" Function to compute the projection matrix to project X from its ambient space p to a lower dimensional space d using directional regression """
	""" Parameters:
	    -------------------------------------------------------------------
	    Y is the target vector
	    X is the data matrix (NXp)
	    d is the specified sufficient subspace dimension
	    ytype can be either 'disc' for discrete y or 'cont' for continuous y
	    nslices is the number of slices required only for continous y
	    -------------------------------------------------------------------
	    Outputs: alpha (pxd) the projection matrix
	    -------------------------------------------------------------------
	"""

	# Discretize y if continuous
	if ytype=='cont':
		if slices ==0:
			raise ValueError("For continuous label, slices should be greater than 0.")
		y = slices(y, nslices) - 1


	# Obtain necessary statistics to compute the moment matrix 
	mat1, mat2, mat3, mat4, mat5,nsqrtx = setaux(X, y, d)

	# Top d eigenvector of the moment matrix (eigs = \eta)
	p = X.shape[1]
	dr = 2*mat1 + 2*mat2@mat2 + 2*mat3*mat2 - 2*np.eye(p)
	vals, eigs = linalg.eigh(dr)
	eigs = eigs[:,-d:]


	# eigs (\eta) transformed back to the original space from the standardised space
	dr = nsqrtx@eigs
	alpha = linalg.orth(dr)


	# returns the projection matrix alpha
	return alpha







