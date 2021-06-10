
from utils.utils import *
from pymanopt.manifolds import Grassmann
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions
import autograd.numpy as np
import pymanopt

"""" This code was ported from LDR packaange in MATLAB """

def LAD(Y, X, d, ytype='disc', nslices=0, theta_init=None):
	""" Implements Likelihood-Acquired Directions (LAD) from Cook (2009)"""
	""" Parameters:
	    ---------------------------------------------------------------------------------------
	    - Y is the target vector
	    - X is the array of predictors such that X[k,:] predicts Y
	    - d is the specified sufficient subspace dimension
	    - ytype can be either 'disc' for discrete y or 'cont' for continuous y
	    - nslices is the number of slices required only for continous y
	    - theta_init is the initial guess for alpha to be provided to the optimisation routine
	    ---------------------------------------------------------------------------------------
	    Outputs: alpha (pxd) the projection matrix
	    ---------------------------------------------------------------------------------------
	"""

	np.random.seed(10)

	Y = np.array(Y)
	X = np.array(X)


	if ytype=='disc':
		miny = min(Y)
		Y = Y-(miny-1)
		nslices = max(Y)

	elif ytype == 'cont':
		if nslices==0:
			print("No. of slices not specificed, setting slices to 5")
			nslices=5
		Y = slices(Y,nslices)


	# Finding sample statistics
	sigmag, sigmas, counts = get_pars(Y,X,nslices)

	nj = counts
	n = np.sum(counts)
	p = sigmag.shape[0]
	h = nj/n


	# Defining Objective Function
	@pymanopt.function.Autograd
	def obj_func(theta):
		loglikelihood = (n*p*(1+np.log(2*np.pi))/2 + n/2 *np.linalg.slogdet(sigmag)[1] - n/2 * np.linalg.slogdet(theta.T  @ sigmag @ theta)[1] + n/2*(h.T @ np.linalg.slogdet(theta.T @ sigmas @ theta)[1]))
		return loglikelihood

	# Optimising using pymanopt
	manifold = Grassmann(p,d)
	problem = Problem(manifold=manifold, cost=obj_func, verbosity=0)
	solver =  TrustRegions(logverbosity=0, maxiter=5000, mingradnorm=1e-10, minstepsize=1e-30)
	alpha = solver.solve(problem, x=theta_init)

	# Returns the projection matrix alpha 
	return alpha


	







