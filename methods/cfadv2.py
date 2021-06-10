import autograd.numpy as np
np.set_printoptions(precision=4)
from pymanopt.manifolds import Product, Euclidean, PositiveDefinite, Grassmann, Stiefel
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions, linesearch, ConjugateGradient, LBFGS
import scipy
from utils.utils import *

def CFAD(y, X, d, q, ytype = 'disc', nslices=0, theta_init=None, lamda=0):
    """ Function to implement CFAD """
    """ Parameters:
            --------------------------------------------------------------------------------------
            - y is the target vector
            - X is the feature matrix (pXN)
            - d is the specified sufficient subspace dimension
            - theta_init is the initial guess for alpha to be provided to the optimisation routine
            - Lamda can be set (0-1) to add smoothness prior to CFAD 
            - ytype can be either 'disc' (discrete) or 'cont' (continuous)
            - nslices is only required for continuous data to define the number of slices
            ---------------------------------------------------------------------------------------
            Outputs: alpha (pxd) the projection matrix
            ---------------------------------------------------------------------------------------
    """

    if ytype == 'cont':
        if slices ==0:
            raise ValueError("For continuous label, slices should be greater than 0.")
        y = slices(y, nslices) - 1

    # Number of elements in each class
    u, ny = np.unique(y, return_counts=True)
    # Number of classes
    h = u.shape[0]
    # Number of samples
    N = X.shape[1]
    # Dimension of ambient space
    p = X.shape[0]
    # Graph laplacian which acts as a smoothness prior on the columns of the projection vectors from ambient space to low-dimensional subspace
    L = gen_laplacian(p)


    # Computing statistics of input data
    nu_sample = np.empty((h,X.shape[0]))
    cov_sample = np.empty((h, X.shape[0], X.shape[0]))
    for i in range(h):
        indices = (i==y)
        nu_sample[i] = np.mean(X[:,indices], axis=1)
        cov_sample[i] = np.cov(X[:,indices])

    cov_sqrtinv = np.linalg.pinv(scipy.linalg.sqrtm(np.cov(X)))
    cov_X = np.cov(X)



    # Initialising Product manifold
    edim = (d)*h+1+q
    k = d+q
    manifold = Product((Euclidean(edim), Stiefel(p, k)))

    eps = 1e-3

    def cost(theta):
        """ Defines the log likelihood/posterior to optimise (If lamda>0, then we add a prior and hence optimise the log-posterior"""
        #Extracting all parameters from a single list (theta) 
        alpha = theta[1][:,:d]
        alpha_0 = theta[1][:,d:d+q]

        j = 0
        LambdaLambdaT=[]
        for k in range(h):
            LambdaLambdaT.append(np.diag(theta[0][j:j+d]))
            j=j+d
        LambdaLambdaT = np.array(LambdaLambdaT)
        Lambda0Lambda0T = np.diag(theta[0][h*(d):h*(d)+q])
        sigma = theta[0][-1]

        #-------------------------------------------------------

        # Defining log-likelihood/posterior function
        if q==0:
            classinv = np.linalg.inv(alpha@LambdaLambdaT@ LambdaLambdaT@alpha.T + (sigma*sigma + eps)*np.eye(p,p))
            sign, logdet = np.linalg.slogdet(alpha@LambdaLambdaT@ LambdaLambdaT@alpha.T + (sigma*sigma + eps)*np.eye(p,p))
            term1 = np.sum(-0.5*ny*sign*lodget)
        else:
            alpha0_cov = alpha_0 @ Lambda0Lambda0T @ Lambda0Lambda0T @ alpha_0.T
            classinv = np.linalg.inv(alpha@LambdaLambdaT@ LambdaLambdaT@alpha.T + alpha0_cov + (sigma*sigma + eps)*np.eye(p,p))
            sign, logdet = np.linalg.slogdet(alpha@LambdaLambdaT@ LambdaLambdaT@alpha.T + alpha0_cov + (sigma*sigma + eps)*np.eye(p,p))
            term1 = np.sum(-0.5*ny*sign*logdet)
                
        term2 = 0
        term3 = 0

        for i in range(h):
            term2 = term2 -0.5*ny[i]*np.trace(classinv[i]@cov_sample[i])
            term3 = term3 -0.5*ny[i]*nu_sample[i] @ (np.eye(p,p) - alpha@alpha.T) @ classinv[i] @ (np.eye(p,p) - alpha@alpha.T) @ nu_sample[i].T

        # Smoothing prior 
        prior = -0.5 * np.trace(alpha.T @ L @ alpha)
        loglike = -(-N*p*np.log(np.pi)/2 + term1 + term2 +term3 + lamda*prior)
        return loglike


    # Defining the problem and solver
    problem = Problem(manifold=manifold, cost=cost,  verbosity=0)
    solver = LBFGS()
    if theta_init == None:
        Wopt,_ = solver.solve(problem)
    else:
        Wopt,_ = solver.solve(problem, x = theta_init)

    # Extract the predicted projection matrix (pxd) from the list of optimal parameters 
    alpha_opt = Wopt[1][:,:d]

    return alpha_opt


