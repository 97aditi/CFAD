import autograd.numpy as np
np.set_printoptions(precision=4)
from pymanopt.manifolds import Product,Euclidean,Grassmann, Stiefel
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions, linesearch, ConjugateGradient, LBFGS
import scipy
from utils import utils
import time
import pymanopt

# SUPPORTED_BACKENDS = (
#     "Autograd", "Callable", "PyTorch", "TensorFlow", "Theano"
# )


def CFAD(y,X,d,q,theta_init=[],lamda=0):
    """ Function to implement CFAD """
    """ Parameters:
            ----------------------------------------------------------------------------------------------------
            - y is the target vector
            - X is the feature matrix (pXN)
            - d is the specified sufficient subspace dimension
            - theta_init is the initial guess for alpha to be provided to the optimisation routine
            - Lamda can be set (0-1) to add smoothness prior to CFAD 
            -----------------------------------------------------------------------------------------------------
            Outputs: alpha (pxd) the projection matrix
            -----------------------------------------------------------------------------------------------------
    """

    # Number of elements in each class
    u, ny = np.unique(y, return_counts=True)
    # Number of classes
    h = u.shape[0]
    # Number of samples
    N = X.shape[1]
    # Dimension of ambient space
    p = X.shape[0]
    

    # Computing statistics of input data
    nu_sample = np.empty((h,X.shape[0]), dtype=np.float32)
    cov_sample = np.empty((h, X.shape[0], X.shape[0]), dtype=np.float32)
    for i in range(h):
        indices = (i==y)
        nu_sample[i] = np.mean(X[:,indices], axis=1)
        cov_sample[i] = np.cov(X[:,indices])

    # Initialising Product manifold
    edim = (d)*h+q+1
    k = d+q
    manifold = Product([Euclidean(edim), Stiefel(p, k)])

    eps = 1e-3
    
    @pymanopt.function.Autograd
    def cost(theta_0, theta_1):
        # theta_0 = theta[0]
        # theta_1 = theta[1]
        """ Defines the log likelihood/posterior to optimise (If lamda>0, then we add a prior and hence optimise the log-posterior"""
        #Extracting all parameters from a single list (theta) 
        alpha = theta_1[:,:d]
        alpha_0 = theta_1[:,d:d+q]

        j = 0
        LambdaLambdaT=[]
        for k in range(h):
            LambdaLambdaT.append(np.diag(theta_0[j:j+d]))
            j=j+d
        LambdaLambdaT = np.array(LambdaLambdaT)
        Lambda0Lambda0T = np.diag(theta_0[h*(d):h*(d)+q])
        sigma = theta_0[-1]

        #-------------------------------------------------------

        # Defining log-likelihood/posterior function
        if q==0:
            classinv = np.linalg.inv(alpha@LambdaLambdaT@ LambdaLambdaT@alpha.T + (sigma*sigma + eps)*np.eye(p,p))
            logdet = np.linalg.slogdet(alpha@LambdaLambdaT@ LambdaLambdaT@alpha.T + (sigma*sigma + eps)*np.eye(p,p))[1]
            term1 = np.sum(-0.5*ny*lodget)
        else:
            alpha0_cov = alpha_0 @ Lambda0Lambda0T @ Lambda0Lambda0T @ alpha_0.T
            alpha_cov = alpha@LambdaLambdaT@LambdaLambdaT@alpha.T
            class_cov = alpha_cov + alpha0_cov + (sigma*sigma + eps)*np.eye(p,p)
            classinv = np.linalg.inv(class_cov)
            logdet = np.linalg.slogdet(class_cov)[1]
            term1 = np.sum(-0.5*ny*logdet)

        term2 = 0
        for i in range(h):
            nu = np.expand_dims(nu_sample[i].T, axis=1)
            C_y = (np.eye(p,p) - alpha@alpha.T) @ nu @ nu.T @ (np.eye(p,p) - alpha@alpha.T)
            term2 = term2 -0.5*ny[i]*np.trace(classinv[i]@(cov_sample[i] + C_y))

        L = utils.gen_laplacian(p)
        # Smoothing prior 
        prior = -0.5 * np.trace(alpha.T @ L @ alpha)

        loglike = -(term1 + term2 + lamda*prior)
        return loglike

        loglike = -(term1 + term2)
        return loglike


    # Defining the problem and solver
    problem = Problem(manifold=manifold, cost=cost,  verbosity=2)
    solver = LBFGS()
    if len(theta_init)==0:
        Wopt,_ = solver.solve(problem)
    else:
        Wopt,_= solver.solve(problem, x = theta_init)

    # Extract the predicted projection matrix (pxd) from the list of optimal parameters 
    alpha_opt = Wopt[1][:,:d]

    return alpha_opt


