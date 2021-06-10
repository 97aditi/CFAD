import autograd.numpy as np
import torch
# np.set_printoptons(precision=4)
from pymanopt.manifolds import Product, Euclidean,Grassmann, Stiefel
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions, linesearch, ConjugateGradient, LBFGS
import scipy
from utils import utils
import time
import pymanopt


SUPPORTED_BACKENDS = (
    "Autograd", "Callable", "PyTorch", "TensorFlow", "Theano"
)

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)

def cfad(y,X,d,q,theta_init=[],lamda=0):
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
    # Convert to torch
    X = torch.from_numpy(X)
    X = X.to(device)
    y = torch.from_numpy(y)
    y = y.to(device)


    # Number of elements in each class
    u, ny = torch.unique(y, return_counts=True)
    # Number of classes
    h = u.shape[0]
    # Number of samples
    N = X.shape[1]
    # Dimension of ambient space
    p = X.shape[0]
    

    # Computing statistics of input data
    nu_sample = torch.empty((h,X.shape[0])).to(device)
    cov_sample = torch.empty((h, X.shape[0], X.shape[0])).to(device)

    for i in range(h):
        indices = (i==y)
        nu_sample[i] = torch.mean(X[:,indices], axis=1)
        X_ = X[:,indices]
        X_ = X_ - torch.unsqueeze(nu_sample[i], axis=1)
        cov_sample[i] = (X_@X_.T)/N

    # Initialising Product manifold
    edim = (d)*h+q+1
    k = d+q
    manifold = Product((Euclidean(edim), Stiefel(p, k)))


    @pymanopt.function.PyTorch
    def cost(theta_0, theta_1):
        theta_0 = theta_0.to(device)
        theta_1 = theta_1.to(device)

        """ Defines the log likelihood/posterior to optimise (If lamda>0, then we add a prior and hence optimise the log-posterior"""
        #Extracting all parameters from a single list (theta) 
        alpha = theta_1[:,:d]
        alpha_0 = theta_1[:,d:d+q]

        j = 0
        LambdaLambdaT=[]
        for k in range(h):
            LambdaLambdaT.append(torch.diag(theta_0[j:j+d]))
            j=j+d
        # LambdaLambdaT = np.array(LambdaLambdaT)
        LambdaLambdaT = torch.stack(LambdaLambdaT)
        Lambda0Lambda0T = torch.diag(theta_0[h*(d):h*(d)+q])
        sigma = theta_0[-1]
        #-------------------------------------------------------
        # Defining log-likelihood/posterior function
        eps = 1e-7
        if q==0:
            classinv = torch.inverse(alpha@LambdaLambdaT@ LambdaLambdaT@alpha.T + (sigma*sigma + eps)*torch.eye(p,p))
            sign, logdet = torch.slogdet(alpha@LambdaLambdaT@ LambdaLambdaT@alpha.T + (sigma*sigma + eps)*torch.eye(p,p))
            term1 = torch.sum(-0.5*ny*logdet)
        else:
            alpha0_cov = alpha_0 @ Lambda0Lambda0T @ Lambda0Lambda0T @ alpha_0.T
            alpha_cov = alpha@LambdaLambdaT@LambdaLambdaT@alpha.T
            class_cov = alpha_cov + alpha0_cov + (sigma*sigma + eps)*torch.eye(p,p)
            classinv = torch.inverse(class_cov)
            sign, logdet = torch.slogdet(class_cov)
            term1 = torch.sum(-0.5*ny*logdet)
                
        term2 = 0
        for i in range(h):
            nu = torch.unsqueeze(nu_sample[i].T, axis=1).double()
            C_y = (torch.eye(p,p) - alpha@alpha.T) @ nu @ nu.T @ (torch.eye(p,p) - alpha@alpha.T)
            term2 = term2 -0.5*ny[i]*torch.trace(classinv[i]@(cov_sample[i] + C_y))

        L = utils.gen_laplacian(p)
        L = torch.from_numpy(L)
        # Smoothing prior 
        prior = -0.5 * torch.trace(alpha.T @ L @ alpha)

        loglike = -(term1 + term2 + lamda*prior)
        return loglike


    # Defining the problem and solver
    problem = Problem(manifold=manifold, cost=cost,  verbosity=0)
    solver = LBFGS()
    if len(theta_init)==0:
        Wopt,_ = solver.solve(problem)
    else:
        Wopt,_ = solver.solve(problem, x = theta_init)

    # Extract the predicted projection matrix (pxd) from the list of optimal parameters 
    alpha_opt = Wopt[1][:,:d]

    return alpha_opt


