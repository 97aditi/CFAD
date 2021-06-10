import autograd.numpy as np 
import scipy
import scipy.linalg as sla
import numpy.linalg as la


def slices(y,nslices):
    """ Function to discretize continuous y
        Args:
        - y: continuous target vector y
        - nslices: number of bins to discretize y
        Returns:
        - Y: Discrete y
    """
    indy = np.argsort(y)
    corte = int(np.floor(float(y.shape[0])/nslices))
    Y = np.ones(y.shape[0])
    k=1
    for j in range(nslices):
        Y[indy[corte*j:corte*(j+1)]] = k
        k=k+1
    return Y


def get_pars(Y,X,nslices):
    """ Outputs class-wise covariances, sample covariance and n_y required for LAD
        Args:
        - X: data in original space (N x p)
        - y: discrete target vector
        - nslices: number of slices/classes in y
        Returns:
        - sigmag: covarinace of X
        - sigmas: class-covariances of X
        - counts: number of samples per class
    """
    h = int(nslices)
    nfeatures = X.shape[1]

    sigmas = np.zeros((h,nfeatures,nfeatures))
    # means = np.zeros((nfeatures,h))
    counts = np.zeros((h,1))

    for j in range(h):    
        Xj = X[Y==(j+1),:]
        counts[j,0]  = Xj.shape[0]
        sigmas[j,:,:] = np.cov(Xj, rowvar=False, bias=True)

    sigmag = np.cov(X, rowvar=False, bias=True)
    return sigmag, sigmas, counts


def gen_laplacian(p):
    """ Generates graph laplacian matrix of the size pxp """
    ntfilt = p
    mat = np.ones((ntfilt,2))
    mat[:,0] = mat[:,0]*(-1)
    Dx1 = scipy.sparse.spdiags(mat.T,np.array([0,1]),ntfilt-1,ntfilt)
    Dx = Dx1.T @ Dx1
    D = Dx.toarray()
    return D


""" This code was ported from LDR package in MATLAB """
def setaux(X, y, d):
    """"This function sets shared auxiliary partial results to compute generating
        vectors for the central subspace according to SIR, SAVE and DR methods. 
        Args:
        - X is an {Nxp} matrix with N = number of samples
        - y is the discrete array of labels 
        - d is the dimension of the reduced subspace
        Returns:
        Statistics required by eigen-value based SDR methods
     """

    N = len(y)
    p = X.shape[1]

     # Fraction of points belonging to each class
    u, ny = np.unique(y, return_counts=True)
    h = u.shape[0]
    pv = ny/N

     # Standardise X
    cov_X = np.cov(X.T)
    S, V = la.eig(cov_X)
    nsqrtx = V @ np.diag(np.sqrt(1.0/S)) @ V.T
    Z = (X - np.expand_dims(np.mean(X, axis=0), axis=1).T)@nsqrtx


     # Construct auxilliary arrays
    means = np.zeros((p, h))
    covs = np.zeros((p,p,h))

    for i in range(h):
        indices = (i==y)
        means[:,i] = np.mean(Z[indices, :], axis=0)
        covs[:,:,i] = covs[:,:,i] + Z[indices,:].T@Z[indices,:]
        covs[:,:,i] = covs[:,:,i]/ny[i]

    mat1 = np.zeros((p,p))
    mat2 = np.zeros((p,p))
    mat3 = 0
    mat4 = np.zeros((p,p))
    mat5 = np.zeros((p,p))


    for k in range(h):
        mat1 = mat1 + pv[k]*covs[:,:,k]@covs[:,:,k]
        mat2 = mat2 + pv[k]*np.expand_dims(means[:,k], axis=1)@np.expand_dims(means[:,k], axis=1).T
        mat3 = mat3 + np.sum(means[:,k]*means[:,k])*pv[k]
        mat4 = mat4 + pv[k]*covs[:,:,k]@np.expand_dims(means[:,k], axis=1)@np.expand_dims(means[:,k], axis=1).T
        mat5 = mat5 + pv[k]*np.sum(means[:,k]*means[:,k])*np.expand_dims(means[:,k], axis=1)@np.expand_dims(means[:,k], axis=1).T


    return mat1, mat2, mat3, mat4, mat5, nsqrtx


def initialise_cfad (seed, X, y, d, q, alpha):
    """ Function to initialise CFAD given an estimate of alpha by any other method
        Args: 
        - seed: random seed to reproduce results
        - X: data in original space (N x p)
        - y: discrete target vector
        - d: subspace dimensionality
        - q: class-independent subspace dimensionality
        - alpha: estimates alpha by a different method as initial guess
        Returns:
        - theta_init: list of parameters which serve as initial guess for all parameters required by CFAD
    """
    np.random.seed(seed)
    u, ny = np.unique(y, return_counts=True)
    h = u.shape[0]

    K = h

    # Computing statistics of input data
    cov_sample = np.empty((K, X.shape[1], X.shape[1]))
    for i in range(K):
        indices = (i==y)
        cov_sample[i] = np.cov(X[indices, :].T)

    # Generating null_space of alpha and projecton of X on that space
    null_alpha = scipy.linalg.null_space(alpha.T)
    null_X = X @ null_alpha

    # Obtain the top q-components of that null-space and set them to alpha_0, and the estimate noise to sigma
    from sklearn.decomposition import PCA
    pca = PCA(n_components=q)
    pca.fit(null_X)
    alpha_0_initialise = pca.components_.T
    noise_initialise = pca.noise_variance_

    # theta_0 is the list of euclidean parameters to initialise CFAD
    theta_0 = []

    # Initialising Lambda_y as diag(alpha^T@Cov(X|Y)@alpha) 
    # (Note: CFAD decomposes Lambda_y into its cholesky form (LL^T), hence the sqrt to initialise L)
    for i in range(K):
        theta_0.append(np.sqrt(np.diag((alpha[:,:d].T @ cov_sample[i] @ alpha[:,:d])).ravel()))

    # Initialising Lambda_0 as diag(alpha_0^T@Cov(null_X)@alpha_0)
    theta_0.append(np.sqrt(np.diag(alpha_0_initialise.T @ np.cov(null_X.T) @ alpha_0_initialise)))

    theta_0 = np.array(theta_0)
    theta_0 = theta_0.tolist()
    flatten = lambda l: [item for sublist in l for item in sublist]
    theta_0 = flatten(theta_0)

    # Appending initial guess for sigma
    theta_0.append(noise_initialise)
    theta_0 = np.array(theta_0)

    #theta_1 is the list of 
    theta_1 = np.zeros((X.shape[1],d+q))
    theta_1[:,:d] = alpha
    theta_1[d:,d:d+q] = alpha_0_initialise

    theta_init = [theta_0, theta_1]

    return theta_init


def csep(mean, mu, LambdaLambdaT, c_min, c_max):
    """ Function to check if newly sampled mean is c-separated with the list of means previouslty samples """
    """ Args:
        - mean: New sampled mean
        - mu: list of means already known
        - LambdaLambdaT: list of class-covariances 
        Returns:
        - Boolean Output: whether mean is c-separated from mu or not
    """
    n = len(mu)
    # Trace of covariance pertaining to the class for which the mean has been sampled
    Tr1 = np.trace(LambdaLambdaT[n])
    flag = True
    for i in range(n):
        Tr2 = np.trace(LambdaLambdaT[i])
        if not (la.norm(mean - mu[i])>=c_min*max(Tr1,Tr2) and la.norm(mean - mu[i])<=c_max*max(Tr1, Tr2)):
            flag = False
            break 
    return flag



def generate_data(seed, N, p, d, q, h, sep='low', smooth=False):
    """ Generates a synthetic dataset from the CFAD model
        Args:
        - seed: random seed to reproduce results
        - N: number of data samples
        - p: original data dimensionality
        - d: dimension of class-dependent subspace 
        - q: dimension of class-independent subspace 
        - h: number of slices/classes in y
        - sep: degree of separation of the means ('low', 'med', 'high')
        - smooth: whether or not to add the smoothness prior
        Returns:
        - X: data of the form pxN (p-dimensional, N samples) from CFAD
        - y: target labels
        - A: [alpha alpha_0]
    """

    # Generating a random alpha (pxd) such that alpha.T@alpha = I 
    # [A faster way to do this when alpha is "not-smooth" is to set alpha = Q from the QR decomposition of any matrix]
    A = genA(seed, p, d, q, smooth=smooth)

    # Define fraction of elements from each class
    pi = [0.3, 0.4, 0.3]

    # Define low-dimensional parameters (in the alpha subspace)
    LambdaLambdaT = [np.diag(np.array([2.0,4.0])), np.diag(np.array([5.0,3.0])), np.diag(np.array([2.0,2.0]))]
    # means are bounded such that: c_min max(tr_i,tr_j)<=||mu_i-mu_j||<=c_max max(tr_i,tr_j)
    if sep == 'low':
        c_min, c_max = 0.2, 0.5
    elif sep == 'med':
        c_min, c_max = 1, 3
    elif sep == 'high':
        c_min, c_max = 3, np.inf
    else:
        c_min, c_max = 0, np.inf
    mu = genmeans(seed, d, h, LambdaLambdaT, c_min, c_max)

    #Define covarinace in alpha_0 subspace
    Lambda0Lambda0T = np.diag(np.array([2.0,8.0,8.0]))

    #Define isotropic noise std
    sigma = np.random.rand(1)

    # For each class, generate samples (data in the alpha subspace)
    classes = np.random.choice(h, size=N, p=pi)
    samples = np.zeros((N, d))
    for k in range(h):
        # Indices of current class
        indices = (k == classes)
        # Number of samples for that class
        n_k = indices.sum()
        if n_k > 0:
            samples[indices] = np.random.multivariate_normal(mu[k], LambdaLambdaT[k], n_k)

    # Generate samples (data in the alpha0 subspace)
    samples0 = np.random.multivariate_normal(np.zeros((q)), Lambda0Lambda0T, N)
    samples = np.array(samples)
    samples0 = np.array(samples0)
    _w = np.concatenate((samples,samples0),axis=1)
    # Project into ambient space
    X  = A @ _w.T
    # Then add white (measurement) noise
    noise =  np.random.multivariate_normal(np.zeros(p), sigma*np.eye(p), N)
    X  = X + np.array(noise).T

    return X, classes, A


def genmeans(seed, d, h, LambdaLambdaT, c_min = 0, c_max = np.inf):
    """ Function to generate a random means which satisfy the c-separation criteria (args follow from gen_data)"""
    np.random.seed(seed)
    scale = 6
    mu = []
    mu.append(scale*np.random.randn(d))

    for k in range(h-1):
        for i in range(10000):
            mean = scale*np.random.randn(d)
            if csep(mean, mu, LambdaLambdaT, c_min, c_max):
                mu.append(mean)
                break
    return mu


def genA(seed, p, d, q, smooth=False):
    """ Function to generate a random instance of A = [alpha alpha_0] (args follow from gen_data)"""
    seed = np.random.seed(seed)
    # If smoothing prior is added, covariance of alpha (to fit the model) is (Moore-penrose) inverse of graoh-Laplacian (p)
    if smooth == True:
        cov = la.pinv(gen_laplacian(p))
    else:
        cov = np.eye(p)

    # Randomly sample first column of alpha
    alpha = np.zeros((p,d))
    alpha_col0 = np.random.multivariate_normal(np.zeros((p)), cov , (1)).T
    alpha_col0 = alpha_col0/np.linalg.norm(alpha_col0, axis=0)
    alpha[:, 0] = alpha_col0.ravel()

    # Sample subsequent columns such that they arw orthogonal to the previous one
    k=1
    for i in range(10000):
        alpha_k = np.random.multivariate_normal(np.zeros((p)), cov , (1)).T
        alpha_k= alpha_k/np.linalg.norm(alpha_k, axis=0)
        if (np.sum(np.abs(alpha_k.T @ alpha))<1e-3):
            alpha[:,k] = alpha_k.ravel()
            k = k+1
        if k==d:
            break

    # Sample alpha_0 from the nullspace of alpha
    A = np.zeros((p,p))
    A[:,:d] = np.array(alpha)
    alpha_0 = sla.null_space(A.T)
    A[:,d:] = alpha_0
    A = A[:,:(d+q)]

    return A



