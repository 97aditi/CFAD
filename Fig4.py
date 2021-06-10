import scipy
import numpy as np 
from sliced import SlicedInverseRegression
from sliced import SlicedAverageVarianceEstimation
from methods.lad import LAD
from methods.dr import DR
from methods.cfad import CFAD
from utils.utils import *
from utils.plot_utils import *
import warnings
warnings.filterwarnings("ignore")


""" This code generates subplots from Fig 4: Performnace of CFAD and other SDR methods on different regressive relationships between X and y """
""" (at a range of values of $a$)"""

# Choose which subfig to generate ('a', 'b', 'c', 'd')
subfig = 'b'
# Set to true to plot the results 
plot = True

# Number of samples
nsamples = 500
# Data dimensionality
p = 8
# Number of repititions for each method
nrep = 20

# Set 'd', projection matrix beta (p x d), and range of a (a parameter for X-y relationships)
if subfig == 'a' or subfig == 'b' or subfig == 'c':
    if subfig == 'c':
        amax = 10
    else:
        amax = 20
    d = 1
    beta = np.zeros((p,d))
    beta[0,0] =1
else:
    amax = 14
    d = 2
    beta = np.zeros((p,d))
    beta[0:3, 0], beta[0, 1], beta[6, 1], beta[7, 1] = 1, 1, 1, 1
# Set number of slices
h = 5


# The names of the methods with their corresponding subspace angle (between estimated and true beta) along with the value of parameter a are stored for plotting
names = []
angles = []
alist = []

for a in range(amax):
    a = a + 1
    alist.extend([a]*nrep*1)
    
    for j in range(nrep):
        print("Doing rep: "+str(j))
        np.random.seed(j)
        # Generate X
        X = np.random.randn(nsamples, p)
        # Define the relationship between X and y
        if subfig == 'a':
            yr = 4*X[:,0]/(a)
            # Adding noise
            y = np.array(yr) + np.random.randn(nsamples)
        if subfig == 'b':
            yr = (X[:,0]**2)/(20*a)
            # Adding noise
            y = np.array(yr) + (0.1)**2*np.random.randn(nsamples)
        if subfig == 'c':
            yr = (X[:,0])/(10*a) + a*(X[:,0]**2)/100 
            # Adding noise
            y = np.array(yr) + (0.6)**2*np.random.randn(nsamples)
        if subfig == 'd':
            yr = 0.4*a*(X@beta[:, 0])**2 + 3*np.sin(X@beta[:, 1]/4)
            # Adding noise
            y = np.array(yr) + (0.2)**2*np.random.randn(nsamples)


        # Run the list of algorithms and compare thier estimated beta to actual beta
        from scipy.linalg import subspace_angles

        sir = SlicedInverseRegression(n_directions=d, n_slices=h)
        sir.fit(X, y)
        compang = subspace_angles(sir.directions_.T, beta)*(180/np.pi)
        names.append('SIR')
        angles.append(compang[0])

        save = SlicedAverageVarianceEstimation(n_directions=d, n_slices=h)
        save.fit(X, y)
        compang = subspace_angles(save.directions_.T, beta)*(180/np.pi)
        names.append('SAVE')
        angles.append(compang[0])


        est_beta = DR(y, X, d, 'cont', h)
        compang = subspace_angles(est_beta, beta)*(180/np.pi)
        names.append('DR')
        angles.append(compang[0])


        est_beta = LAD(y, X, d, 'cont', h)
        compang = subspace_angles(est_beta, beta)*(180/np.pi)
        names.append('LAD')
        angles.append(compang)

        #Set q = p-d i.e. the nullspace of beta
        q = p-d
        est_beta = CFAD(y-1, X.T, d, q)
        compang = subspace_angles(est_beta, beta)*(180/np.pi)
        names.append('CFAD')
        angles.append(compang[0])



# Include GLLiM results from MATLAB
import scipy.io
alistgllim = scipy.io.loadmat('alist2b.mat')
anglesgllim = scipy.io.loadmat('angles2b.mat')

alistg = np.array(alistgllim['alist'].T[:-20]).ravel().tolist()
anglesg = np.array(anglesgllim['angles'].T[:-20]*(180/np.pi)).ravel().tolist()

alist = alist + alistg
angles = angles + anglesg

names = names + ['GLLiM']*len(alist)


# Save results in a dataframe
import pandas as pd
data_list = {'a': np.array(alist), 'Method': names, 'Principal Subspace Angle': angles}
df = pd.DataFrame(data_list, columns = ['a', 'Method', 'Principal Subspace Angle'])
df.to_pickle("results/Fig4"+subfig+".pkl")

# df = pd.read_pickle("results/Fig4"+subfig+".pkl")

# Plot results
if plot:
    plotfig4(subfig)

    
