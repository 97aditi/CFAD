import autograd.numpy as np
np.set_printoptions(precision=4)
from utils.utils import *
import warnings
warnings.filterwarnings("ignore")


# Number of datapoints 
N_list = [50, 101, 500, 1000, 5000, 10000]
# Define number of runs (random data generation) at every N 
iters = 100
# Define sepration between means ('low', 'med', 'high')
sep = 'low'
# Set to true to plot the results 
plot = True

for N in N_list:
    sirl = []
    savel = []
    drl = []
    ladl = []
    cfadl = []
    scfadl = []

    for seed in range(iters):
        np.random.seed(seed)

        """ Generate synthetic data from the CFAD model """
        #-----------------------------------------------------------------------------------------------------------------------------------
        # Dimension of measurement space
        p = 100
        # Dimension of class-dep and class-indep subspaces
        d = 2
        q = 3
        #Number of classes
        h = 3
        X, y, A = generate_data(seed, N, p, d, q, h, sep=sep, smooth=True)
        # Change to {N xp}
        X = X.T
        #---------------------------------------------------------------------------------------------------------------------------------------
        """ Run all methods on the data generated and save their principal subspace angles """

        from scipy.linalg import subspace_angles
 
        if N>p:
            """ SIR/SAVE/DR require at least $p+1$ samples """
            from sliced import SlicedInverseRegression
            sir = SlicedInverseRegression(n_directions=d)
            sir.fit(X, y)
            compang = subspace_angles(sir.directions_.T, A[:,:d])*(180/np.pi)
            sirl.append(compang[0])

            from sliced import SlicedAverageVarianceEstimation
            save = SlicedAverageVarianceEstimation(n_directions=d)
            save.fit(X, y)
            compang = subspace_angles(save.directions_.T, A[:,:d])*(180/np.pi)
            savel.append(compang[0])

            from methods.dr import DR
            alpha = DR(y, X, d, 'disc')
            compang = subspace_angles(alpha, A[:,:d])*(180/np.pi)
            drl.append(compang[0])

        if N>=h*(p+1):
            """ LAD require at least $h(p+1)$ samples """
            from methods.lad import LAD
            alpha = LAD(y, X, d, 'disc', 0)
            compang = subspace_angles(alpha, A[:,:d])*(180/np.pi)
            ladl.append(compang[0])


        from methods.cfad import CFAD
        alpha = CFAD(y, X.T, d, q)
        compang = subspace_angles(alpha, A[:,:d])*(180/np.pi)
        cfadl.append(compang[0])


        best_scfad = []
        for lamda in [1e-2, 1e-1, 1, 1e1, 1e2, 1e3]:
        	alpha = CFAD(y, X.T, d, q, lamda = lamda)
        	compang = subspace_angles(alpha, A[:,:d])*(180/np.pi)
        	best_scfad.append(compang[0])
        scfadl.append(min(best_scfad))
        	
        # ######################################################################################################

    # Save all results
    path = "results/"+sep+"s_"+str(N)+"_"
    if N>p:
        np.save(path + "SIR", sirl)
        np.save(path + "SAVE", savel)
        np.save(path + "DR", drl)
    if N>p*h:
        np.save(path + "LAD", ladl)
    np.save(path + "CFAD", cfadl)
    np.save(path + "sCFAD", scfadl)

if plot == True:
    from utils.plot_utils import *
    plotfig3(N_list, sep)
        