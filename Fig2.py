import autograd.numpy as np
np.set_printoptions(precision=4)
from utils.utils import *

# Number of datapoints
N_list = [500, 1000, 5000, 10000]
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
        X, y, A = generate_data(seed, N, p, d, q, h,sep=sep)
        # Change to {N xp}
        X = X.T
        #---------------------------------------------------------------------------------------------------------------------------------------
        """ Run all methods on the data generated and save their principal subspace angles """

        from scipy.linalg import subspace_angles
 
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
        Wopt = DR(y, X, d, 'disc')
        compang = subspace_angles(Wopt, A[:,:d])*(180/np.pi)
        drl.append(compang[0])

        from methods.lad import LAD
        Wopt = LAD(y, X, d, 'disc', 0)
        compang = subspace_angles(Wopt, A[:,:d])*(180/np.pi)
        ladl.append(compang[0])

        from methods.cfad import CFAD
        Xopt = CFAD(y, X.T, d, q)
        compang = subspace_angles(Xopt, A[:,:d])*(180/np.pi)
        cfadl.append(compang[0])


        # ######################################################################################################
    # Printing mean subspace angle at a given N for every method
    print("For N=" + str(N))
    print(np.mean(sirl))
    print(np.mean(savel))
    print(np.mean(drl))
    print(np.mean(ladl))
    print(np.mean(cfadl))

    # Save all results
    path = "results/"+sep+"_"+str(N)+"_"
    np.save(path + "SIR", sirl)
    np.save(path + "SAVE", savel)
    np.save(path + "DR", drl)
    np.save(path + "LAD", ladl)
    np.save(path + "CFAD", cfadl)


if plot == True:
    from utils.plot_utils import *
    plotfig2(N_list, sep)
        
        