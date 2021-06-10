""" This code has been taken from nibabel's tutorial and modified for this paper
-------------------------------------------------------------------------------------------------------------------------
    Generates fMRI classification accuracy for a given subject at a given d (and q) for a specified method using 5-fold CV
    (If the selected method is sCFAD, the best lambda is selected using cross validation.)
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sub', type=int, default='6',
                    help='Enter subject number (1--6)')
parser.add_argument('--method', type=str, default='SIR',
                    help='Enter dimensionality reduction method to use (SIR, SAVE, DR, LAD, LOL, PCA, CFAD, sCFAD)')
parser.add_argument('--d', type=int, default = 10,
                    help='Enter latent space dimension (d)')
parser.add_argument('--q', type=int, default='0',
                    help='Enter dimension of class-independent subspace (only required for CFAD/sCFAD)')

args = parser.parse_args()

"""
Decoding with Dimension Reduction + SVM: 
===============================================================
"""
#############################################################################
# Retrieve the files of the Haxby dataset
# ----------------------------------------
from nilearn import datasets
import numpy as np

np.random.seed(10)


# Select subject 
haxby_dataset = datasets.fetch_haxby(subjects=(args.sub,))

# print basic information on the dataset
print('Mask nifti image (3D) is located at: %s' % haxby_dataset.mask)
print('Functional nifti image (4D) is located at: %s' %
      haxby_dataset.func[0])

#############################################################################
# Load the behavioral data
# -------------------------
import pandas as pd

# Load target information as string and give a numerical identifier to each
behavioral = pd.read_csv(haxby_dataset.session_target[0], sep=" ")
conditions = behavioral['labels']

# Choosing all 8 classes here
condition_mask = behavioral['labels'].isin(['cat', 'face', 'house', 'chair', 'bottle', 'shoe', 'scissors', 'scrambledpix'])
#condition_mask = behavioral['labels'].isin(['shoe', 'cat', 'bottle'])
conditions = conditions[condition_mask]

# Record these as an array of sessions, with fields
# for condition chosed and run
session = behavioral[condition_mask].to_records(index=False)

#############################################################################
# Prepare the fMRI data: smooth and apply the mask
# -------------------------------------------------
from nilearn.input_data import NiftiMasker

mask_filename = haxby_dataset.mask_vt[0]
# For decoding, standardizing is often very important
# note that we are also smoothing the data
masker = NiftiMasker(mask_img=mask_filename, smoothing_fwhm=4,
                     standardize=True, memory="nilearn_cache", memory_level=1)
func_filename = haxby_dataset.func[0]
X = masker.fit_transform(func_filename)
X = X[condition_mask]
print(X.shape)

# Changing labels to numbers
labels = conditions.array
classes = []
for i in range(X.shape[0]):
    if labels[i]=='face':
        classes.append(0)
    if labels[i]=='house':
        classes.append(1)
    if labels[i]=='chair':
        classes.append(2)
    if labels[i]=='bottle':
        classes.append(3)
    if labels[i]=='cat':
        classes.append(4)
    if labels[i]=='shoe':
        classes.append(5)
    if labels[i]=='scissors':
        classes.append(6)
    if labels[i]=='scrambledpix':
        classes.append(7)
y = np.array(classes)
#############################################################################
# Printing the number of components that explain 90% variance
# ------------------------------------------------------------
# Useful for CFAD, to decide $d+q$

from sklearn.decomposition import PCA

for i in range(X.shape[1]):
    pca = PCA(n_components = i+1)
    pca.fit(X)
    if np.sum(pca.explained_variance_ratio_) >= 0.9:
        print("90% variance explained by "+str(i+1)+" components")
        break
    # if pca.explained_variance_ratio_ >= 0.95:
    #     print("95% variance explained by "+str(i+1)+" components")
    #     break
#############################################################################
# Define a dimension reduction method here
#------------------------------------------

from utils.utils import *

# Extract $d$ and selected method from arguments
d = args.d 
method = args.method

from sklearn.model_selection import StratifiedKFold, GroupKFold


def eval_model (method, lamda = 0):
    """ Chooses number of folds, split data into train val folds and reports mean validation accuracy using the specified method
        Args: 
        - method: name of specified method given by user
        - Lambda: required only for sCFAD
        Outputs:
        - mean validation accuracy over 5 folds 
    """
    skf = GroupKFold(n_splits=5)
    scores = []
    for train_index, test_index in skf.split(X, y, session):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

        Xmean = np.mean(X_train, axis=0)
        X_train = X_train - Xmean
        X_test =  X_test - Xmean
        # Estimate alpha using a dimenionality reduction method
        if method == 'SIR':
            from sliced import SlicedInverseRegression
            sir = SlicedInverseRegression(n_directions=d)
            sir.fit(X_train, y_train)
            alpha = sir.directions_.T
        if method == 'SAVE': 
            from sliced import SlicedAverageVarianceEstimation
            save = SlicedAverageVarianceEstimation(n_directions=d)
            save.fit(X_train, y_train)
            alpha = save.directions_.T
        if method == 'DR':
            from methods.dr import DR
            alpha = DR(y_train, X_train, d, 'disc')
        if method == 'LAD':
            from methods.lad import LAD
            alpha = LAD(y_train, X_train, d, 'disc')
        if method == 'LOL':
            from methods.lol import LOL
            lol = LOL(n_components = d, orthogonalize=True)
            X_lol, alpha = lol.fit_transform(X_train, y_train+1)
        if method == 'PCA':
            pca = PCA(n_components = d)
            pca.fit(X_train)
            alpha = pca.components_.T
        if method == 'LDA':
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            lda = LinearDiscriminantAnalysis(solver='eigen')
            lda.fit(X_train, y_train)
            alpha = lda.scalings_[:,:d]
        if method == 'RRR':
            from sklearn.preprocessing import OneHotEncoder
            from methods.rrr import ReducedRankRegressor
            enc = OneHotEncoder(handle_unknown='ignore')
            y_train_one = enc.fit_transform(y_train.reshape(-1,1)).toarray()
            rrr = ReducedRankRegressor(X_train, y_train_one, rank=d, reg=lamda)
            y_pred, alpha = rrr.predict(X_test)
        if method == 'CFAD':
            from methods.cfad import CFAD
            q = args.q
            theta_init = init_cfad(X_train, y_train, X_test, y_test, d, q)
            alpha = CFAD(y_train, X_train.T, d,theta_init=theta_init[1][:,:d])
        if method == 'sCFAD':
            from methods.cfad import CFAD
            q = args.q
            theta_init = init_cfad(X_train, y_train, X_test, y_test, d, q)
            alpha = CFAD(y_train, X_train.T, d, q, theta_init, lamda)
            

        X_test_low = X_test @ alpha
        X_train_low = X_train @ alpha
        from sklearn.svm import SVC
        svc = SVC(kernel='linear')
        svc.fit(X_train_low, y_train)
        scores.append(svc.score(X_test_low, y_test))

    return np.mean(scores)


from utils.utils import initialise_cfad
# To initialise CFAD/sCFAD
def init_cfad(X_train, y_train, X_test, y_test, d, q):
    """ Function to initialise CFAD"""
    from sliced import SlicedInverseRegression
    sir = SlicedInverseRegression(n_directions=d)
    sir.fit(X_train, y_train)
    alpha = sir.directions_.T
    theta_init = initialise_cfad(10, X_train, y_train, d, q, alpha)
    return theta_init


if method == 'sCFAD':
    """ Call eval_model parallely for all lambdas and report the best"""
    lamdas = [1e-3, 1e-2, 1e-1, 1, 1e2, 1e3]
    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    processed_list = Parallel(n_jobs=num_cores)(delayed(eval_model)(method, l) for l in lamdas)
    class_acc = np.max (processed_list)
    print("For method "+method+", score: "+str(class_acc))
if method == 'RRR':
    lamdas = [1e-3, 1e-2, 1e-1, 0, 1, 1e2, 1e3]
    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    processed_list = Parallel(n_jobs=num_cores)(delayed(eval_model)(method, l) for l in lamdas)
    class_acc = np.max (processed_list)
    print("For method "+method+", score: "+str(class_acc))
else:     
    """ Call eval_model and prints accuracy"""
    class_acc = eval_model(method)
    print("For method "+method+", score: "+str(class_acc))
#############################################################################
