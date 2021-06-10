import numpy as np 
import scipy.io as sio
from utils import utils
import warnings
import argparse

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--sub', type=int, default='102311',
                    help='Enter subject ID (102311/100206/100307/100610/101107/100408/101006/101309/101410/101915)')
parser.add_argument('--d', type=int, default = 10,
                    help='Enter latent space dimension (d)')
parser.add_argument('--method', type=str, default = 'CFAD',
                    help='Enter desired dimensionality reduction method (LDA/PCA/LOL/RRR/CFAD)')
parser.add_argument('--lamda', type=float, default=0,
                    help='Enter smoothness coefficient/regularisation coefficient (applicable only for CFAD and RRR)')

#Note: To run sCFAD, select CFAD with a non-zero lamda

args = parser.parse_args()

print("Running " +str(args.method) + " at lamda " + str(args.lamda) +" for sub " +str(args.sub))


# Seeding for reproducibility
seed = 10
np.random.seed(seed)

# Load mask
mask_contents = sio.loadmat("HCP_WM/WM_mask0.mat")
mask = mask_contents['mask']

# Load HCP_WM data for a subject
subid = args.sub
mat_content = sio.loadmat("HCP_WM/hcp_sub" + str(subid) +"_WM.mat")
wholebrain = mat_content['wholebrain']
y = mat_content['labels'].ravel()
coords = mat_content['coords']

ind = np.where(mask==1)[0]
# Data of shape Nxp; mask to select voxels
X = wholebrain[ind,:].T
xgrid = coords[ind,:]

# Removing resting state data
samples = np.where(y>0)[0]
y = y[samples]
X = X[samples]


d = args.d
print("Setting d = "+str(d))

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from methods.cfad_torch import cfad
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA


def run_a_fold(X_train, y_train, X_test, y_test, d, method = 'CFAD', lamda = 0):
    """ Function to run one CV fold
        Args:
        X_train, y_train: Train dataset and its true labels
        X_test, y_test: Test dataset and its true labels
        d: desired reduced dimensionality of data
        method: Dimensionality reduction method to use
        lamda: Smoothness coefficient, applicable only when using sCFAD
        Outputs:
        Classification score over one fold
    """

    # Zero meaning the data
    Xmean = np.mean(X_train, axis=0)
    X_train = X_train - Xmean
    X_test =  X_test - Xmean

    if method == 'LDA':
        lda = LinearDiscriminantAnalysis(n_components=d)
        lda.fit(X_train, y_train)
        alpha = lda.scalings_[:,:d]


    if method == 'PCA':
        pca = PCA(n_components = d)
        pca.fit(X_train)
        alpha = pca.components_.T


    if method == 'LOL':
        from methods.lol import LOL
        lol = LOL(n_components = d, orthogonalize=True)
        X_lol, alpha = lol.fit_transform(X_train, y_train)


    if method == 'RRR':
        from sklearn.preprocessing import OneHotEncoder
        from methods.rrr import ReducedRankRegressor
        enc = OneHotEncoder(handle_unknown='ignore')
        y_train_one = enc.fit_transform(y_train.reshape(-1,1)).toarray()
        rrr = ReducedRankRegressor(X_train, y_train_one, rank=d, reg=lamda)
        y_pred, alpha = rrr.predict(X_test)


    if method == 'CFAD':
        # Computing number of components required to explain 905 variance in data
        n_comp = 0
        for i in range(X_train.shape[1]):
            pca = PCA(n_components=i+1)
            pca.fit(X_train)
            var = np.sum(pca.explained_variance_ratio_)
            if var>0.9:
                n_comp = i+1
                break 

        q = n_comp-d
        print("Setting q = "+str(q))

        # Initialising with PCA first
        pca = PCA(n_components = d)
        pca.fit(X_train)
        alpha = pca.components_.T
        # y_train-1 because CFAD wants class labels starting from 0
        theta_init = utils.initialise_cfad(seed, X_train, y_train-1, d, q, alpha)
        # Applying CFAD
        alpha = cfad(y_train-1, X_train.T, d, q, theta_init, lamda=lamda)


    # Training an SVM on this
    svc = SVC(kernel='linear')
    # Projectin data to d-dimensional subspace spanned by the estimated alpha
    X_train_low = X_train@alpha
    X_test_low = X_test@alpha
    svc.fit(X_train_low, y_train)
    sv_score = svc.score(X_test_low, y_test)

    return sv_score
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



import multiprocessing
from joblib import Parallel, delayed

skf = StratifiedKFold(n_splits=3)

scores = []
for train_index, test_index in skf.split(X, y):
    scores.append(run_a_fold(X[train_index], y[train_index], X[test_index], y[test_index], args.d, args.method, args.lamda))

print(np.mean(scores))


