Code for the paper: Factor-analytic inverse regression for high-dimension, small-sample dimensionality reduction

Contains an environment.yml with all required packages:
- Run 'conda env create -f environment.yml' to create an environment will all dependencies for running the scripts.
- To manually install packages, requirements are listed as follows:
  - python 3.8
  - autograd 1.3
  - joblib 0.14
  - matplotlib 3.10
  - nibabel 3.0
  - numpy 1.18
  - pandas 1.0
  - scikit-learn 0.22
  - scipy 1.4
  - seaborn 0.10
  - sliced 0.7
  - pymanopt (included with package, note that the pymanopt folder includes a routine for Riemannian LBFGS)

Required:
  To run hcp_run.py which outputs results for Table 4 in the paper, HCP working memory dataset is required which is openly available post registration at http://www.humanconnectomeproject.org. 

------------------------------------------------------------------------------------------------------------------------------------------------------------------
Contains scripts to generate figures (2,3,4) in the paper and to obtain values for the tables:

- Fig2.py generates the subplots from Fig.2 in the paper. The degree of separation can be selected in the script to generate the respective plot.

- Fig3.py generates the subplots from Fig.3 in the paper. The degree of separation can be selected in the script to generate the respective plot.

- Fig4.py generates the subplots from Fig.4 in the paper. Select 'a'/'b'/'c'/'d' in the script to generate the respective subplot (from left to right in the figure).

- tablefmri.py computes values for Tables 2 and 3. Pass arguments for: subject number, method name, $d$, and $q$ (only for CFAD and sCFAD). 
  The script will output classification accuracy for this setting. 
  Example: python table_fmri.py --sub 1 --method CFAD --d 10 --q 13
  Note: In case of RRR, the variable $d$ captures the reduced rank, it can be set between (0, n_classes).

- hcp_run.py computes values for Table 4. Pass arguments for: subject number, method name, $d$, lamda (for CFAD and RRR) ($q$ is set automatically in this code)
  The script will output classification accuracy for this setting. 
  Example: python hcp_run.py --sub 102311 --method CFAD --d 10 

All scripts have detailed instructions embedded.
------------------------------------------------------------------------------------------------------------------------------------------------------------------ 
  
Other folders:
  - methods: contains implementations of LAD/LOL/DR/CFAD/RRR (SIR and SAVE are included in the Sliced package; LDA and PCA are available in scikit-learn)
    (Note: there are two versions of CFAD with autograd and torch backend; torch is faster for higher-d)
  - utils: contains utils.py and plot_utils.py 
  - pymanopt: added our implementation of Riemannian LBFGS (not included in the original pymanopt package)
  - results: all scripts store their results in this folder

----------------------------------------------------------------------------------------------------------------------------------------------------------------  
