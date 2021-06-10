import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl 
mpl.use('TkAgg') 
import seaborn as sns
import pandas as pd 
import os

def plotfig2(N_list, sep):
	""" Plots the saved results for Figure 2"""
	labels = []
	nsamples = []
	angles = []

	for method in ["SIR", "SAVE", "DR", "LAD", "CFAD"]:
		for N in N_list:
			path = "results/"+sep + "_" +str(N)+"_" +method+".npy"
			arr = np.load(path).tolist()
			iters = len(arr)
			angles = angles+arr
			labels.extend([method]*iters)
			nsamples.extend([N]*iters)

	data_list = {"No. of samples": nsamples, "Method": labels, "Principal Subspace Angle": angles}
	df = pd.DataFrame(data_list, columns = ['No. of samples', 'Method', 'Principal Subspace Angle'])
	fig, ax = plt.subplots()
	sns.lineplot(x="No. of samples", y="Principal Subspace Angle", hue="Method", data=df, ax = ax)
	plt.xscale('log')
	plt.xlabel("$N$", fontsize=15)
	plt.ylabel("Principal Subspace Angle", fontsize=20) 
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles=handles[1:], labels=labels[1:], fontsize=15)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.grid()
	plt.show()


def plotfig3(N_list, sep):
	""" Plots the saved results for Figure 3"""
	labels = []
	nsamples = []
	angles = []
	
	for method in ["SIR", "SAVE", "DR", "LAD", "CFAD", "sCFAD"]:
		for N in N_list:
			path = "results/"+sep + "s_" +str(N)+"_" +method+".npy"
			if os.path.exists(path):
				arr = np.load(path).tolist()
				iters = len(arr)
				angles = angles+arr
				labels.extend([method]*iters)
				nsamples.extend([N]*iters)

	data_list = {"No. of samples": nsamples, "Method": labels, "Principal Subspace Angle": angles}
	df = pd.DataFrame(data_list, columns = ['No. of samples', 'Method', 'Principal Subspace Angle'])
	fig, ax = plt.subplots()
	sns.lineplot(x="No. of samples", y="Principal Subspace Angle", hue="Method", data=df, ax = ax)
	plt.xscale('log')
	plt.xlabel("$N$", fontsize=15)
	plt.ylabel("Principal Subspace Angle", fontsize=20) 
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles=handles[1:], labels=labels[1:], fontsize=15)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.grid()
	plt.show()



def plotfig4(subfig):
	""" Plots the saved results for Figure 4"""
	df = pd.read_pickle("results/Fig4"+subfig+".pkl")
	print(df)
	fig, ax = plt.subplots(figsize=(8.5,10))
	sns.lineplot(x="a", y="Principal Subspace Angle", hue="Method", data=df, ax = ax)
	plt.xlabel("$a$", fontsize=30)
	plt.ylabel("Principal Subspace Angle", fontsize=30) 
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles=handles[1:], labels=labels[1:], fontsize=20)
	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)
	plt.ylim(0,90)
	if subfig == 'a':
		plt.title("$Y = 4X_1/a + \epsilon$", fontsize=30)
	if subfig == 'b':
		plt.title("$Y = X_1^2/(20a) + 0.1\epsilon$", fontsize=30)
	if subfig == 'c':
		plt.title("$Y = X_1/(10a) + aX_1^2/100 + 0.6\epsilon$", fontsize=30)
	if subfig == 'd':
		plt.title("$Y = 0.4a(\\beta^\\top X) + 3\\sin(\\beta_2^\\top X/4) + 0.2\\epsilon$", fontsize=30)
	plt.grid()
	plt.show()


