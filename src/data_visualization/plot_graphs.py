import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def plot_loss_graph(i_values, loss_train, loss_val, title, x_label, y_label):
	"""Plots loss vs hyperparameter value graph.
	Parameters
	---------
		i_values: list
			list of indexes
		loss_train: list
			list of training set losses
		loss_val: list
			list of validation set losses
	"""
	plt.plot(i_values, loss_train, "-b", label="loss_train")
	plt.plot(i_values, loss_val, "-g", label="loss_validation")
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.legend()
	plt.show()

def plot_hist(data_set, title):
	"""This function plots the histogram of a dataset.
	Parameters
	----------
		data_set: 2D array
			Dataset whose histogram needs to be plotted
		title: String
			Title of histogram plot
	"""
	figure(figsize=(10, 5), dpi=80)
	plt.hist(data_set, bins = 10)
	plt.title(title)
	plt.xlabel("bins")
	plt.ylabel("Number of data")
	plt.show()

def plot_all_dataset_hist(train_set, val_set, test_set, title):
	"""This function plots the histogram of all train set, validation set and test set at once in a subplot.
	Parameters
	----------
		train_set: 2D array
			Train set whose histogram needs to be plotted
		val_set: 2D array
			Validation set whose histogram needs to be plotted
		test_set: 2D array
			Test set whose histogram needs to be plotted
		title: String
			Title of histogram plot
	"""
	f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex='col', sharey='row', figsize=(25,7)) # gets the current figure and then the axes

	f.suptitle(title, fontsize=10)
	ax1.hist(train_set, bins = 10)
	ax1.set_title("Train set", fontsize=10)
	ax1.set_xlabel("bins", fontsize=10)
	ax1.set_ylabel("Number of data", fontsize=10)

	ax2.hist(val_set, bins = 10)
	ax2.set_title("Validation set", fontsize=10)
	ax2.set_xlabel("bins", fontsize=10)
	ax2.set_ylabel("Number of data", fontsize=10)

	ax3.hist(test_set, bins = 10)
	ax3.set_title("Test set", fontsize=10)
	ax3.set_xlabel("bins", fontsize=10)
	ax3.set_ylabel("Number of data", fontsize=10)	
	
	plt.show()

def visualize_all_attributes(df):
	"""
	This function plots the attribute graph.
	Parameters
	----------
		Dataframe whose attribute graph is to be plotted.
	"""
	plt.rcParams['figure.figsize'] = [10, 15]
	fig, axes = plt.subplots(nrows=4, ncols=3)

	df.hist(column = 'ACE_Bx', ax=axes[0,0])
	df.hist(column = 'ACE_By', ax=axes[0,1])
	df.hist(column = 'ACE_Bz', ax=axes[0,2])
	df.hist(column = 'ACE_x', ax=axes[1,0])
	df.hist(column = 'ACE_y', ax=axes[1,1])
	df.hist(column = 'ACE_z', ax=axes[1,2])
	df.hist(column = 'ACE_Vx', ax=axes[2,0])
	df.hist(column = 'ACE_Temp', ax=axes[2,1])
	df.hist(column = 'MMS_x', ax=axes[2,2])
	df.hist(column = 'MMS_y', ax=axes[3,0])
	df.hist(column = 'MMS_z', ax=axes[3,1])
	df.hist(column = 'Delay', ax=axes[3,2])
