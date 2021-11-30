import numpy as np
from scipy import interpolate

def find_linearly_interpolated_row(df):
	"""This function adds a row using linear interpolation
	Parameters
	----------
		df: DataFrame
			Dataframe for which linear interpolation needs to be done
	"""
	interpolated_row = []
	for i in range(0, len(df)):
		interpolated_column = []
		for k in range(0, len(df[0][1])):
			x = [df[i][j][k] for j in range(0, len(df[0]))]
			yy = np.arange(0,len(df[0]),1)
			yn = np.arange(0,len(df[0])+1,1)
			f = interpolate.interp1d(yy, x, fill_value='extrapolate')(yn)
			interpolated_column.append(f[len(df[0])])
		interpolated_row.append(np.array(interpolated_column))
	return np.array(interpolated_row)

	

	
