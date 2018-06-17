import numpy as np
import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
import seaborn as sns

#-----------------------------PRE-EXPLORATION---------------------------#

def pandasReport(df,path):
	pandas_profiling.ProfileReport(df)
	rejected_variables = profile.get_rejected_variables(threshold=0.99)
	profile.to_file(outputfile=path)


def heatmap(df):
	corr = df.corr()
	f, ax = plt.subplots(figsize=(10, 8))		
	sns.heatmap(corr, 
				mask=np.zeros_like(corr, dtype=np.bool),
				cmap=sns.diverging_palette(250, 20, sep=20,as_cmap=True),
				square=True, ax=ax)
	plt.title('Sensor Data Correlations Heat Map', 
			fontsize=15, 
			fontweight='bold')
	plt.gcf().subplots_adjust(bottom=0.15)
	plt.tight_layout()
	plt.show()
	
def plotall(df):
	df = df.fillna(0)
	sns.pairplot(df)
	plt.show()
	
def HowManyBlanks(df):
	print(df.isnull().sum())

def plotTime(df,target,stle,datarange):
	if daterange == None:
		df.plot(y=target, style=stle)
	else:
		df_pd.loc[df_pd.index > daterange].plot(y=target, style=stle)
		df_pd.plot
		df_corrected.loc[df[target].isnull(), target] = df_pred

#------------------------MODEL VISUALIZATIONS---------------------------------#	
from mpl_toolkits.mplot3d import Axes3D

def outlier_plot(df,x,y,z):
	l = [1,2,3]
	fig = plt.figure()
	ax = Axes3D(fig)
	plot=ax.scatter(l,l,l)
	return plot
	
def plot(x):
	p=plt.plot(range(x))
	return p

'''ax.scatter(df.loc[df['outliers'] == 1,x],
				   df.loc[df['outliers'] == 1,y],
				   df.loc[df['outliers'] == 1,z],
				   'blue')
				   
				   
				   
				   
				   
'''				   
'''
	ax.scatter(df.loc[df['outliers'] == -1,x],
				   df.loc[df['outliers'] == -1,y],
				   df.loc[df['outliers'] == -1,z],
				   'red')
	plt.show()



'''
'''
plt.plot(df['c3h8'].loc['outlier' == -1],df['c4h10'].loc[p==-1],'rx')
plt.plot(df['c3h8'].loc['outler' == 1],df['c4h10'].loc[p==1],'kx')
'''
	
