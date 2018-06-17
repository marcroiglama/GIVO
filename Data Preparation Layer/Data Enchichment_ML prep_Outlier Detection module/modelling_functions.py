from __future__ import division, print_function
import time
from IPython.display import display

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from sklearn.externals import joblib
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------BASICS-----------------------------------
	
def readCSV(filename):
	df = pd.read_csv(filename)
	df = df.set_index(pd.DatetimeIndex(df['timestamp']))
	df = df.drop(['timestamp'], axis=1)
	return df
	

def error(y_test, y_pred):
	mse = metrics.mean_squared_error(y_test, y_pred)
	rmse = np.sqrt(mse)
	print('MSE: ', mse, '\nRMSE: ', rmse)

# -------------------------DATA ENRICHMENT------------------------------

def createValidationSet(df):
	rand_values = np.random.random(df.shape)
	cond = (rand_values < .1)
	df_validation = df.mask(cond)
	df_validation['value set to NaN'] = cond
	return df_validation


def interpolate(df,method,l,d):
	if d != None:
		interpolations = pd.DataFrame(df.interpolate(method='polynomial',
													 limit=l,
													 order = int(d), 
													 limit_area='inside'))
	else:
		interpolations = pd.DataFrame(df.interpolate(method=method,
													 limit=l, 
													 limit_area='inside'))
	interpolations.columns = [method]
	return interpolations


def dropNotInterpolatedBlancs(df):
	'''
	drop not interpolated values, save as interpolated_sensors.csv,
	standarize data, save as standarized_sensors.csv,
	split into train and test sets, save as train.csv and test.csv
	'''
	df = mf.readCSV('interpolated_sensors.csv')
	print('data after interpolation')
	print('# of records: '+ str(df.shape[0]) + 
		  '\t # of variables: ' + str(df.shape[1]))
	print(df.isnull().sum())

	df = df.dropna()
	print('data after drop blancs that interpolation can not solve')
	print('# of records: '+ str(df.shape[0]) + 
		  '\t # of variables: ' + str(df.shape[1]))
	print(df.isnull().sum())
	return df


#------------------------------PREPARE FOR ML---------------------------

def splitData(df,test_volume):
	df_train, df_test = train_test_split(df, test_size=test_volume)	
	df_train.is_copy = False
	df_test.is_copy = False
	return df_train,df_test


def fitStandardScaler(df):
	scaler = StandardScaler()
	scaler.fit(df.copy())
	joblib.dump(scaler, 'StandardScaler/StandardScaler.sav')
	print('\n StandardScaler is ready to normalize your data,')
	print('find the model at StandardScaler/StandardScaler.sav')


def standarizing(df):
	scaler = joblib.load('StandardScaler/StandardScaler.sav')	
	std = scaler.transform(df)
	df_std = pd.DataFrame(data = std,
					  index = df.index,
					  columns = df.columns)
	return df_std


def inverseStandarizing(df_std):
	scaler = joblib.load('StandardScaler/StandardScaler.sav')
	original = scaler.inverse_transform(df_std)
	df_original = pd.DataFrame(data = original,
							   index = df_std.index,
							   columns = df_std.columns)
	return df_original


	
# ---------------------DIMENSIONALITY REDUCTION-------------------------

def pca(df_train, df_test):
	pca = PCA(n_components=2)
	pca.fit(df_train)
	principalComponents = pca.transform(df_train)
	
	df_pca_train = pd.DataFrame(data = principalComponents, 
						  columns = ['pc1', 'pc2'], 
						  index = df_train.index)

	principalComponents = pca.transform(df_test)
	
	df_pca_test = pd.DataFrame(data = principalComponents, 
						  columns = ['pc1', 'pc2'], 
						  index = df_test.index)				  
	return df_pca_train, df_pca_test


def tsne(df_train, df_test):
	print('start TSNE')
	time_start = time.time()

	df = pd.concat([df_train, df_test])
	
	tsne = TSNE(n_components=2, verbose=1, n_iter=1000, perplexity=35)
	principalComponents = tsne.fit_transform(df)	
	
	print ('t-SNE done! Time elapsed: {} seconds'.format(
			time.time()-time_start))
			
	df_tsne = pd.DataFrame(data = principalComponents,
						   columns = ['c1', 'c2'],
						   index = df.index)
						   
	df_tsne_train = df_tsne[:len(df_train)]
	df_tsne_test = df_tsne[len(df_test):]
	return df_tsne_train, df_tsne_test

# -------------------------MACHINE LEARNING-----------------------------

def instanceModel(df_train, folder, model_file, c):
	
	if folder == 'IsolationForest':
		model = IsolationForest(n_estimators=100, verbose=0, 
								contamination=c, max_samples =0.03, 
								max_features = 1.0)	
								
	elif folder == 'EllipticEnvelope':
		model = EllipticEnvelope(contamination=c, 
								 store_precision=False)
								 
	elif folder == 'LocalOutlierFactor':
		k = int(len(df_train)*c)
		model = LocalOutlierFactor(n_neighbors=k, algorithm='auto',
								   leaf_size=30, metric='euclidean',
								   metric_params=None,
								   contamination=c)
								   
	elif folder == 'OneClassSVM':
		model = OneClassSVM(kernel='rbf', degree=3, nu=c, 
						    cache_size=500, verbose=0)
	
	model.fit(df_train)
	
	joblib.dump(model, folder + '/' + model_file)
	
	print('\n'+ folder + ' trained succesfully, saved as ' + model_file)
 

def predict(df, folder, pred_file, clf):
	if folder == 'LocalOutlierFactor':
		predictions = clf._predict(df) 
		
	else:
		predictions = clf.predict(df)
	
	predictions = pd.DataFrame(data = predictions, columns = ['outliers'],
							   index = df.index)
	
	predictions.to_csv(folder + '/' + pred_file)
	
	print('\nPredictions done succesfully, saved as ' 
		 + folder + '/' + pred_file)	

	
def ensemble(algorithms,c,dataset):
	print('\n##############    ENSEMBLE CALCULATIONS   ##################')
	
	df_ensemble = pd.DataFrame()
	
	for algorithm in algorithms:
		folder = algorithm
		file_name = dataset + '_' + algorithm + '_' + str(c) + '.csv'
		path = folder + '/' + file_name
		df_ensemble[algorithm] = readCSV(path)['outliers']
		
		df_ensemble.loc[df_ensemble[algorithm] == 1, algorithm] = 0
		df_ensemble.loc[df_ensemble[algorithm] == -1, algorithm] = 1
		
		print('\n' + file_name + ' outliers: ', 
			  df_ensemble.loc[df_ensemble[algorithm] == 1, 
							  algorithm].count())
		
	df_ensemble.index = readCSV(path).index	
	
	df_ensemble['voting'] = df_ensemble.sum(axis=1)
	
	df_ensemble['ee_VS_if'] = df_ensemble[['EllipticEnvelope',
										   'IsolationForest']].sum(axis=1)
	df_ensemble['ee_VS_lof'] = df_ensemble[['EllipticEnvelope',
											'LocalOutlierFactor']].sum(axis=1)
	df_ensemble['if_VS_lof'] =  df_ensemble[['IsolationForest',
											 'LocalOutlierFactor']].sum(axis=1)
	
	df_ensemble.loc[df_ensemble['ee_VS_if'] == 2, 'ee_VS_if'].count()	

	ee_out = df_ensemble.loc[df_ensemble['EllipticEnvelope'] == 1,
										 algorithm].count()
	if_out = df_ensemble.loc[df_ensemble['IsolationForest'] == 1,
										 algorithm].count()
	lof_out = df_ensemble.loc[df_ensemble['LocalOutlierFactor'] == 1, 
										  algorithm].count()
	
	ee_VS_if = df_ensemble.loc[df_ensemble['ee_VS_if'] == 2, 'ee_VS_if'].count()
	ee_VS_lof = df_ensemble.loc[df_ensemble['ee_VS_lof'] == 2, 'ee_VS_lof'].count()
	if_VS_lof = df_ensemble.loc[df_ensemble['if_VS_lof'] == 2, 'if_VS_lof'].count()
	
	total = df_ensemble.loc[df_ensemble['voting'] == 3, 'voting'].count()

	venn3(( ee_out - ee_VS_if - ee_VS_lof + total,
			if_out - ee_VS_if - if_VS_lof + total,
			ee_VS_if - total,
			lof_out - ee_VS_lof - if_VS_lof + total,
			ee_VS_lof- total,
			if_VS_lof - total,
			total),
			set_labels = ('Elliptic Envelope', 'IsolationForest',
						  'Local Outlier Factor'))
	
	plt.title('contaminacion = '+ str(c))
	plt.show()
	
	print('\nElliptic Envelope & Isolation Forest superposed outliers: ',
		  ee_VS_if, '\n')
	print('\nElliptic Envelope & LOF superposed outliers: ',
		  ee_VS_lof, '\n')
	print('\nLOF & Isolation Forest superposed outliers: ',
		  if_VS_lof, '\n')
			
	print('\n2 times superposed outliers: ',
			df_ensemble.loc[df_ensemble['voting'] == 2, 'voting'].count(), '\n')
	print('\n3 times superposed outliers: ', total, '\n')
	
	df_ensemble['outliers'] = df_ensemble['voting']
	df_ensemble.loc[df_ensemble['outliers'] == 0, 'outliers'] = 1
	df_ensemble.loc[df_ensemble['outliers'] == 1, 'outliers'] = 1
	df_ensemble.loc[df_ensemble['outliers'] == 2, 'outliers'] = -1
	df_ensemble.loc[df_ensemble['outliers'] == 3, 'outliers'] = -1	
	
	df_ensemble[['outliers']].to_csv('ensemble/'+ dataset + '_'+ str(c))

def outlier_metrics(predictions,dataset):
	inliers = len(predictions.loc[predictions['outliers'] == 1])
	outliers = len(predictions.loc[predictions['outliers'] == -1])
	outlier_ratio = outliers / (inliers + outliers)
	
	print('\n##############    ',dataset,'   ##################\n')
	print('# inliers: ', inliers)
	print('# outliers: ', outliers)
	print('# ratio: ', outlier_ratio)
	return outlier_ratio


# ---------------------VISUALIZATIONS-----------------------------------	

def interplots(df,df_ip,method):
	plt.figure()
	plt.scatter(df.index,df, c='b')
	plt.scatter(df.loc[df.isnull()].index,
				df.loc[df.isnull()].fillna(df.mean()), 
				c='r', marker='x', zorder=10,label ='blancs')
	plt.scatter(df_ip.index[df.isnull()], df_ip.loc[df.isnull()],
				c='goldenrod',label='interpolations')
	plt.legend()
	plt.show()


def plotDimensionReduction(df,method):
	components = list(df)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_title('3 component '+ method, fontsize = 10)
	ax.scatter(df[components[0]], df[components[1]], df[components[2]])
	ax.set_xlabel(components[0])
	ax.set_ylabel(components[1])
	ax.set_zlabel(components[2])
	plt.show()


def plotDimensionReductionOutliers3D(df,method):
	components = list(df)
	components.remove('outliers')
	print(df.head())
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_title('3 component '+ method, fontsize = 10)
	
	inl = df.loc[df['outliers'] == 1]
	out = df.loc[df['outliers'] == -1]
	
	ax.scatter(inl[components[0]],
			   inl[components[1]],
			   inl[components[2]],
			   c='b', marker='o')
	ax.scatter(out[components[0]],
			   out[components[1]],
			   out[components[2]],
			   c='r', marker='x')
	ax.set_xlabel(components[0])
	ax.set_ylabel(components[1])
	ax.set_zlabel(components[2])
	plt.show()	


def plotDimensionReductionOutliers2D(df,method, ax):
	components = list(df)
	components.remove('outliers')
	dot_size = 10
	cross_size = 20
	
	inl = df.loc[df['outliers'] == 1]
	out = df.loc[df['outliers'] == -1]
	
	ax.scatter(inl[components[0]],
			   inl[components[1]],
			   c='b', marker='o', s=dot_size)
	ax.scatter(out[components[0]],
			   out[components[1]],
			   c='r', marker='x', s=cross_size)
	ax.set_xlabel(components[0])
	ax.set_ylabel(components[1])


def outlier_plot2D(df_train,df_test,x,y,method_name,ratios):
	fig = plt.figure()
	# plot train_set
	ax = fig.add_subplot(121)
	dot_size = 10
	cross_size = 20
	
	ax.scatter(df_train.loc[df_train['outliers'] == 1, x],
				   df_train.loc[df_train['outliers'] == 1, y],
				   c='b', marker='o', s=dot_size)			   

	ax.scatter(df_train.loc[df_train['outliers'] == -1,x],
				   df_train.loc[df_train['outliers'] == -1,y],
				   c='r', marker='x', s=cross_size)
	
	title1 = ('Train set | ' + method_name + 
			  '| outlierr ratio: ' + str(round(ratios[0],3)) + '%')
			  
	ax.set_title(title1,fontsize=9)			   
	ax.set_xlabel(x)
	ax.set_ylabel(y)
	ax.set_xlim(0,15)
	# plot test_test
	ax = fig.add_subplot(122)
	
	ax.scatter(df_test.loc[df_test['outliers'] == 1,x],
				   df_test.loc[df_test['outliers'] == 1,y],
				   c='b',marker='o', s=dot_size)
	
	ax.scatter(df_test.loc[df_test['outliers'] == -1,x],
				   df_test.loc[df_test['outliers'] == -1,y],
				   c='r', marker='x', s=cross_size)
				   
	title2 = ('Test set | ' + method_name + 
			  '| outlierr ratio: ' + str(round(ratios[1],3)) + '%')	
			  
	ax.set_title(title2,fontsize=9)
	ax.set_xlabel(x)
	ax.set_ylabel(y)
	ax.set_xlim(0,15)
	ax.legend(('inlier','outlier'), loc='lower right')
	

def outlier_plot3D(df_train,df_test,x,y,z,method_name,ratios):
	fig = plt.figure()
	# plot train_set
	inl = df_train.loc[df_train['outliers'] == 1]
	out = df_train.loc[df_train['outliers'] == -1]
	
	ax = fig.add_subplot(121, projection='3d')
	ax.scatter(inl[x], inl[y], inl[z], c='b', marker='o')
	
	title1 = ('data: train set | method: ' + method_name + '| detected \
			  as outlier: ' + str(ratios[0]) + '%')
	
	ax.set_title(title1,fontsize=7)
	ax.scatter(out[x], out[y], out[z], c='r', marker='x')
	ax.set_xlabel(x)
	ax.set_ylabel(y)
	ax.set_zlabel(z)
	
	# plot test_test
	inl = df_test.loc[df_test['outliers'] == 1]
	out = df_test.loc[df_test['outliers'] == -1]
	
	ax = fig.add_subplot(122, projection='3d')
	ax.scatter(inl[x], inl[y], inl[z], c='b', marker='o')
	ax.scatter(out[x], out[y], out[z], c='r', marker='x')			   
	
	title2 = 'data: test set |  method: '+ method_name + ' | detected \
			  as outlier: '+ str(ratios[1]) + '%'
	
	ax.set_title(title2,fontsize=7)
	ax.set_xlabel(x)
	ax.set_ylabel(y)
	ax.set_zlabel(z)
	ax.legend(('inlier','outlier'), loc='lower right')
	plt.show()


