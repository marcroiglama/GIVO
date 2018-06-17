from __future__ import division, print_function
import numpy as np
import pandas as pd

from sklearn.externals import joblib

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import modelling_functions as mf


def mainEvaluateInterpolation():
	# 1.Get spark data and transform to pandas df
	df = mf.readCSV('curated_sensors.csv')
	print(df.columns)
	variable = raw_input("\nCHOOSE A VARIABLE: ")
	methods = ['time', 'polynomial1','polynomial2','polynomial3',
			   'polynomial5','polynomial7','polynomial9']

	# 2. Create a validation set with some extra NaNs
	df_target = df[[variable]].copy()
	df_validation = mf.createValidationSet(df_target.dropna().copy())

	cond = df_validation['value set to NaN'].index
	print('\nEvaluation of filled NaNs with the mean\n')
	df_mean = df_validation.fillna(df_validation[variable].mean())
	mf.error(df_target.loc[cond], df_mean.loc[cond,variable])
	for method in methods:
		# 3. Interpolate to fill blanks
		if method[0:4] == 'poly':
			d = method[10]
			df_ip_validation = mf.interpolate(df_validation[variable].copy(),
											  method, 4, int(d))
			print('\nEvaluation of polynomial with degree ' + d + '\n')
		else: 
			df_ip_validation = mf.interpolate(df_validation[variable].copy(),
											  method, 4, None)
			print('\nEvaluation of ' + method +' interpolation method\n')									  
		# 4. Compare results of the df_ip_validate and original df
		mf.error(df_target.loc[cond],df_ip_validation.loc[cond,method])
		del [df_ip_validation]


def mainApplyInterpolation():	
	df = mf.readCSV('curated_sensors.csv')
	
	print(df.columns)
	variable = raw_input("\nCHOOSE A VARIABLE: ")
	
	# 5. Select to best method to apply to the data and keep these values
	print('\nlist of methods: mean, time, polynomial1, polynomial2,\
		   polynomial3, polynomial5, polynomial7, polynomial9\n')
	method = raw_input("CHOOSE THE BEST METHOD: ")
	
	df_target = df[[variable]].copy()

	if method == 'mean':
		df_ip = df_target.fillna(df_target.mean())
	elif method[0:4] == 'poly':
		df_ip = mf.interpolate(df_target.copy(), method, 4, method[10])
	else:
		df_ip = mf.interpolate(df_target.copy(), method, 4, None)
	blancs = df_target[variable].isnull()
	df_target.loc[blancs] = df_ip.loc[blancs]

	#6. Show results
	mf.interplots(df[variable].copy(),df_target.copy(),method)
	
	print('\n number of blancs on the original data:\n', 
		  df[variable].isnull().sum())
	print('\n number of blancs after interpolation:\n', 
		  df_ip.isnull().sum())
	print('\n if any blanc persist means that its sorrounded \
			by more than other 3 blancs, they will be deleted \n')	 
	# 7. Save modified data (UNCOMMENT!)
	'''
	df_full = mf.readCSV('interpolated_sensors.csv')
	df_full[variable] = df_target
	df_full.to_csv('interpolated_sensors.csv')
	'''
	#firts time that the script runs:
	#df_target.to_csv('interpolated_sensors.csv')
	
	print("interpolation done!")


def mainPreModel():
	df_ip = mf.readCSV('interpolated_sensors.csv')
	# df_ip = mf.dropNotInterpolatedBlancs(df_ip)
	df_ip.to_csv('interpolated_sensors.csv')
	
	df_train, df_test = mf.splitData(df_ip, 0.3)
		
	mf.fitStandardScaler(df_train)
	df_train = mf.standarizing(df_train)
	df_test = mf.standarizing(df_test)
	
	df_train.to_csv('train_test_sets/train_set.csv')
	df_test.to_csv('train_test_sets/test_set.csv')
	
	print('\nfind train and test sets as csv files\n')


def mainOutlierDetection(folder, model_file, train_pred_file, test_pred_file, c):
	# 1.Load train and test sets
	df_train = mf.readCSV('train_test_sets/train_set.csv')
	df_test = mf.readCSV('train_test_sets/test_set.csv')
	print(len(df_train))

	# 2.Create sklearn model and save it
	clf = mf.instanceModel(df_train, folder, model_file, c)	
	
	# 3.Load sklearn model
	clf = joblib.load(folder + '/' + model_file)		
	
	# 4.Predict and save predictions
	mf.predict(df_train, folder, train_pred_file, clf)
	mf.predict(df_test, folder, test_pred_file, clf)


def mainOutlierMetrics(folder, train_pred_file, test_pred_file,):
	# Load predictions
	train_predicted = mf.readCSV(folder + '/' + train_pred_file)
	test_predicted = mf.readCSV(folder + '/' + test_pred_file)
	
	# Calcule outlier ratios
	ratios = [mf.outlier_metrics(train_predicted, train_pred_file),
			  mf.outlier_metrics(test_predicted, test_pred_file)]
	return ratios

			  
def mainPlotOutliers(folder, model_file, train_pred_file, test_pred_file, v1, v2, v3, ratios):
	# Load predictions
	df_train = mf.readCSV('train_test_sets/train_set.csv')
	df_test = mf.readCSV('train_test_sets/test_set.csv')
	
	df_train = mf.inverseStandarizing(df_train)
	df_test = mf.inverseStandarizing(df_test)

	train_predicted = mf.readCSV(folder + '/' + train_pred_file)
	test_predicted = mf.readCSV(folder + '/' + test_pred_file)
	
	df_train['outliers'] = train_predicted['outliers']
	df_test['outliers'] = test_predicted['outliers']
		
	# Plot variables
	if v3 != None: 
		mf.outlier_plot3D(df_train, df_test,
						  v1, v2, v3, model_file, ratios)
	else:
		mf.outlier_plot2D(df_train, df_test,
						  v1, v2, model_file, ratios)


def mainDimensionReduction(folder, model, train_pred_file, test_pred_file):
	# Read predictions
	df_train = mf.readCSV('train_test_sets/train_set.csv')
	df_test = mf.readCSV('train_test_sets/test_set.csv')

	train_predicted = mf.readCSV(folder + '/' + train_pred_file)
	test_predicted = mf.readCSV(folder + '/' + test_pred_file)
	
	# Plot PCA
	f, (ax1,ax2) = plt.subplots(1,2)

	df_pca_train, df_pca_test = mf.pca(df_train.copy(), df_test.copy())
	
	df_pca_train['outliers'] = train_predicted['outliers']
	
	columns_except_outliers = test_predicted.columns != 'outliers'

	df_pca_test['outliers'] = test_predicted['outliers']
	
	mf.plotDimensionReductionOutliers2D(df_pca_train, 'PCA', ax1)
	mf.plotDimensionReductionOutliers2D(df_pca_test, 'PCA',ax2)
	
	title1 = ('Train set | ' + model + ' | PCA')
	ax1.set_title(title1, fontsize = 9)
	ax1.set_xlim(-7,7)
	ax1.set_ylim(-10,10)
	
	title2 = ('Test set | ' + model + ' | PCA') 
	ax2.set_title(title2, fontsize = 9)
	ax2.set_xlim(-7,7)
	ax2.set_ylim(-10,10)
	plt.show()
	
	# Plot T-SNE	
	f, (ax1,ax2) = plt.subplots(1,2)
	
	df_tsne_train, df_tsne_test = mf.tsne(df_train.copy(),
										  df_test.copy())
	
	df_tsne_train['outliers'] = train_predicted['outliers']
	mf.plotDimensionReductionOutliers2D(df_tsne_train, 't-SNE', ax1)
	
	df_tsne_test['outliers'] = test_predicted['outliers']
	mf.plotDimensionReductionOutliers2D(df_tsne_test, 't-SNE', ax2)
	
	title1 = ('Train set | ' + model + ' | T-SNE')
	title2 = ('Test set | ' + model + ' | T-SNE')			  
	ax1.set_title(title1, fontsize = 9)
	ax2.set_title(title2, fontsize = 9)
	
	plt.show()




if __name__ == '__main__':
	'''
	# mainEvaluateInterpolation()
	
	# mainApplyInterpolation()
	 
	# mainPreModel()

	'''

	# inizialize	
	
	algorithm = 'EllipticEnvelope'
	# algorithm = 'IsolationForest'
	# algorithm = 'LocalOutlierFactor'
	# algorithm = 'OneClassSVM'
	# algorithm = 'ensemble'
	
	c = 0.04
	folder = algorithm	
	model = folder + '_' + str(c)
	model_file = model + '.sav'
	train_pred_file = 'train_' + model + '.csv'
	test_pred_file = 'test_' + model + '.csv'


	# fit and predict
	
	#mainOutlierDetection(folder, model_file, train_pred_file, test_pred_file, c)
	
	
	# evaluation
	plt.rcParams["figure.figsize"] = [8,4]


	ratios = mainOutlierMetrics(folder, train_pred_file, test_pred_file,)

	mainPlotOutliers(folder, model_file, train_pred_file, test_pred_file,
					 'co', 'humidity', None, ratios)	
	
	mainDimensionReduction(folder, model, train_pred_file, test_pred_file)
	
	'''
	algorithms = ['EllipticEnvelope', 'IsolationForest', 'LocalOutlierFactor']
	mf.ensemble(algorithms, c,'train')
	#mf.ensemble(algorithms, c,'test')
	'''
	
	print('\n##############    WORKS FINE    ##################')
	
