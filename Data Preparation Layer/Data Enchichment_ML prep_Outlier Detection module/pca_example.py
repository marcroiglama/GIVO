import modelling_functions as mf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plotNewDimension(df):
	components = list(df)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title('Dimension reduction')
	ax.scatter(df[components[0]], df[components[1]])
	ax.set_xlabel('component 1')
	ax.set_ylabel('component 2')

def plotvariables(df,x,y,z):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(df[x], df[y], df[z], marker='o')
	title1 = ('3d variables')
	ax.set_title(title1,fontsize=7)
	ax.set_xlabel(x)
	ax.set_ylabel(y)
	ax.set_zlabel(z)
	
df_train = mf.readCSV('train_test_sets/train_set.csv')
x = 'temperature'
y = 'o3'
z = 'humidity'
#df_train = df_train[[x,y,z]]
#PCA
pca = PCA(n_components = 2)
pca.fit(df_train)
pca_train = pca.transform(df_train)
n_samples = df_train.shape[0]

df_pca = pd.DataFrame(data = pca_train, 
					  columns = ['pc1', 'pc2'], 
					  index = df_train.index)
#TSNE

tsne = TSNE(n_components=2, verbose=1, n_iter=1000, perplexity=35)
tsne_train = tsne.fit_transform(df_train)
df_tsne = pd.DataFrame(data = tsne_train, 
					  columns = ['pc1', 'pc2'], 
					  index = df_train.index)



#plotvariables(df_train,x,y,z)
plotNewDimension(df_tsne)
plt.show()

