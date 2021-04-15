import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.cluster import KMeans
data=pd.read_csv("E:/ip.csv")
data.head()

# statistics of the data
data.describe()

# standardizing the data
from sklearn.preprocessing import StandardScaler
array=df.values
x=array[:,0:8]
y=array[:,8]
scaler=MinMaxScaler(feature_range=(0,1))
numpy.set_printoptions(precision=3)
x[0:5,:]
from sklearn.preprocessing import scale
data_standardized=scale(df)
data_standardized.mean(axis=0)
>>> correlations=df.corr()
>>> fig=plt.figure()
>>> ax=fig.add_subplot(111)
>>> cax=ax.matshow(correlations,vmin=-1,vmax=1)
>>> fig.colorbar(cax)
>>> ticks=numpy.arange(0,9,1)
>>> ax.set_xticks(ticks)
>>> ax.set_yticks(ticks)
>>> plt.show()
pandas.plotting.scatter_matrix(df)


names = ROW_ID,names=['SUBJECT_ID','HADM_ID','ICUSTAY_ID','ITEMID','CHARTTIME','STORETIME','CGID','VALUE','VALUENUM','UOM','WARNING','ERROR']
d = pandas.read_csv("E:/ip.csv", names=names)
# Univariate Density Plots
