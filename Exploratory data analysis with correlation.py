import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
%matplotlib inline

#set dataset (used in previous phase)
heart_df=pd.read_csv("C:/dataset.csv")
heart_df.head()
 
 #prepare DS and drop missing values
heart_df.isnull().sum()
count=0
for i in heart_df.isnull().sum(axis=1):
    if i>0:
        count=count+1
print('Total number of rows with missing values is ', count)
print('since it is only',round((count/len(heart_df.index))*100), 'percent of the entire dataset the rows with missing values are excluded.')
heart_df.dropna(axis=0,inplace=True)

##Exploratory Data Analysis
#correlation matrix
corr = heart_df.corr()
plt.subplots(figsize=(15,10))
sn.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sn.diverging_palette(220, 20, as_cmap=True))
sn.heatmap(corr, xticklabels=corr.columns,
            yticklabels=corr.columns, 
            annot=True,
            cmap=sn.diverging_palette(220, 20, as_cmap=True))
corr.to_csv('correlation.csv')
#calc pairplots 
def draw_histograms(dataframe, features, rows, cols):
    fig=plt.figure(figsize=(20,20))
    for i, feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins=20,ax=ax,facecolor='midnightblue')
        ax.set_title(feature+" Distribution",color='DarkRed')
        
    fig.tight_layout()  
    plt.show()
draw_histograms(heart_df,heart_df.columns,6,3)

#it takes some time to run...
sn.pairplot(data=heart_df)

#Calc +ve and -ve heart disease patients
sn.catplot(x="target", y="oldpeak", hue="slope", kind="bar", data=heart_df);

plt.title('ST depression (induced by exercise relative to rest) vs. Heart Disease',size=25)
plt.xlabel('Heart Disease',size=20)
plt.ylabel('ST depression',size=20)

##Violin & Box Plots
#B/w Thalach level and HD
plt.figure(figsize=(12,8))
sn.violinplot(x= 'target', y= 'oldpeak',hue="sex", inner='quartile',data= heart_df )
plt.title("Thalach Level vs. Heart Disease",fontsize=20)
plt.xlabel("Heart Disease Target", fontsize=16)
plt.ylabel("Thalach Level", fontsize=16)

#B/w ST depression level and HD
plt.figure(figsize=(12,8))
sn.boxplot(x= 'target', y= 'thalach',hue="sex", data=heart_df )
plt.title("ST depression Level vs. Heart Disease", fontsize=20)
plt.xlabel("Heart Disease Target",fontsize=16)
plt.ylabel("ST depression induced by exercise relative to rest", fontsize=16)

##Filtering data by positive & negative Heart Disease patient
# Filtering data by POSITIVE Heart Disease patient
pos_data = heart_df[heart_df['target']==1]
pos_data.describe()
pos_data.to_csv('pos_patient.csv')

# Filtering data by NEGATIVE Heart Disease patient
neg_data = heart_df[heart_df['target']==0]
neg_data.describe()
neg_data.to_csv('neg_patient.csv')

#+ve and _ve ST depression patients
print("(Positive Patients ST depression): " + str(pos_data['oldpeak'].mean()))
print("(Negative Patients ST depression): " + str(neg_data['oldpeak'].mean()))
pos_data['oldpeak'].to_csv('ST_pos.csv')

#+ve and _ve thalach patients
print("(Positive Patients thalach): " + str(pos_data['thalach'].mean()))
print("(Negative Patients thalach): " + str(neg_data['thalach'].mean()))
pos_data['thalach'].to_csv('tha_pos.csv')
