# -*- coding: utf-8 -*-
"""
@author: Kristiina Pokk
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster.bicluster import SpectralCoclustering
#from sklearn.cluster.bicluster import SpectralBiclustering
import pandas as pd 


# Reading data

data = pd.read_csv('meangamma_ventral_w250_10hz.csv')
Mat = data.as_matrix()
image_numbers = np.array(pd.read_csv('stimsequence.txt', sep=" ", header=None)).flatten()
areas = data.columns

# Removing outliers
up = np.amax(Mat, axis=0)
down = np.amin(Mat, axis=0)
mapping = np.logical_or(np.log10(up)>1,np.log10(down)<-2)
mapping = np.logical_not(mapping)
areas=areas[mapping]
MatOut = Mat[:,mapping]


##   Histogram
#fig = plt.figure('Histogram')
#mH = []
#for k in MatOut:
#    for l in k:
#        mH.append(l)
#plt.hist(mH,100)


## Shuffling data
#scrambled = MatOut.transpose()
#for i in range(0,len(scrambled)):
#    scrambled[i] = np.random.permutation(scrambled[i])    
#MatOut = scrambled.transpose()



# Renaming image id-s
indx = ['C' + str(nr) if str(image_numbers[nr]).startswith('SCR') else image_numbers[nr][0] + str(nr) 
            for  nr in range(0,len(image_numbers))]


# Binary values
MatOut[MatOut < 0.6] = 0.1
MatOut[MatOut >= 0.6] = 1

# Ternary values

#mM[mM < 0.25] = 0.1
#mM[np.where(np.logical_and(mM >= 0.25, mM < 1.25))] = 0.5
#mM[mM >= 1.25] = 1


#####

matDF = pd.DataFrame(MatOut).set_index(np.array(indx))
matDF.columns = areas


# Original plot
plt.matshow(MatOut, cmap=plt.cm.Blues)
plt.title("Original dataset")
clusters = 8 #6 

model = SpectralCoclustering(n_clusters=clusters)
#model = SpectralBiclustering(n_clusters=clusters)
model.fit(matDF)
    
    

fitData_c = matDF.columns[np.argsort(model.column_labels_)]
matDF = matDF[fitData_c]
fitData_i = matDF.index[ np.argsort(model.row_labels_)]
matDF = matDF.reindex(fitData_i)

column_names =  np.array([i[13:16] for i in fitData_c])

# plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Clusters = %d' % clusters)
cax = ax.matshow(matDF, cmap=plt.cm.Blues)
ax.set_xticks(np.arange(len(column_names)))
ax.set_xticklabels(column_names,rotation='vertical')
ax.set_yticks(np.arange(len(fitData_i)))
ax.set_yticklabels(fitData_i)
ax.tick_params(labelsize = 'small')
  
plt.show()

    