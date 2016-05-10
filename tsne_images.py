# -*- coding: utf-8 -*-
"""
@author: Kristiina Pokk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE

## reading files
data = pd.read_csv('meangamma_ventral_w250_10hz.csv')
M = data.as_matrix()
group_numbers = np.array(pd.read_csv('stimgroups.txt', sep=" ", header=None)).flatten()


# removing outliers
up = np.amax(M, axis=0)
down = np.amin(M, axis=0)
mapping = np.logical_or(np.log10(up)>1,np.log10(down)<-2)
mapping = np.logical_not(mapping)
mM = M[:,mapping]



#normalize
#mM = normalize(mM)  
per = 15  
dist='euclidean' # cosine euclidean        
dist_mM = np.absolute(pairwise_distances(mM,metric=dist))
    
#sklearn tsne
mappedM = TSNE(n_components=2,perplexity=per,learning_rate=500, n_iter_without_progress=150, metric='precomputed').fit_transform(dist_mM)    
   
#grouping data for plotting
mappedData = pd.DataFrame(dict(x = mappedM[:,0], y = mappedM[:,1], label = group_numbers))
groups = mappedData.groupby('label')    

#plot
fig, ax = plt.subplots()

fig.subplots_adjust(right=0.8)
ax.set_title('images p=%d m=%s' %(per, dist) )
for name, group in groups:
    ax.plot(group.x, group.y,marker='o', linestyle='', label=name)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
          fancybox=True, shadow=True)
plt.show()
