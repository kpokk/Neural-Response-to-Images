# -*- coding: utf-8 -*-
"""
@author: Kristiina Pokk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


## reading files
data = pd.read_csv('meangamma_ventral_w250_10hz.csv')
M = data.as_matrix()
eti = data.columns


# removing outliers
up = np.amax(M, axis=0)
down = np.amin(M, axis=0)
mapping = np.logical_or(np.log10(up)>1,np.log10(down)<-2)
mapping = np.logical_not(mapping)
mM = M[:,mapping]
Mm = mM.transpose()
labels = eti[mapping]
labels = [l[:16] for l in labels]



# t-SNE

# Mm_norm = normalize(Mm)
dist='cosine' #euclidean cosine
comp=10
per = 15
pca = PCA(n_components=comp) 
mM_pca = pca.fit_transform(Mm) #Mm_norm
dist_mM = np.absolute(pairwise_distances(mM_pca,metric=dist))

#sklearn tsne
mappedM = TSNE(n_components=2,perplexity=per,learning_rate=500, n_iter_without_progress=150, metric='precomputed').fit_transform(dist_mM)

#grouping data for plotting
mappedData = pd.DataFrame(dict(x = mappedM[:,0], y = mappedM[:,1], label = labels))
groups = mappedData.groupby('label')
    

#plot
fig, ax = plt.subplots()
fig.set_size_inches(12, 8.5, forward=True)
fig.subplots_adjust(bottom=0.22, right=0.87)
colormap = plt.cm.jet
plt.gca().set_prop_cycle(cycler('color',[colormap(i) for i in np.linspace(0,1, 9)]))
ax.set_title('i=%d p=%d m=%s' %(comp, per, dist) )
ax.margins(0.2)
for name, group in groups:
    ax.plot(group.x, group.y,marker='o', linestyle='', label=name)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=3)
plt.show()