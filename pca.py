#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 20:01:17 2019

@author: deepthisen
"""

import numpy as np
from matplotlib import pyplot as plt

X_pca = np.reshape(training[0],[len(training[0]),784])
u,d,vh = np.linalg.svd(X_pca,full_matrices=False)
v = vh.T
#Projection scores along PC's. Here v represents the PCs

scores =  np.dot(X_pca,v)

#Reconstructing the training set using 10 PC's
dim = 400
scores_trunc = scores[:,0:dim]
X_recon = np.dot(scores_trunc,v[:,0:dim].T)

#plt.plot(tdata, tlabels)



f, axes = plt.subplots(1, 2,figsize=[10, 5])
plt.subplots_adjust(bottom = 0)

map0 = axes[0].imshow(np.reshape(X_pca[3,:],[28,28]))
axes[0].set_title('Actual')
f.colorbar(map0,ax=axes[0],fraction=0.046, pad=0.04)
map1= axes[1].imshow(np.reshape(X_recon[3,:],[28,28]))
axes[1].set_title('Reconstructed')
f.colorbar(map1,ax=axes[1],fraction=0.046, pad=0.04)
plt.show()


#plt.scatter(X[:,4],y_mean[:,0])
#plt.scatter(X[:,4],y_mean_recon[:,0])
#plt.show()