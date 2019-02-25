#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 20:38:16 2019

@author: deepthisen
"""
import pandas as pd
import heapq

#train = np.concatenate([np.reshape(training[0],[len(training[0]),784]),training[1]],axis=1)
train = np.concatenate([scores_trunc,training[1]],axis=1)

dim = len(train.T)-1

df_train = pd.DataFrame(train)
df_train.rename(columns={dim:"target"},inplace=True)

#Compute mean vector per class
mean_class=df_train.groupby(['target']).mean()


#Compute scatter matrix (covariance matrix) per class and do evd of each
i=0
w = np.zeros([dim,10])
eigen_store = np.zeros([dim,dim,10])
C = np.zeros([dim,dim,10])
for group in df_train.groupby(['target']):
    C[:,:,i] = np.cov(group[1].values[:,:dim].T)
    w[:,i], eigen_store[:,:,i] = np.linalg.eig(C[:,:,i])
    i+=1


##Choose the highest k eigenvalues. k =  20 here
#    
#def k_largest_index_argsort(a, k):
#    idx = np.argsort(a.ravel())[:-k-1:-1]
#    return np.column_stack(np.unravel_index(idx, a.shape))
#    
#k = 20
#max_ind = k_largest_index_argpartition_v1(w,k)
#w_max = np.sort(w[max_ind[:,0],max_ind[:,1]])[::-1]
    
Sw = np.sum(C,axis=2)
Sb = np.cov(mean_class.values.T)
Sx = np.dot(np.linalg.inv(Sw),Sb)

u, v = np.linalg.eig(Sx)

LD_scores = np.dot(scores_trunc,v)

df_train['LD_score1'] = LD_scores[:,0]
df_train['LD_score2'] = LD_scores[:,1]
df_train.groupby(['target']).mean()

for group in df_train.groupby(['target']):
    l1 = group[1]['LD_score1'].values
    l2 = group[1]['LD_score2'].values
    plt.scatter(l1,l2,label=group[0])
plt.legend()
plt.title('Using LDA')
plt.show()


for group in df_train.groupby(['target']):
    pc1 = group[1][0].values
    pc2 = group[1][1].values
    plt.scatter(pc1,pc2,label=group[0])
plt.legend()
plt.title('Using PCA')
plt.show()
