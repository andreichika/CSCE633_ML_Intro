#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 20:31:48 2019

@author: deepthisen
"""

from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.preprocessing import MinMaxScaler as mms
from scipy.optimize import minimize as mnmz
import pandas as pd


# In logistic regression, P(Ck|x) is modeled as a logistic function
def logit_func(a):
    return 1/(1+np.exp(-a))

def cross_entropy(w,X,targ,n_obs,dim,n_cats):
    w = w.reshape([dim,n_cats])
    y = logit_func(np.dot(w.T,X.T)).T
    ce= -np.sum(targ*np.log(y))
    return ce

def calc_grad(w,X,targ,n_obs,dim,n_cats):
    w = w.reshape([dim,n_cats])
    y = logit_func(np.dot(w.T,X.T)).T
    k = targ*(1-y)
    grad= -np.ravel(np.dot(k.T,X).T)
    #grad =  -np.sum(np.sum(targ*(1-y),axis=1).reshape([n_obs,1])*X,axis=0)
    return grad

def class_efficiency(t_act,t_pred):
   cols = ['t_act','t_pred']
   df = pd.DataFrame(np.concatenate([training[1],pred_cat.reshape([n_obs,1])],axis=1),columns=cols)
   ct = pd.crosstab(df.t_act, df.t_pred)
   return  ct


ohe1 = ohe(handle_unknown='ignore')
ohe1 = ohe1.fit(training[1])
targ = ohe1.transform(training[1]).toarray()

X = scores_trunc.copy()
dim = len(X.T)
mms1 = mms()
X = mms1.fit_transform(X)
train = np.concatenate([X,targ],axis=1)

n_cats = len(targ.T)
n_obs = len(train)
#dim = n_obs-n_cats

df_train = pd.DataFrame(train)
old_col = np.arange(dim,n_obs).tolist()
new_col=[]
for i in range(0,n_cats):
    new_col.append("target"+str(i))
df_train.rename(columns={i:j for i,j in zip(old_col,new_col)},inplace=True)

#Compute mean vector per class
w0=np.ravel((np.ones([dim,n_cats]))/dim)

res = mnmz(cross_entropy, w0, args=(X,targ,n_obs,dim,n_cats),method='BFGS', jac=calc_grad,options={'gtol': 1e-1, 'disp': True})

# Testing classifier

w_opt =  res.x.reshape([dim,n_cats])
y_pred = logit_func(np.dot(X,w_opt))
tot_prob = np.sum(logit_func(np.dot(X,w_opt)),axis=1).reshape([n_obs,1])
pred_cat = np.argmax(y_pred/tot_prob,axis=1)


print(class_efficiency(training[1],pred_cat.reshape([n_obs,1])))
   
#compare with scikitlearn implementation

from sklearn.linear_model import LogisticRegression as lr

lr = lr() 
lr_fit = lr.fit(X,training[1])
pred = lr_fit.predict(X)

print(class_efficiency(training[1],pred.reshape([n_obs,1])))