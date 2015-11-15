# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 14:51:43 2015

@author: Alireza
"""

import statsmodels as sm
from sklearn import linear_model
from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


def bootstrap_it(n, clf):
    """
    n : number of bootstrap resamplings
    clf : regressor
    returns : coefficients vector
    """
    bs_coefs = []
    for i in range(n):
        if (i%100==0):
            print i,
        X_r, y_r = resample(X, y)
        clf.fit(X_r, y_r)
        bs_coefs.append(clf.coef_)
    return bs_coefs
    

# ******************************************************************
#                  Getting Data and its preparation
# ******************************************************************
data =sm.datasets.get_rdataset('airquality').data
data.isnull()
clean_data = data.dropna()
y = clean_data['Ozone']
X = clean_data[['Solar.R', 'Wind', 'Temp', 'Month', 'Day']]

# ******************************************************************
#                  Regression on the original sample (classic)
# ******************************************************************

clf = linear_model.LinearRegression(normalize=True)
clf.fit(X, y)
coefs_classic = clf.coef_
classic_coeffs_s = pd.Series(coefs_classic.T, index=X.columns)

# ******************************************************************
#                  Ex. 1 - Q 1
# ******************************************************************

n=10000
bs_coefs = np.array(bootstrap_it(n, clf))
means_bs = np.mean(bs_coefs, axis=0)
medians_bs = np.median(bs_coefs, axis=0)

q1_df = pd.DataFrame([coefs_classic, means_bs, medians_bs], columns=X.columns, index=['Classic', 'Mean', 'Median'])

print q1_df

# ******************************************************************
#                  Ex. 1 - Q 2
# ******************************************************************

percentile_005 = np.percentile(bs_coefs, 0.5, axis=0)
percentile_995 = np.percentile(bs_coefs, 99.5, axis=0)
q2_df = pd.DataFrame(np.array([coefs_classic, percentile_005, medians_bs, percentile_995]), columns=X.columns, index=['Classic', 'CI_005', 'Median', 'CI_995'])
print q2_df

# ******************************************************************
#                  Ex. 1 - Q 3
# ******************************************************************
 
start  = 1
stop = 5001
step = 500
points = []
for i in range(start, stop+step, step):
    bs_coefs = bootstrap_it(i, clf)
    median = np.median(bs_coefs, axis=0)[1]
    percentile_005 = np.percentile(bs_coefs, 0.5, axis=0)[1]
    percentile_995 = np.percentile(bs_coefs, 99.5, axis=0)[1]
    points.append([percentile_005, median, percentile_995])
    
points = np.array(points)
plt.plot(range(start, stop+step, step), points[:,0], linestyle='--', color='k', label='Min')
plt.plot(range(start, stop+step, step), points[:,2], linestyle='--', color='k', label='Max')
plt.plot(range(start, stop+step, step), points[:,1], color='k', label='Median')
plt.ylabel('Coefficient of Wind feature')
plt.xlabel('Number of bootsrap resamplings')
plt.title('Median and 95% CI for the regression coefficient of Wind feature')
# plt.legend()
plt.show()


# ******************************************************************
#                  Ex. 1 - Q 4
# ******************************************************************

import seaborn as sns; sns.set(color_codes=True)

g = sns.lmplot(x="Wind", y="Ozone", hue="Month", data=clean_data)

# ******************************************************************
#                  Ex. 2 - Q 1
# ******************************************************************

def stpforward(y, X, M):
    t=0
    r=y
    S=[]
    clf = linear_model.LinearRegression(normalize=True)
    for i in range(M):
        alpha_max = 0
        j_max = 0
        first_loop = True
        for j in X.columns:
            if (j not in S):
                alpha = np.dot(X[j],r)
                if (first_loop):
                    alpha_max = alpha
                    j_max = j
                    first_loop = False
                else:
                    if (alpha > alpha_max):
                        alpha_max = alpha
                    j_max = j
        S.append(j_max)
        clf.fit(X[S], y)
        t = clf.coef_
        r = y - clf.predict(X[S])
    return t, S

t, S = stpforward(y, X, 3)               
            