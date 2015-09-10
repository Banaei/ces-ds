# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 16:13:18 2014

@author: jo
"""

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

np.random.seed(seed=42)

# for saving files
saving_activated = False

##############################################################################
# Part I: Indicator and least square fitting for classification
##############################################################################


##############################################################################
#      Load some real data and check some statistics
##############################################################################
iris_dataset = load_iris()
X = pd.DataFrame(iris_dataset.data)
y = (iris_dataset.target)
classes = list(set(y))

# palette = ['#994fa1', '#ff8101', '#fdfc33'] # HEXA
palette = [(153. / 255, 79. / 255, 161. / 255),
           (255. / 255, 129. / 255, 1. / 255),
           (253. / 255, 252. / 255, 51. / 255)]  # HEXA
# palette= [(153./255,79./255,161./255),
#            (255/255, 129/255, 1/255),
#            (253/255, 252/255, 51/255)]  # HEXA

color_map = dict(zip(classes, palette))
colors = [color_map[y[i]] for i in xrange(len(y))]
axeslist = pd.scatter_matrix(X, color=colors, diagonal='kde')

X = iris_dataset.data
y = iris_dataset.target

regr_0 = LinearRegression()
regr_1 = LinearRegression()
regr_2 = LinearRegression()

# Classes are 0, 1 and 2

indexes_0 = np.squeeze(np.argwhere(y == 0))
y_0 = np.zeros(y.shape)
y_0[indexes_0] = 1

indexes_1 = np.squeeze(np.argwhere(y == 1))
y_1 = np.zeros(y.shape)
y_1[indexes_1] = 1

indexes_2 = np.squeeze(np.argwhere(y == 2))
y_2 = np.zeros(y.shape)
y_2[indexes_2] = 1

x_to_class = [4.9, 3.1, 1.5, 0.1]
regr_0 = regr_0.fit(X, y_0)
a_0 = regr_0.predict(x_to_class)

regr_1 = regr_1.fit(X, y_1)
a_1 = regr_1.predict(x_to_class)

regr_2 = regr_1.fit(X, y_2)
a_2 = regr_2.predict(x_to_class)

prob_by_classes = [a_0, a_1, a_2]
print prob_by_classes

# Note that it can weirdly be a negative value, ie you estimate a probability
# by a negative number

print 'The winner is class ' + \
    repr(np.argmax([a_0, a_1, a_2])) + ', for x_to_class=' + repr(x_to_class)

print np.sum(prob_by_classes)


##############################################################################
# Exo : try with the constant variable removed from the fitting
##############################################################################

# TODO

##############################################################################
# Exo : create a function taking a data sets with label between
# 0 and K-1 implement the method above
# input   : x_to_class, X, y, K=3
# output  :label_pred, proba_vector
##############################################################################

# Solution is in module5_source.py if needed

##############################################################################
# Limits on a simulated example
##############################################################################

n_samples = 100
X = np.zeros([3 * n_samples, 2])
mean_0 = [1, 1]
mean_1 = [2, 2]
mean_2 = [3, 3]
noise_level = 0.20
cov = noise_level * noise_level * np.array([[1, 0], [0, 1]])
X[0:n_samples, ] = np.random.multivariate_normal(mean_0, cov, n_samples)
X[n_samples: 2 * n_samples, ] = np.random.multivariate_normal(mean_1,
                                                              cov, n_samples)
X[2 * n_samples:3 * n_samples, ] = np.random.multivariate_normal(mean_2, cov,
                                                                 n_samples)
y = np.zeros(3 * n_samples,)
y[n_samples:2 * n_samples] = 1
y[2 * n_samples:3 * n_samples, ] = 2


##############################################################################
# Ploting step
##############################################################################
import matplotlib.pyplot as plt
from module5_source import (classi_ind_regr, plot_2d, frontiere)

fig1 = plt.figure()
plot_2d(X, y)
plt.show()

display_1 = [2, 2]
display_2 = [3, 1]
display_2bis = [3, 3]
display_2ter = [1.5, 2.5]
display_2quad = [1.5, 2]

values_proba1 = classi_ind_regr(display_1, X, y, k=3)[1]
values_proba2 = classi_ind_regr(display_2, X, y, k=3)[1]
values_proba2bis = classi_ind_regr(display_2bis, X, y, k=3)[1]
values_proba2ter = classi_ind_regr(display_2ter, X, y, k=3)[1]
values_proba2quad = classi_ind_regr(display_2quad, X, y, k=3)[1]

resolution_param = 50  # 500 for nice plotting, 50 for fast version

frontiere(lambda xx: classi_ind_regr(xx, X, y, k=3)[0], X,
          step=resolution_param)

color_text = '#ff8101'
plt.annotate(r'' + '(%.2f' % values_proba1[0] + ', %.2f' % values_proba1[1] +
             ', %.2f)' % values_proba1[2],
             xy=(display_1[0], display_1[1]), xycoords='data',
             color =color_text, xytext=(-15, -99), textcoords='offset points',
             fontsize=12, arrowprops=dict(arrowstyle="->",
             connectionstyle="arc3,rad=.2", color=color_text))

plt.plot(display_1[0], display_1[1], 'o', color=color_text, markersize=12)

plt.annotate(r'' + '(%.2f' % values_proba2[0] + ', %.2f' % values_proba2[1] +
             ', %.2f)' % values_proba2[2], xy=(display_2[0], display_2[1]),
             xycoords='data', color =color_text, xytext=(-150, -40),
             textcoords='offset points', fontsize=12,
             arrowprops=dict(arrowstyle="->",
             connectionstyle="arc3,rad=.2", color=color_text))

plt.plot(display_2[0], display_2[1], 'o', color=color_text, markersize=12)

plt.annotate(r'' + '(%.2f' % values_proba2bis[0] + ', %.2f'
             % values_proba2bis[1] + ', %.2f)' % values_proba2bis[2],
             xy=(display_2bis[0], display_2bis[1]), xycoords='data',
             color =color_text, xytext=(-160, 20), textcoords='offset points',
             fontsize=12, arrowprops=dict(arrowstyle="->",
             connectionstyle="arc3,rad=.2", color=color_text))
plt.plot(display_2bis[0], display_2bis[1], 'o',
         color=color_text, markersize=12)

plt.annotate(r'' + '(%.2f' % values_proba2ter[0] + ', %.2f'
             % values_proba2ter[1] + ', %.2f)' % values_proba2ter[2],
             xy=(display_2ter[0], display_2ter[1]), xycoords='data',
             color=color_text, xytext=(-110, 50), textcoords='offset points',
             fontsize=12, arrowprops=dict(arrowstyle="->",
             connectionstyle="arc3,rad=.2", color=color_text))
plt.plot(display_2ter[0], display_2ter[1], 'o', color=color_text,
         markersize=12)


plt.annotate(r'' + '(%.2f' % values_proba2quad[0] + ', %.2f'
             % values_proba2quad[1] + ', %.2f)' % values_proba2quad[2],
             xy=(display_2quad[0], display_2quad[1]), xycoords='data',
             color =color_text, xytext=(-110, 40), textcoords='offset points',
             fontsize=12, arrowprops=dict(arrowstyle="->",
             connectionstyle="arc3,rad=.2", color=color_text))

plt.plot(display_2quad[0], display_2quad[1], 'o',
         color=color_text, markersize=12)

plt.draw()
plt.show()

if saving_activated:
    fig1.savefig('../srcimages/classif_exple_indicator_ls.svg')
else:
    print "no saving performed for classi_ind_regr"


##############################################################################
# Plotting the underlying distrib.: mixture of isotropic Gaussian
##############################################################################
step = 200
xx = np.linspace(0, 4, step)
yy = xx
Xg, Yg = np.meshgrid(xx, yy)
Z1 = plt.mlab.bivariate_normal(Xg, Yg, sigmax=noise_level, sigmay=noise_level,
                               mux=mean_0[0], muy=mean_0[1], sigmaxy=0.0)
Z2 = plt.mlab.bivariate_normal(Xg, Yg, sigmax=noise_level, sigmay=noise_level,
                               mux=mean_1[0], muy=mean_1[1], sigmaxy=0.0)
Z3 = plt.mlab.bivariate_normal(Xg, Yg, sigmax=noise_level, sigmay=noise_level,
                               mux=mean_2[0], muy=mean_2[1], sigmaxy=0.0)

fig3 = plt.figure(figsize=(9, 6), dpi = 90)
ax = fig3.add_subplot(111, projection='3d')
ax.plot_surface(Xg, Yg, (Z1 + Z2 + Z3) / 3, cmap='Oranges',
                rstride=3, cstride=3, alpha=0.9, linewidth=0.5)
plt.show()


Z2 = plt.mlab.bivariate_normal(Xg, Yg, sigmax=noise_level,
                               sigmay=noise_level, mux=mean_1[0],
                               muy=mean_1[1], sigmaxy=0.0)

fig3_bis = plt.figure(figsize=(9, 6), dpi = 90)
ax = fig3_bis.add_subplot(111, projection='3d')
ax.plot_surface(Xg, Yg, Z2, cmap='Oranges', rstride=3, cstride=3,
                alpha=0.9, linewidth=0.5)
ax.set_zlim(0, 4)
plt.show()


Z3 = plt.mlab.bivariate_normal(Xg, Yg, sigmax=noise_level,
                               sigmay=2 * noise_level, mux=mean_1[0],
                               muy=mean_1[1], sigmaxy=0.0)


fig3_ter = plt.figure(figsize=(9, 6), dpi = 90)
ax = fig3_ter.add_subplot(111, projection='3d')
ax.plot_surface(Xg, Yg, Z3, cmap='Oranges', rstride=3, cstride=3,
                alpha=0.9, linewidth=0.5)
ax.set_zlim(0, 4)
plt.show()


Z4 = plt.mlab.bivariate_normal(Xg, Yg, sigmax=noise_level,
                               sigmay=noise_level * 2,
                               mux=mean_2[0], muy=mean_2[1],
                               sigmaxy=noise_level ** 2 / 2)

fig3_quart = plt.figure(figsize=(9, 6), dpi = 90)
ax = fig3_quart.add_subplot(111, projection='3d')
ax.plot_surface(Xg, Yg, Z4, cmap='Oranges', rstride=3, cstride=3,
                alpha=0.9, linewidth=0.5)
ax.set_zlim(0, 4)
plt.show()


if saving_activated:
    fig3.savefig('../srcimages/mixt_of_iso_gaussian.svg')
    fig3_bis.savefig('../srcimages/iso_gaussian.svg')
    fig3_ter.savefig('../srcimages/aniso_gaussian.svg')
    fig3_quart.savefig('../srcimages/aniso_gaussian_general.svg')

else:
    print "no saving performed for LDA"


##############################################################################
#  Exo : Investigate the influence of noise_level on the previous method
##############################################################################

# TODO

##############################################################################
# Part II' : LDA
##############################################################################


from sklearn.lda import LDA
clf = LDA()
clf.fit(X, y)

display_3 = [2.5, 2.5]

values_proba_lda_1 = np.exp(clf.predict_log_proba(display_1))[0]
values_proba_lda_2 = np.exp(clf.predict_log_proba(display_2))[0]
values_proba_lda_3 = np.exp(clf.predict_log_proba(display_3))[0]

fig2 = plt.figure()
plot_2d(X, y)
frontiere(lambda xx: clf.predict(xx), X, step=resolution_param)

plt.annotate(r'' + '(%.2f' % values_proba_lda_1[0] + ', %.2f'
             % values_proba_lda_1[1] + ', %.2f)' % values_proba_lda_1[2],
             xy=(display_1[0], display_1[1]), xycoords='data',
             color =color_text, xytext=(-150, 100), textcoords='offset points',
             fontsize=12, arrowprops=dict(arrowstyle="->",
             connectionstyle="arc3,rad=.2", color=color_text))

# plt.annotate(r'' + '(%.2f' % values_proba_lda_1[0] + ', %.2f'
#              % values_proba_lda_1[1] + ', %.2f)' % values_proba_lda_1[2],
#              xy=(display_1[0], display_1[1]), xycoords='data',
#              color=color_text, xytext=(-150, +100), fontsize=12,
#              arrowprops=dict(arrowstyle="->", textcoords='offset points',
#              connectionstyle="arc3,rad=.2", color=color_text))

plt.plot(display_1[0], display_1[1], 'o', color=color_text, markersize=12)
# plt.annotate(r'' + '(%.2f' % values_proba_lda_2[0] + ', %.2f'
#              % values_proba_lda_2[1] + ', %.2f)' % values_proba_lda_2[2],
#              xy=(display_2[0], display_2[1]), xycoords='data',
#              color =color_text, xytext=(-150, -40), textcoords='offset points',
#              fontsize=12, arrowprops=dict(arrowstyle="->",
#              connectionstyle="arc3,rad=.2", color=color_text))

plt.annotate(r'' + '(%.2f' % values_proba_lda_2[0] + ', %.2f'
             % values_proba_lda_2[1] + ', %.2f)' % values_proba_lda_2[2],
             xy=(display_2[0], display_2[1]), xycoords='data',
             color =color_text, xytext=(-150, 40), textcoords='offset points',
             fontsize=12, arrowprops=dict(arrowstyle="->",
             connectionstyle="arc3,rad=.2", color=color_text))


plt.plot(display_2[0], display_2[1], 'o', color=color_text, markersize=12)
# plt.annotate(r'' + '(%.2f' % values_proba_lda_3[0] + ', %.2f'
#              % values_proba_lda_3[1] + ', %.2f)' % values_proba_lda_3[2],
#              xy=(display_3[0], display_3[1]), xycoords='data',
#              color =color_text, xytext=(10, -80), textcoords='offset points',
#              fontsize=12, arrowprops=dict(arrowstyle="->",
#              connectionstyle="arc3,rad=-.2", color=color_text,))


plt.annotate(r'' + '(%.2f' % values_proba_lda_3[0] + ', %.2f'
             % values_proba_lda_3[1] + ', %.2f)' % values_proba_lda_3[2],
             xy=(display_3[0], display_3[1]), xycoords='data',
             color =color_text, xytext=(10, -80), textcoords='offset points',
             fontsize=12, arrowprops=dict(arrowstyle="->",
             connectionstyle="arc3,rad=.2", color=color_text))

plt.plot(display_3[0], display_3[1], 'o', color=color_text, markersize=12)

plt.show()


if saving_activated:
    fig2.savefig('../srcimages/classif_exple_lda.svg')
else:
    print "no saving performed for LDA"


##############################################################################
# Exo: color palette challenge: find a better palette for
#      displaying the 3 classes examples.
#      potential inspiration:
#      http://scikit-learn.org/stable/auto_examples/plot_lda_qda.html#example-plot-lda-qda-py
##############################################################################


##############################################################################
# Part III : QDA
##############################################################################

from sklearn.qda import QDA
clf = QDA()
clf.fit(X, y)

display_3 = [2.5, 2.5]

values_proba_qda_1 = np.exp(clf.predict_log_proba(display_1))[0]
values_proba_qda_2 = np.exp(clf.predict_log_proba(display_2))[0]
values_proba_qda_3 = np.exp(clf.predict_log_proba(display_3))[0]

fig3 = plt.figure()
plot_2d(X, y)
frontiere(lambda xx: clf.predict(xx), X, step=resolution_param)
plt.annotate(r'' + '(%.2f' % values_proba_qda_1[0] + ', %.2f'
             % values_proba_qda_1[1] + ', %.2f)' % values_proba_qda_1[2],
             xy=(display_1[0], display_1[1]), xycoords='data',
             color=color_text, xytext=(-150, +100), textcoords='offset points',
             fontsize=12, arrowprops=dict(arrowstyle="->",
             connectionstyle="arc3,rad=.2", color=color_text))

plt.plot(display_1[0], display_1[1], 'o', color=color_text, markersize=12)
plt.annotate(r'' + '(%.2f' % values_proba_qda_2[0] + ', %.2f'
             % values_proba_qda_2[1] + ', %.2f)' % values_proba_qda_2[2],
             xy=(display_2[0], display_2[1]), xycoords='data',
             color =color_text, xytext=(-150, -40), textcoords='offset points',
             fontsize=12, arrowprops=dict(arrowstyle="->",
             connectionstyle="arc3,rad=.2", color=color_text))
plt.plot(display_2[0], display_2[1], 'o', color=color_text, markersize=12)
plt.annotate(r'' + '(%.2f' % values_proba_qda_3[0] + ', %.2f'
             % values_proba_qda_3[1] + ', %.2f)' % values_proba_qda_3[2],
             xy=(display_3[0], display_3[1]), xycoords='data',
             color =color_text, xytext=(10, -80), textcoords='offset points',
             fontsize=12, arrowprops=dict(arrowstyle="->",
             connectionstyle="arc3,rad=.2", color=color_text))
plt.plot(display_3[0], display_3[1], 'o', color=color_text, markersize=12)

plt.show()


if saving_activated:
    fig3.savefig('../srcimages/classif_exple_qda.svg')
else:
    print "no saving performed for QDA"


##############################################################################
# Part II : GaussianNB
##############################################################################


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, y)

display_3 = [2.5, 2.5]

values_proba_gnb_1 = np.exp(clf.predict_log_proba(display_1))[0]
values_proba_gnb_2 = np.exp(clf.predict_log_proba(display_2))[0]
values_proba_gnb_3 = np.exp(clf.predict_log_proba(display_3))[0]

fig1_bis = plt.figure()
plot_2d(X, y)
frontiere(lambda xx: clf.predict(xx), X, step=resolution_param)
plt.annotate(r'' + '(%.2f' % values_proba_gnb_1[0] + ', %.2f'
             % values_proba_gnb_1[1] + ', %.2f)' % values_proba_gnb_1[2],
             xy=(display_1[0], display_1[1]), xycoords='data',
             color =color_text, xytext=(-150, +100),
             textcoords='offset points', fontsize=12,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2",
                             color=color_text))

plt.plot(display_1[0], display_1[1], 'o', color=color_text, markersize=12)
plt.annotate(r'' + '(%.2f' % values_proba_gnb_2[0] + ', %.2f'
             % values_proba_gnb_2[1] + ', %.2f)' % values_proba_gnb_2[2],
             xy=(display_2[0], display_2[1]), xycoords='data',
             color =color_text, xytext=(-150, -40),
             textcoords='offset points', fontsize=12,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2",
                             color=color_text))

plt.plot(display_2[0], display_2[1], 'o', color=color_text, markersize=12)
plt.annotate(r'' + '(%.2f' % values_proba_gnb_3[0] + ', %.2f'
             % values_proba_gnb_3[1] + ', %.2f)' % values_proba_gnb_3[2],
             xy=(display_3[0], display_3[1]), xycoords='data',
             color =color_text, xytext=(10, -80), textcoords='offset points',
             fontsize=12,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2",
                             color=color_text))
plt.plot(display_3[0], display_3[1], 'o', color=color_text, markersize=12)

plt.show()


if saving_activated:
    fig1_bis.savefig('../srcimages/classif_exple_GaussianNB.svg')
else:
    print "no saving performed for GaussianNB"


##############################################################################
# Exo : compute the confusion matrix for this method
##############################################################################

# Hint: sklearn has it

##############################################################################
# Part IV : Logistic regression
##############################################################################

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X, y)

display_3 = [2.5, 2.5]

values_proba_logi_1 = np.exp(clf.predict_log_proba(display_1))[0]
values_proba_logi_2 = np.exp(clf.predict_log_proba(display_2))[0]
values_proba_logi_3 = np.exp(clf.predict_log_proba(display_3))[0]

fig4 = plt.figure()
plot_2d(X, y)
frontiere(lambda xx: clf.predict(xx), X, step=resolution_param)
plt.annotate(r'' + '(%.2f' % values_proba_logi_1[0] + ', %.2f'
             % values_proba_logi_1[1] + ', %.2f)' % values_proba_logi_1[2],
             xy=(display_1[0], display_1[1]), xycoords='data',
             color =color_text, xytext=(-150, +100), textcoords='offset points',
             fontsize=12,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2",
                             color=color_text))
plt.plot(display_1[0], display_1[1], 'o', color=color_text, markersize=12)
plt.annotate(r'' + '(%.2f' % values_proba_logi_2[0] + ', %.2f'
             % values_proba_logi_2[1] + ', %.2f)' % values_proba_logi_2[2],
             xy=(display_2[0], display_2[1]), xycoords='data',
             color =color_text, xytext=(-150, -40), textcoords='offset points',
             fontsize=12,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2",
                             color=color_text))
plt.plot(display_2[0], display_2[1], 'o', color=color_text, markersize=12)
plt.annotate(r'' + '(%.2f' % values_proba_logi_3[0] + ', %.2f'
             % values_proba_logi_3[1] + ', %.2f)' % values_proba_logi_3[2],
             xy=(display_3[0], display_3[1]), xycoords='data',
             color =color_text, xytext=(10, -80), textcoords='offset points',
             fontsize=12,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2",
                             color=color_text))
plt.plot(display_3[0], display_3[1], 'o', color=color_text, markersize=12)

plt.show()


if saving_activated:
    fig4.savefig('../srcimages/classif_exple_LogisticRegression.svg')
else:
    print "no saving performed for classif_exple_LogisticRegression"


##############################################################################
# Exo: Are LogisticRegression and LDA the same?.
##############################################################################

##############################################################################
# Exo: invsetigate the part 2 in :
# http://nbviewer.ipython.org/github/cs109/content/blob/master/labs/lab4/Lab4full.ipynb
# What do you think is the C parameter?
# Determine its influence.
##############################################################################


##############################################################################
#      Part V : KNN
##############################################################################

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(X, y)

display_3 = [2.5, 2.5]

fig5 = plt.figure()
plot_2d(X, y)
frontiere(lambda xx: clf.predict(xx), X, step=resolution_param)
plt.show()

# BEWARE: now it's not that easy to get the probability estimates


if saving_activated:
    fig5.savefig('../srcimages/classif_exple_KNN.svg')
else:
    print "no saving performed for classif_exple_KNN"


##############################################################################
# Exo: consider now the iris_dataset. Propose a way, inspired by
# "Learning Curve"
# http://scikit-learn.org/stable/auto_examples/plot_learning_curve.html
# to compare the performance of the following methods:
#       -KNeighborsClassifier
#       -GaussianNB
#       -LogisticRegression
#       -LDA
#       -QDA
##############################################################################
