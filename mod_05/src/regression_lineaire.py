from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(seed=42)

# for saving files
saving_activated = False

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


from module5_source import (classi_ind_regr, plot_2d, frontiere)

fig1 = plt.figure()
plot_2d(X, y)
plt.show()


from sklearn.lda import LDA
clf = LDA()
clf.fit(X, y)

display_1 = [2, 2]
display_2 = [3, 1]
display_3 = [2.5, 2.5]

values_proba_lda_1 = np.exp(clf.predict_log_proba(display_1))[0]
values_proba_lda_2 = np.exp(clf.predict_log_proba(display_2))[0]
values_proba_lda_3 = np.exp(clf.predict_log_proba(display_3))[0]

resolution_param = 50  # 500 for nice plotting, 50 for fast version
color_text = '#ff8101'

fig2 = plt.figure()
plot_2d(X, y)
frontiere(lambda xx: clf.predict(xx), X, step=resolution_param)

titre = r'' + '(%.2f' % values_proba_lda_1[0] + ', %.2f' % values_proba_lda_1[1] + ', %.2f)' % values_proba_lda_1[2]
plt.annotate(titre,
             xy=(display_1[0], display_1[1]), xycoords='data',
             color =color_text, xytext=(-150, 200), textcoords='offset points',
             fontsize=12, arrowprops=dict(arrowstyle="->",
             connectionstyle="arc3,rad=.2", color=color_text))


plt.plot(display_1[0], display_1[1], 'o', color=color_text, markersize=12)


plt.annotate(r'' + '(%.2f' % values_proba_lda_2[0] + ', %.2f'
             % values_proba_lda_2[1] + ', %.2f)' % values_proba_lda_2[2],
             xy=(display_2[0], display_2[1]), xycoords='data',
             color =color_text, xytext=(-150, 200), textcoords='offset points',
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
             color =color_text, xytext=(10, -200), textcoords='offset points',
             fontsize=12, arrowprops=dict(arrowstyle="->",
             connectionstyle="arc3,rad=.2", color=color_text))

plt.plot(display_3[0], display_3[1], 'o', color=color_text, markersize=12)

plt.show()





