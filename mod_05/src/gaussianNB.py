
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
from module5_source import (classi_ind_regr, plot_2d, frontiere)


# Generation des donnees
n_samples = 100
X = np.zeros([3 * n_samples, 2])
mean_0 = [1, 1]
mean_1 = [1.5, 1.5]
mean_2 = [2, 2]
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


# Gaussain Naiv Bais

clf = GaussianNB()
clf.fit(X, y)

display_1 = [2, 2]
display_2 = [3, 1]
display_3 = [2.5, 2.5]

values_proba_gnb_1 = np.exp(clf.predict_log_proba(display_1))[0]
values_proba_gnb_2 = np.exp(clf.predict_log_proba(display_2))[0]
values_proba_gnb_3 = np.exp(clf.predict_log_proba(display_3))[0]

ig1_bis = plt.figure()
plot_2d(X, y)

resolution_param = 50  # 500 for nice plotting, 50 for fast version
color_text = '#ff8101'

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
             color =color_text, xytext=(-150, -200),
             textcoords='offset points', fontsize=12,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2",
                             color=color_text))

plt.plot(display_2[0], display_2[1], 'o', color=color_text, markersize=12)
plt.annotate(r'' + '(%.2f' % values_proba_gnb_3[0] + ', %.2f'
             % values_proba_gnb_3[1] + ', %.2f)' % values_proba_gnb_3[2],
             xy=(display_3[0], display_3[1]), xycoords='data',
             color =color_text, xytext=(100, -200), textcoords='offset points',
             fontsize=12,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2",
                             color=color_text))
plt.plot(display_3[0], display_3[1], 'o', color=color_text, markersize=12)

plt.show()

# Matrice de confusion
y_pred=clf.predict(X)

mc = np.zeros((3,3))

for i in range(3):
    for j in range(3):
        mc[i,j] = len(y_pred[y_pred[y==i]==j])

print mc

clf.score(X, y)
