import numpy as np
import time
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.qda import QDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import svm
import matplotlib.pyplot as plt
from module5_source import frontiere

# ***********************************************************
#                Definition des fonctions
# ***********************************************************


# Visualisation des donnees pour voir la tete qu'elles ont
def show_data(XX, yy, legend):
    """
	Cette fonction affiche les deux premieres colonnes des XX et leurs labels (en bleu, jaune et rouge)
    """
    for i, c in zip(xrange(3), "byr"):
        idx = np.where(yy == i)
        plt.scatter(XX[idx, 0], XX[idx, 1], c=c, label=iris.target_names[i], cmap=plt.cm.Paired)
    plt.title("Les deux premieres colonnes des donnees Iris : " + legend)
    plt.xlabel('Col 1')
    plt.ylabel('Col 2')
    plt.legend(('y=0', 'y=1', 'y=2'), loc='upper left')
    plt.show()

# Calcul de la matrice de confusion
def calcul_matrice_confusion(clf, X_t, y_t, X_v, y_v):
    """
    Fonction pour calculer la matrice de confusion
    """
    labels = np.unique(y_t)
    mc = np.zeros((labels.size,labels.size))
    y_pred = clf.fit(X_t,y_t).predict(X_v)
    for i in labels:
        for j in labels:
            mc[i,j] = len(y_pred[y_pred[y_v==i]==j])
    return mc


def fill_stats(stats, clf, line):
    """
    Cette fonction remplie la ligne "line" de la matrice "stats" comme suit :
    - temps de d'apprentissage pour le classifieur clf
    - temps de validation pour le classifieur clf
    - le score du classifieur clf
    """
    t_0 = time.time()
    clf.fit(X_training, y_training)
    t_1 = time.time()
    score = clf.score(X_validation, y_validation)
    t_2 = time.time()
    stats[line, trainig_time_col] = t_1 - t_0
    stats[line, validation_time_col] = t_2 - t_1
    stats[line, error_rate_col] = 1 - score
    
def fill_stats_with_cross_val(stats, clf, line):
    """
    Cette fonction remplie la ligne "line" de la matrice "stats" comme suit :
    - temps de d'apprentissage pour le classifieur clf
    - temps de cross-validation pour le classifieur clf
    - le taux d'erreur du classifieur clf
    """
    t_0 = time.time()
    clf.fit(X_training, y_training)
    t_1 = time.time()
    score = cross_val_score(clf, X_training, y_training, cv=5)
    t_2 = time.time()
    stats[line, trainig_time_col] = t_1 - t_0
    stats[line, validation_time_col] = t_2 - t_1
    stats[line, error_rate_col] = 1 - np.mean(score)
   
def trace_it(clf, X, y, title) :
    """
    Cette fonction trace les frontieres de decision pour le classieur clf et les donnees labelisees superposees
    """
    frontiere(lambda xx: clf.predict(xx), X)
    for i, c in zip(xrange(3), "byr"):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=c, label=iris.target_names[i], cmap=plt.cm.Paired)
    plt.title(title)
    plt.show()


def calcul_pd(x, y, mean, cov):
    """
    Cette fonction calcule la distribution de probabilite de la loi normale a 2 variable pour x et y 
    avec le vecteur de moyennes "mean" et la matrice de covariance "cov"
    """
    x_mean = np.array([x-mean[0], y-mean[1]])
    cov_inv = inv(cov)
    x_mean_T = x_mean.T
    r = math.exp(-0.5 * np.dot(np.dot(x_mean_T, cov_inv), x_mean))
    return r/math.sqrt(((2*math.pi)**2)*np.linalg.det(cov))
 

def trace_courbe_save_pdf(mean, cov, fileName):
    """"
    Cette fonction trace la courbe d'une loi normale a 2 variables ayant comme moyennes le vecteur "mean" et
    comme covariance la matrice "cov", et enregiste cette courbe sous format PDF.
    """
    if (cov == cov.T).all():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = y = np.arange(-3.0, 3.0, 0.02)
        Xs, Ys = np.meshgrid(x, y)
        zs = np.array([calcul_pd(xx,yy, mean, cov) for xx,yy in zip(np.ravel(Xs), np.ravel(Ys))])
        Zs = zs.reshape(Xs.shape)
        ax.plot_surface(Xs, Ys, Zs)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('P')
        ax.view_init(elev=30., azim=120)
        plt.show()
        pp = PdfPages(fileName)
        pp.savefig(fig)
        pp.close()
    else:
        print 'Erreur : Matrice de covariance symetriques'

# ***********************************************************
#                    Preparatoin des donnees
# ***********************************************************


# Load data
iris = datasets.load_iris()
X, y = iris.data[:, :2], iris.target

# Shuffle
X, y = shuffle(X, y, random_state=42)

# Normalize
mean, std = X.mean(axis=0), X.std(axis=0)
X = (X - mean) / std

# On utilise 80% des donnees pour l'apprentissage
training_index = int(y.size * 0.8)

X_training = X[0:training_index,:]
y_training = y[0:training_index]

# Et les 20% restants pour la validation
X_validation = X[training_index:,:]
y_validation = y[training_index:]

# ***********************************************************
#                    Visualisation des donnees
# ***********************************************************

show_data(X, y, "Tout")
show_data(X_training, y_training, "Apprentissage")
show_data(X_validation, y_validation, "Validation")

# ******************************************************
#                        Question 1
# ******************************************************

trainig_time_col = 0
validation_time_col = 1
error_rate_col = 2

gnb_line = 0
lda_line = 1
lr_line = 2
qda_line = 3
knn_line = 4
knn_cv_line = 5
svm_line = 6

names = ('GaussianNB', 'LDA', 'Logistic Regrassion', 'QDA', 'KNN', 'KNN CV', 'SVM')

stats = np.zeros((7, 3))

fill_stats(stats, GaussianNB(), gnb_line)
fill_stats(stats, LDA(), lda_line)
fill_stats(stats, LogisticRegression(), lr_line)
fill_stats(stats, QDA(), qda_line)
fill_stats(stats, KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree'), knn_line)
fill_stats_with_cross_val(stats, KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree'), knn_cv_line)
fill_stats(stats, svm.SVC(), svm_line)

for i in range(0, stats.shape[0]):
    print "%s : fit time=%10.5f, validation time=%10.5f, error=%10.3f" % (names[i], stats[i,0], stats[i,1], stats[i,2])


# ******************************************************
#                        Question 2
# ******************************************************

gnb = GaussianNB()
lda = LDA()
lr = LogisticRegression()
qda = QDA()
knn = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree')
svmm = svm.SVC()

# Calcul des differents matrices de confusion
mc_gaussainNB = calcul_matrice_confusion(gnb, X_training, y_training, X_validation, y_validation)
mc_lda = calcul_matrice_confusion(lda, X_training, y_training, X_validation, y_validation)
mc_lr = calcul_matrice_confusion(lr, X_training, y_training, X_validation, y_validation)
mc_qda = calcul_matrice_confusion(qda, X_training, y_training, X_validation, y_validation)
mc_knn = calcul_matrice_confusion(knn, X_training, y_training, X_validation, y_validation)
mc_svm = calcul_matrice_confusion(svmm, X_training, y_training, X_validation, y_validation)

# ******************************************************
#                        Question 3
# ******************************************************

# Interpretation

trace_it(gnb, X, y, 'GaussianNB')
trace_it(lda, X, y, 'LDA')
trace_it(lr, X, y, 'Logistic Regression')
trace_it(qda, X, y, 'QDA')
trace_it(knn, X, y, 'KNN')
trace_it(svmm, X, y, 'SVM')


# ******************************************************
#                        Question 4
# ******************************************************

from numpy.linalg import inv
import math as math
from matplotlib.backends.backend_pdf import PdfPages
    
mean = np.array([0, 0])
cov = np.array(([0.2, 0.1], [0.1, 0.2]))
trace_courbe_save_pdf(mean, cov, 'gausse_1.pdf')
cov = np.array(([.2, -.2], [-.2, .6]))
trace_courbe_save_pdf(mean, cov, 'gausse_2.pdf')
