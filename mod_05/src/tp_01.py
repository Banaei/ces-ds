from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt



digits = load_digits();
X, y = digits.data, digits.target

plt.imshow(np.reshape(X[0], (8, 8)), cmap=plt.cm.gray, interpolation='nearest')
plt.show()

# nombre de class differenes dans la base de donnees
print 'Nombre des classes de la base de donnees' 
n_classes = len(np.unique(y))
print n_classes

n, p = X.shape
mus = np.empty((n_classes, p), dtype=float)
sigma2s = np.empty((n_classes, p), dtype=float)

for k in range(n_classes):
    Xk = X[y==k]
    mus[k] = np.mean(Xk, axis=0)
    sigma2s[k] = np.var(Xk, axis=0)
    
fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(14,2))
for k in range(n_classes):
    axes[k].imshow(np.reshape(mus[k],(8,8)), interpolation='nearest', cmap=plt.cm.gray)
    
fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(14,2))
for k in range(n_classes):
    axes[k].imshow(np.reshape(sigma2s[k],(8,8)), interpolation='nearest', cmap=plt.cm.gray)
   
# standardisation des donnees : moyenne 0 et ecart_type 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
print "The standard mean=%s" % (np.mean(X_std))
print "The standard std=%s" % (np.std(X_std))


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_std, y)
y_pred = logreg.predict(X_std)

# La moyenne dy vecteur ou il y a 1 la y=y_pred
print "La moyenne dy vecteur ou il y a 1 la y=y_pred est %s" % (np.mean(y==y_pred))


logreg.fit(X_std[::2, :], y[::2])
y_pred = logreg.predict(X_std[1::2, :])

# cross-validation sur 2 folds a la main
print "La moyenne apprentissage sur les numeros paires, test sur les numeros impaires %s" % (np.mean(y[1::2]==y_pred))

from sklearn.cross_validation import cross_val_score

scores = cross_val_score(logreg, X_std, y, cv=3, scoring='accuracy')
print 'Les performances des 3 scores %s' % (scores)



# Verification de l'evolution des performances en augmentant la taille de l'echantillon

X_train = X_std[500:]
y_train = y[500:]
X_valid = X_std[:500]
y_valid = y[:500]

print X_train.shape
print X_valid.shape


n_train_samples = range(100, len(X_train), 100)
scores = np.empty(len(n_train_samples))
for k, n_train in enumerate(n_train_samples):
    logreg.fit(X_train[:n_train], y_train[:n_train])
    scores[k] = logreg.score(X_valid, y_valid)

plt.plot(n_train_samples, scores)


from learning_curve import plot_learning_curve
plot_learning_curve()


