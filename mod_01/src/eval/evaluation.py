from tp_eval_functions import (generate_data, plot_data, calcule_y)
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


# Q1
plt.subplot(1,2,1)
plt.title('Premier jeu de données')
plot_data(generate_data())
plt.subplot(1,2,2)
plt.title('Deuxième jeu de données')
plt.legend('toto')
plot_data(generate_data())


# Q2
data = generate_data();
data_plus = data[data[:,2]==1]
data_moins = data[data[:,2]==-1]
p_plus_estime = data_plus.shape[0]/data.shape[0]
means_plus = data_plus.mean(0)
means_moins = data_moins.mean(0)
means = data.mean(0)
print 'Probabilite estimee des Y=+1 : ', p_plus_estime
print 'Mu plus estime : ', means_plus[:2]
print 'Mu moins estime : ', means_moins[:2]
print 'Mu globale : ', means[:2]
p_plus_estime*means_plus[:2] + (1-p_plus_estime)*means_moins[:2]

# Q3
var_plus = data_plus.var(0)
var_moins = data_moins.var(0)
print 'Varience plus estimee (biaisée) : ', var_plus[:2]
print 'Varience plus estimee (non biaisée) : ', (data_plus.shape[0]/(data_plus.shape[0]-1)) * var_plus[:2]
print 'Varience moins estime (biaisée): ', var_moins[:2]
print 'Varience moins estime (non biaisée): ', (data_moins.shape[0]/(data_moins.shape[0]-1)) * var_moins[:2]


# Q4

regr = linear_model.LinearRegression()
regr.fit(data[:,:2], data[:,2])

T1 = regr.coef_
T0 = regr.intercept_
print 'Theta 1 = ', T1
print 'Theta 0 = ', T0

T0 = T0.reshape(1,1)
T1 = T1.reshape(2,1)
Theta = np.vstack([T0, T1])
Z = np.hstack([np.ones((data.shape[0],1)), data[:,:2]])
Y_estime = np.sign(np.dot(Z,Theta))
data_with_y_estime = np.hstack([data[:,:2], Y_estime])
plot_data(data_with_y_estime)
ax_1 = np.arange(-10,10,0.1).reshape(200,1)
ax_2 = calcule_y(ax_1, T1, T0).reshape(200,1)
plt.plot(ax_1, ax_2, 'k-')
