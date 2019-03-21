##### scikit - learn . tutorials. #1
# source  https://scikit-learn.org/stable/auto_examples/plot_isotonic_regression.html#sphx-glr-auto-examples-plot-isotonic-regression-py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state



n = 100 
x = np.arange(n)
rs = check_random_state(50)   #randomState object based on np.random

y = rs.randint(-50, 50, size=(n,)) + 50. * np.log1p(np.arange(n))  #it crates random numbers as numpy ndarray

#print('This random array \n',y)  

#######################################################
# Fitting proccess 
# Fit isotonicRegression and Linear Regression models

ir = IsotonicRegression()

y_ = ir.fit_transform(x, y)

lr = LinearRegression()
lr.fit(x[:,np.newaxis], y) #x needs to be 2d for Linear Regression;  [1,2,3]  array became after newaxis [[1],[2],[3]]


#Open these two to see the difference 
#print(x)
#print(x[:,np.newaxis])


#######################################################
# Plotting

segments = [[[i, y[i]], [i, y_[i]]] for i in range(n)] 
lc = LineCollection(segments, zorder=0)
lc.set_array(np.ones(len(y)))
lc.set_linewidths(np.full(n, 0.5))

fig = plt.figure()
plt.plot(x, y, 'r.', markersize=12)
plt.plot(x, y_, 'g.-', markersize=12)
plt.plot(x, lr.predict(x[:,np.newaxis]), 'b-')

print(lr.predict(x[:,np.newaxis]))
plt.gca().add_collection(lc)
plt.legend(('Data', 'Isotonic Fit', 'Linear Fit'), loc='lower right')
plt.title('Isotonic Regression')
plt.show()
