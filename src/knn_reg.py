import numpy as np
import matplotlib.pyplot as plt

# Gnerate training data
n_dots = 40
X = 5 * np.random.rand(n_dots, 1)
y = np.cos(X).ravel()
print('Training Data:', 'X=', X, ', y=', y)

# Add noise to training data
y += 0.2 * np.random.rand(n_dots) - 0.1
print('Training Data: y(with noise)=', y)

# Start Training
from sklearn.neighbors import KNeighborsRegressor
k = 5
knn = KNeighborsRegressor(k)
knn.fit(X, y)

# prediction
T = np.linspace(0, 5, 500)[:, np.newaxis]
y_pred = knn.predict(T)
print('T=', T, 'y_pred=', y_pred, 'knn.score(X, y)=', knn.score(X, y))

plt.figure(figsize=(16, 10), dpi=144)
plt.scatter(X, y, c='g', label='data', s=100)
plt.plot(T, y_pred, c='k', label='prediction', lw=4)
plt.axis('tight')
plt.title("KNeighborsRegressor (k = %i)" % k)
plt.show()