import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

# Generate sample data
centers = [
    [-2, 2],
    [2, 2],
    [0, 4]
]

X, y = make_blobs(n_samples=60, centers=centers, random_state=0, cluster_std=0.60)

c = np.array(centers)
print('Generated Data:', 'X=', X, ', y=', y)

# Training, k=5
from sklearn.neighbors import KNeighborsClassifier
k = 5
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X, y)

# predict new data
X_sample = np.array([[0, 2]])
y_sample = clf.predict(X_sample)
neighbors = clf.kneighbors(X_sample, return_distance=False)
print('X_sample=', X_sample, 'y_sample=', y_sample, ', neighbors=', neighbors)

plt.figure(figsize=(16, 10), dpi=144)
# sample points
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='cool')
# center points
plt.scatter(c[:, 0], c[:, 1], s=100, marker='^', c='k')
# points need to be predicted
plt.scatter(X_sample[:, 0], X_sample[:, 1], marker='x', c=y_sample, s=100, cmap='cool')

for i in neighbors[0]:
    plt.plot([X[i][0], X_sample[0][0]], [X[i][1], X_sample[0][1]], 'k--', linewidth=0.6)

plt.show()