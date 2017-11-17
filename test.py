import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

cluster1=np.random.uniform(0.5,1.5,(2,10))
cluster2=np.random.uniform(5.5,6.5,(2,10))
cluster3=np.random.uniform(3.0,4.0,(2,10))
print(cluster3)
X=np.hstack((cluster1,cluster2,cluster3)).T
#print(X)
plt.scatter(X[:,0],X[:,1])
plt.xlabel('x1')
plt.ylabel('x2')
#plt.show()

kmeans=KMeans(n_clusters=3)
kmeans.fit(X)
print(cdist(X,kmeans.cluster_centers_,'euclidean'))
print(np.min(cdist(X,kmeans.cluster_centers_,'euclidean'),axis=1))
#print(sum(np.min(cdist(X,kmeans.cluster_centers_,'euclidean'),axis=1))/X.shape[0])