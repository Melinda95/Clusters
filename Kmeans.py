import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

digits_train=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',header=None)
digits_test=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',header=None)

X_train=digits_train[np.arange(64)]
y_train=digits_train[64]

X_test=digits_test[np.arange(64)]
y_test=digits_test[64]

from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=10)
kmeans.fit(X_train)
y_pred=kmeans.predict(X_test)

from sklearn import metrics
print(metrics.adjusted_rand_score(y_test,y_pred))