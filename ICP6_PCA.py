import pandas as pd
from sklearn.decomposition import PCA

# You can add the parameter data_home to wherever to where you want to download your data
dataset = pd.read_csv('CC.csv')

x = dataset.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]
x=x.apply(lambda x: x.fillna(x.mean()),axis=0)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(x)

x_scaler = scaler.transform(x)
pca = PCA(2)
x_pca = pca.fit_transform(x_scaler)

df2 = pd.DataFrame(data=x_pca)
#print(df2)
from sklearn.cluster import KMeans
nclusters = 4 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(df2)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(df2)
from sklearn import metrics
score = metrics.silhouette_score(df2, y_cluster_kmeans)
print(score)

