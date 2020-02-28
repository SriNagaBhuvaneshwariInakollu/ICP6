import pandas as pd
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# You can add the parameter data_home to wherever to where you want to download your data
x = pd.read_csv('CC.csv', index_col=0)
x = x.apply(lambda x: x.fillna(x.mean()), axis=0)
scaler.fit(x)
x_scaler= scaler.transform(x)
x_scaled=pd.DataFrame(x_scaler, columns =x.columns)
print(x.isnull().sum())
pca = PCA(2)
x_pca = pca.fit_transform(x_scaler)
df2 = pd.DataFrame(data=x_pca)
from sklearn.cluster import KMeans
nclusters = 4 # this is the k in kmeans
seed=0
km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(df2) # predict the cluster for each data pointy_cluster_kmeans=km.predict(X_scaler)
# predict the cluster for each data point
y_cluster_kmeans = km.predict(df2)
from sklearn import metrics
score = metrics.silhouette_score(df2, y_cluster_kmeans)
print(score)