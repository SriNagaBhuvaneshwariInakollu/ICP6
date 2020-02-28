import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv('CC.csv')

x = dataset.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]
x=x.apply(lambda x: x.fillna(x.mean()),axis=0)
y = dataset.iloc[:,-1]
print(x.isnull().sum())


scaler = StandardScaler()# Fit on training set only.
scaler.fit(x)
# Apply transform to both the training set and the test set.
x= scaler.transform(x)
X_scaled_array=scaler.transform(x)
X_scaled=pd.DataFrame(X_scaled_array)
x=X_scaled
##building the model
from sklearn.cluster import KMeans
nclusters = 4 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(x)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(x)
from sklearn import metrics
score = metrics.silhouette_score(x, y_cluster_kmeans)
print(score)