from datasets import load_dataset
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

dataset_hf = load_dataset("maharshipandya/spotify-tracks-dataset")
# dataset = pd.read_csv('spotify-tracks-dataset.csv')

# convert to pandas dataframe
dataset = pd.DataFrame(dataset_hf['train'])

# unnecessary columns
cols_to_drop = ['Unnamed: 0', 'track_id', 'album_name',
                'loudness', 'explicit', 'time_signature',
                'instrumentalness', 'duration_ms', 'mode', 'valence', 'energy']

dataset.drop(cols_to_drop, axis=1, inplace=True)

# move track_name to the first column
track_name = dataset['track_name']
dataset.drop(labels=['track_name'], axis=1, inplace=True)
dataset.insert(0, 'track_name', track_name)

# move artist_name to the second column
artist_name = dataset['artists']
dataset.drop(labels=['artists'], axis=1, inplace=True)
dataset.insert(1, 'artists', artist_name)

# move genre to the third column
genre = dataset['track_genre']
dataset.drop(labels=['track_genre'], axis=1, inplace=True)
dataset.insert(2, 'track_genre', genre)

# feature scale the data (except first column)
scaler = StandardScaler()

# turn data into float type
dataset.iloc[:, 3:] = dataset.iloc[:, 3:].values.astype(float)

# feature scale the data (except label-encoded data)
dataset.iloc[:, 3:] = scaler.fit_transform(dataset.iloc[:, 3:])

# create heatmap
# plt.figure(figsize = (10, 10))
# sns.heatmap(dataset.iloc[:, 1:].corr(), annot = True, cmap = 'coolwarm', linewidths = 0.5)
# plt.show()

# histogram
# dataset.hist(figsize=(12, 8), bins=20)
# plt.tight_layout()
# plt.show()

# run pca to reduce the number of features to 2
pca = PCA(n_components=2)
data_pca = pca.fit_transform(dataset.iloc[:, 3:])

# create kmeans++ model
kmeans = KMeans(n_clusters=10, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(data_pca)

# print the silhouette score
# print(silhouette_score(data_pca, pred_y))

# plot the clusters
plt.scatter(data_pca[pred_y == 0, 0], data_pca[pred_y == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(data_pca[pred_y == 1, 0], data_pca[pred_y == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(data_pca[pred_y == 2, 0], data_pca[pred_y == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(data_pca[pred_y == 3, 0], data_pca[pred_y == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(data_pca[pred_y == 4, 0], data_pca[pred_y == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.scatter(data_pca[pred_y == 5, 0], data_pca[pred_y == 5, 1], s=100, c='orange', label='Cluster 6')
plt.scatter(data_pca[pred_y == 6, 0], data_pca[pred_y == 6, 1], s=100, c='purple', label='Cluster 7')
plt.scatter(data_pca[pred_y == 7, 0], data_pca[pred_y == 7, 1], s=100, c='brown', label='Cluster 8')
plt.scatter(data_pca[pred_y == 8, 0], data_pca[pred_y == 8, 1], s=100, c='pink', label='Cluster 9')
plt.scatter(data_pca[pred_y == 9, 0], data_pca[pred_y == 9, 1], s=100, c='grey', label='Cluster 10')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of songs')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()

# create dataframe of each sample
cluster_0 = dataset[pred_y == 0].sample(1).columns
cluster_1 = dataset[pred_y == 1].sample(1)
cluster_2 = dataset[pred_y == 2].sample(1)
cluster_3 = dataset[pred_y == 3].sample(1)
cluster_4 = dataset[pred_y == 4].sample(1)

# create a playlist of each cluster
playlist_0 = dataset[pred_y == 0].sample(10)
playlist_1 = dataset[pred_y == 1].sample(10)
playlist_2 = dataset[pred_y == 2].sample(10)
playlist_3 = dataset[pred_y == 3].sample(10)
playlist_4 = dataset[pred_y == 4].sample(10)

# print the playlist
print(playlist_0)