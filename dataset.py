import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

data = pd.read_csv('genres_v2.csv')

# drop the unnecessary columns
cols_to_drop = ['type', 'id', 'uri', 'track_href',
                'analysis_url', 'Unnamed: 0', 'title',
                'song_name']

data.drop(cols_to_drop, axis = 1, inplace = True)

# turn data into float type
data.iloc[:, : -1] = data.iloc[:, : -1].values.astype(float)

# turn nominal data into numerical data
encoder = LabelEncoder()
data['genre'] = encoder.fit_transform(data['genre'])

# feature scale the data (except label-encoded data)
scaler = StandardScaler()
data.iloc[:, : -1] = scaler.fit_transform(data.iloc[:, : -1])

# run pca to reduce the number of features to 2
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data.iloc[:, : -1])

# concat pca with label encoded column
data_final = pd.concat([pd.DataFrame(data_pca, columns = ['pca1', 'pca2']), data['genre']], axis = 1)

# concat pca without label encoded column
data_final_no_genre = pd.DataFrame(data_pca, columns = ['pca1', 'pca2'])

###########
# Get the principal components
components = pca.components_

# Assuming your original data is stored in a DataFrame named 'data'
original_columns = data.columns

# Create a DataFrame to store the importance of each original feature in each principal component
component_importance = pd.DataFrame(components, columns=original_columns)

# Transpose the DataFrame to make it easier to interpret
component_importance = component_importance.T

# Print the importance of each original feature in each principal component
print("Importance of each original feature in each principal component:")
print(component_importance)
###########

# heatmap to show the correlation between the features
# plt.figure(figsize = (10, 6))
# sns.heatmap(data.corr(), annot = True, cmap = 'coolwarm', linewidths = 0.5)
# plt.show()
# print(data.shape)

# plot 2d scatter plot to show the distribution of the data with a legend
plt.figure(figsize = (10, 6))
sns.scatterplot(data = data_final, x = 'pca1', y = 'pca2', palette = 'viridis')
plt.xlabel('pca1')
plt.ylabel('pca2')
plt.show()

# print 3d scatter plot to show the distribution of the data with a legend
# fig = plt.figure(figsize = (10, 6))
# ax = fig.add_subplot(111, projection = '3d')
# ax.scatter(data_final['pca1'], data_final['pca2'], data_final['genre'], c = data_final['genre'], cmap = 'viridis')
# ax.set_xlabel('pca1')
# ax.set_ylabel('pca2')
# ax.set_zlabel('genre')

# plt.show()

# create k means clustering model
# kmeans = KMeans(n_clusters = 10, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

# # fit the model
# y_kmeans = kmeans.fit_predict(data_final_no_genre.iloc[:, : -1])

# # plot the clusters with the centroids
# plt.figure(figsize = (10, 6))
# plt.scatter(data_final_no_genre['pca1'], data_final_no_genre['pca2'], c = y_kmeans, cmap = 'viridis')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'red', label = 'Centroids')
# plt.xlabel('pca1')
# plt.ylabel('pca2')
# plt.show()

# # print the silhouette score
# from sklearn.metrics import silhouette_score
# print(silhouette_score(data_final_no_genre.iloc[:, : -1], y_kmeans))