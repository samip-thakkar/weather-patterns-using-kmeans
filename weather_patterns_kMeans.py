# -*- coding: utf-8 -*-
"""

@author: Samip
"""

#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics

#Load dataset to dataframe
df = pd.read_csv('minute_weather.csv')

""" As the dataset has over 1.5 million rows, we need to sample the data randomly. Taking a random sample of every 10th row."""

sample_df = df[df['rowID'] % 1000 == 0]
#Now we have 158726 rows and 13 columns

#We can drop rain_accumulation and rain_duration columns
df1 = sample_df.drop(columns = ['rain_accumulation', 'rain_duration'])
#Now we have 11 columns

#After dropping some of the columns as they are either unimportant or redundant
cols_of_interest = ['air_pressure', 'air_temp', 'avg_wind_direction', 'avg_wind_speed', 'max_wind_direction', 'max_wind_speed', 'relative_humidity']

data = df1[cols_of_interest]
data = data[np.isfinite(data['avg_wind_direction'])]
#Now we have 7 columns

#As the values are of different sizes, we need to scale them to give them equal importance
ss = StandardScaler()
X = ss.fit_transform(data)

#K-MEANS CLUSTERING

"""We need to determine the optimal number of clusters. We have two methods for this.
1. Silhouete Co-efficient: mean intra-cluster distance(a) and mean near-cluster distance(b). Silhouete co-efficient: (b - a) / max(b - a)
2. Calinski Harabasz Score / Variance Ratio: within-cluster dispersion / between-cluster dispersion """

#Let's implement K-means using n_clusters = 12
k_means = KMeans(n_clusters = 12)

#Run the clustering algorithm
model = k_means.fit(X)

#Generate cluster predictions and store in y_hat
y_hat = k_means.predict(X)

#Calculate the Silhouette coe-efficient
labels = k_means.labels_
print(metrics.silhouette_score(X, labels, metric = 'euclidean'))

#Calculate the Calinski Harabasz Score
print(metrics.calinski_harabasz_score(X, labels))

#Try all the process for K = 8
k_means_8 = KMeans(n_clusters = 8)
model = k_means_8.fit(X)
y_hat_8 = k_means_8.predict(X)
labels_8 = k_means_8.labels_
print(metrics.silhouette_score(X, labels_8, metric = 'euclidean'))
print(metrics.calinski_harabasz_score(X, labels_8))

#We can see k = 8 gives better results, but for these we need to do many iterations. So, we will use Elbow plot to find optimal value of 'k'.
sum_of_squared_distances = []
for k in range(1, 15):
    k_means = KMeans(n_clusters = k)
    model = k_means.fit(X)
    sum_of_squared_distances.append(k_means.inertia_)

#Plot the elbow plot
plt.plot(range(1, 15), sum_of_squared_distances, 'bx-')
plt.xlabel('K')
plt.ylabel('Sum_of_squared_distance')
plt.title('Elbow method for optimal value of k') 
plt.show()   

#From elbow method, we can see that optimal value is k = 5. We will confirm it by Silhouette co-efficient and Calinski Harabasz score
k_means_5 = KMeans(n_clusters=5)
model = k_means_5.fit(X)
y_hat_5 = k_means_5.predict(X)
labels_5 = k_means_5.labels_
print(metrics.silhouette_score(X, labels_5, metric = 'euclidean'))
print(metrics.calinski_harabasz_score(X, labels_5))
#Thus, we can conclude that k = 5 is the optimal value

#Plot the result of items
plt.scatter(X[:, 0], X[:, 1], c=y_hat_5, s=10, cmap='viridis')

centers = k_means_5.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=1100, alpha=0.5)