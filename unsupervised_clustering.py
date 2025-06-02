import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import mean_squared_error

# Load the zoo dataset
df = pd.read_csv('zoo.data')

# Check basic information about our dataset
df.info()

# Find the unique number of high-level classes
unique_classes = np.unique(df['class'])
print(f"Number of unique classes: {len(unique_classes)}")

# Visualize the distribution of classes
plt.figure(figsize=(10, 6))
plt.bar(unique_classes, [sum(df['class'] == c) for c in unique_classes])
plt.xlabel('Class Label')
plt.ylabel('Number of Animals')
plt.title('Distribution of Animals across Classes')
plt.show()

# Extract features for clustering (drop non-feature columns)
features = df.drop(['animal_name', 'class'], axis=1)

# Implement agglomerative clustering
cluster_model = AgglomerativeClustering(n_clusters=7, linkage='average', affinity='cosine')

# Fit the model and predict clusters
labels_pred = cluster_model.fit_predict(features)

# Calculate the root mean squared error
rmse = np.sqrt(mean_squared_error(df['class']-1, labels_pred))
print(f"RMSE: {rmse:.2f}")