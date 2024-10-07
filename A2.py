# Dinh Hoang Viet Phuong - 301123263


# import all necessary libraries
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 1. Retrieve and load the Olivetti faces dataset
# Fetch the Olivetti faces dataset
data = fetch_olivetti_faces(shuffle=True, random_state=42)
faces = data.images
X = data.data
y = data.target

# Display the first few images as an example
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(faces[i], cmap='gray')
    plt.title(f"Face {y[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()


# 2. Split the training set, a validation set, and a test set using stratified sampling to ensure that there are the same number of images per person in each set. Provide your rationale for the split ratio
# Split the data into training and temporary sets (80/20)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Split the temporary set into training and validation sets (75/25 to achieve an overall 60/20/20 split)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

print("Training set:", len(X_train))
print("Validation set:", len(X_val))
print("Test set:", len(X_test))


# 3. Using k-fold cross validation, train a classifier to predict which person is represented in each picture, 
# and evaluate it on the validation set
# Define a classifier
clf = SVC(kernel='linear', random_state=42)

# Define Stratified K-Folds cross-validator
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Compute cross-validation scores
scores = cross_val_score(clf, X_train, y_train, cv=kfold)

# Print the cross-validation scores and their average
print("Cross-validation scores:", scores)
print("Average cross-validation score:", scores.mean())

# Train the classifier on the whole training set
clf.fit(X_train, y_train)

# Evaluate the classifier on the validation set
val_score = clf.score(X_val, y_val)
print("Validation set score:", val_score)


# 4. Use K-Means to reduce the dimensionality of the set. Provide your rationale for the similarity measure used 
# to perform the clustering. Use the silhouette score approach to choose the number of clusters
# Find the optimal number of clusters using silhouette score
silhouette_scores = []
cluster_range = range(2, 50)  # Check for number of clusters from 2 to 50; adjust as necessary

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_train)
    silhouette_avg = silhouette_score(X_train, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters = {n_clusters}, the silhouette score is {silhouette_avg:.4f}.")

# Plot the silhouette scores to determine the optimal number of clusters
plt.plot(cluster_range, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.grid(True)
plt.show()

# Select the number of clusters with the highest silhouette score
optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
print(f"The optimal number of clusters is: {optimal_clusters}")

# Perform KMeans with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
X_train_reduced = kmeans.fit_transform(X_train)
X_val_reduced = kmeans.transform(X_val)


# 5. Use the set from step (4) to train a classifier as in step (3)
# Define a classifier
clf_reduced = SVC(kernel='linear', random_state=42)

# Define Stratified K-Folds cross-validator
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Compute cross-validation scores on the reduced data
scores_reduced = cross_val_score(clf_reduced, X_train_reduced, y_train, cv=kfold)

# Print the cross-validation scores and their average for the reduced data
print("Cross-validation scores (reduced data):", scores_reduced)
print("Average cross-validation score (reduced data):", scores_reduced.mean())

# Train the classifier on the whole transformed training set
clf_reduced.fit(X_train_reduced, y_train)

# Evaluate the classifier on the transformed validation set
val_score_reduced = clf_reduced.score(X_val_reduced, y_val)
print("Validation set score (reduced data):", val_score_reduced)


# 6. Apply DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm to the Olivetti Faces 
# dataset for clustering. Preprocess the images and convert them into feature vectors, then use DBSCAN to group 
# similar images together based on their density. Provide your rationale for the similarity measure used to 
# perform the clustering, considering the nature of facial image data.
# Ensure that X is a 2D array
if len(X.shape) == 1:
    raise ValueError("The data appears to be in the wrong shape. It should be a 2D array.")

# Normalize the images
X_normalized = StandardScaler().fit_transform(X)

# Apply DBSCAN with cosine similarity
dbscan = DBSCAN(eps=0.5, min_samples=5, metric="cosine")  # Adjust eps and min_samples as necessary
clusters = dbscan.fit_predict(X_normalized)

# Print the unique cluster labels
print("Unique cluster labels:", np.unique(clusters))
