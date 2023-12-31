import numpy as np
from collections import Counter
from euclidean import euclidean_distance
#This program is to mimic the KNN (K nearest neighbors) algorithm 
# that is found in many machine learning libraries



class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y): #Fitting points
        self.X_train = X
        self.y_train = y


    def predict(self, X): #Predicting function
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

