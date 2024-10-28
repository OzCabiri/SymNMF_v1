import math
import sys
import pandas as pd
import numpy as np
import kmeans as kmeans
import symnmf as symnmf
from sklearn.metrics import silhouette_score

"""
finds the closets centroid to a given vector and returns the index of the centroid
@param vector: input vector in a list form
@type X: list of lists
@return: norm matrix
@rtype: list of lists (size: N*N)
"""
def findClosestCentroid(vector, centroids): 
    min_distance = math.inf
    cluster_index = 0

    for j in range(len(centroids)):
        distance = np.linalg.norm(vector - centroids[j])
        if distance < min_distance:
            min_distance = distance
            cluster_index = j

    return cluster_index

# Calculate kmeans labels for the vectors
def calculateKmeansLabels(vectors, k): 
    vectors = vectors.to_numpy()  # kmeans requires numpy array
    kmeansMatrix = kmeans.doKmeans(vectors, k)
    kmeansLabels = [-1 for i in range(len(vectors))]

    for i in range(len(vectors)):
        clusterIndex = findClosestCentroid(vectors[i], kmeansMatrix)
        kmeansLabels[i] = clusterIndex

    return kmeansLabels

def findMaxInRow(matrix, row):
    max_value = -math.inf
    max_index = 0

    for i in range(len(matrix[row])):
        if matrix[row][i] > max_value:
            max_value = matrix[row][i]
            max_index = i

    return max_index

def calculateSymnmfLabels(vectors, k):
    vectors = vectors.values.tolist() # Convert data to list of lists
    symnmfMatrix = symnmf.doSymnmf(vectors, k)

    return np.array(symnmfMatrix).argmax(axis=1)

def main():
    try:
        # Get data from console
        input_data = sys.argv
        k, input_file = int(input_data[1]), input_data[2]
        # Create Vectors dataframe from csv file
        vectors = pd.read_csv(input_file, header=None)

        kmeansLables = calculateKmeansLabels(vectors, k)
        scoreKmeans = silhouette_score(vectors, kmeansLables)

        # Convert vectors to python list of lists (to use in SymNMF)
        symnmfLabels = calculateSymnmfLabels(vectors, k)
        scoreSymnmf = silhouette_score(vectors, symnmfLabels)

        print("nmf: " + format(scoreSymnmf, ".4f"))
        print("kmeans: " + format(scoreKmeans, ".4f"))

    except Exception:
        print("An Error Has Occurred")


if __name__ == "__main__":
    main()