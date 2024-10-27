# import math
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
import kmeans as kmeans
import symnmf as symnmf
from sklearn.metrics import silhouette_score

np.random.seed(1234)

def main():
    try:
        # Get data from console
        input_data = sys.argv
        k, input_file = int(input_data[1]), input_data[2]
        # Create Vectors dataframe from csv file
        vectors = pd.read_csv(input_file, header=None)

        # calculate KMeans matrix with pandas DataFrame
        kmeansMatrix = kmeans.doKmeans(vectors, k)
        # Convert vectors to python list of lists (to use in SymNMF)
        vectors = vectors.values.tolist()
        symnmfMatrix = symnmf.doSymnmf(vectors, k) # The SymNMF matrix        

    except Exception:
        print("An Error Has Occurred")


if __name__ == "__main__":
    main()