import math
import sys
import pandas as pd
import numpy as np
import mysymnmfsp as SymNMF

np.random.seed(1234)

def initializeH(w_mat, N, k):
    m = np.mean(w_mat) # Calculate the average of all entries in w_mat
    upper_bound = 2 * np.sqrt(m / k) # Calculate the upper bound for the random values
    H = np.random.uniform(low=0, high=upper_bound, size=(N, k)) # Initialize H with random values from the interval [0, upper_bound]
    return H.tolist()

def doSymnmf(vectors, k):
    w_mat = SymNMF.norm(vectors) # Calling norm function in C to calculate W matrix
    h_mat = initializeH(w_mat, len(vectors), k) # Initialize H matrix
    matrix_goal = SymNMF.symnmf(w_mat, h_mat, k) # Calling symnmf function in C to calculate the matrix
    return matrix_goal

def main():
    try:
        # Get data from console
        input_data = sys.argv
        k, goal, input_file = int(input_data[1]), input_data[2], input_data[3]

        # Create Vectors dataframe from csv file
        vectors = pd.read_csv(input_file, header=None)
        # Convert vectors to python list of lists
        vectors = vectors.values.tolist()
        
        matrix_goal = None # The matrix to calculate and return

        # Choose which matrix to calculate and return
        match goal:
            case "sym":
                matrix_goal = SymNMF.sym(vectors) # Calling sym function in C to calculate the matrix
            case "ddg":
                matrix_goal = SymNMF.ddg(vectors) # Calling ddg function in C to calculate the matrix
            case "norm":
                matrix_goal = SymNMF.norm(vectors) # Calling norm function in C to calculate the matrix
            case "symnmf":
                matrix_goal = doSymnmf(vectors, k)
            case _:
                print("An Error Has Occurred")
                return

        # print matrix_goal until 4 decimal points
        for row in matrix_goal:
            print(','.join(format(x, ".4f") for x in row))

    except Exception:
        print("An Error Has Occurred")


if __name__ == "__main__":
    main()
