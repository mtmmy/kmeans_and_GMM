import numpy as np
import loadCSV

def pca(records):
    means = np.mean(records.T, axis=1)
    centered = records - means
    covM = np.cov(centered.T)
    eigenVals, eigenVectors = np.linalg.eig(covM)

    sortedIdx = eigenVals.argsort()[::-1] 
    eigenVals = eigenVals[sortedIdx]
    eigenVectors = eigenVectors[:,sortedIdx]

    eigenVectors = eigenVectors[:,:2]
    return records.dot(eigenVectors)

