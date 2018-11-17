import loadCSV
import pca
import random
import numpy as np

def getOFV(clusters, centers):
    ofv = 0
    for i in range(len(centers)):
        for cluster in clusters[i]:
            ofv += sum((cluster - centers[i]) ** 2)
    return ofv

def clustering(centers, data):
    clusters = [[] for _ in range(len(centers))]
    for d in data:
        distances = []
        for c in centers:
            distances.append(np.linalg.norm(d - c))
        closest = distances.index(min(distances))
        clusters[closest].append(d)
    return (getOFV(clusters, centers), clusters)

def kmeans(data):
    allOFV = []
    for c in range(2, 11):
        cIdx = random.sample(range(len(data)), c)
        centers = [data[c] for c in cIdx]
        oldOFV = -1
        while 1:
            ofv, clusters = clustering(centers, data)
            if ofv == oldOFV:
                break
            oldOFV = ofv
            centers = [sum(np.array(c), 0) / len(c) for c in clusters]
        allOFV.append(oldOFV)
    print(allOFV)

# main
ks = [k for k in range(2, 11)]
data = loadCSV.loadFile("audioData.csv")
print(ks)
print("Original Data")
kmeans(data)
print("PCAed Data")
kmeans(pca.reduction(data))


# covData = np.cov(data)
# print("")