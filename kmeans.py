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
        np.random.seed(10)
        cIdx = [np.random.randint(0, 127) for _ in range(c)]
        centers = [data[i] for i in cIdx]
        oldOFV = -1
        while 1:
            ofv, clusters = clustering(centers, data)
            if ofv == oldOFV:
                break
            oldOFV = ofv
            centers = [0 if len(cluster) == 0 else sum(np.array(cluster), 0) / len(cluster) for cluster in clusters]
        allOFV.append(oldOFV)
    return allOFV