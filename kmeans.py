import csv
import random
import numpy as np

def loadCSV(filename):
    records = []
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            records.append(np.array(row).astype(float))
    return np.array(records)

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

# main
data = loadCSV("audioData.csv")
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