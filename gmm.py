import random, sys
import numpy as np

def pdfGMM(centers, record, covM, pU):
    ps = []
    for i in range(2):
        c = centers[i]
        distance = record - c
        powNum = -0.5 * np.dot(np.dot(distance, np.linalg.inv(covM)), distance.T)
        ps.append(np.exp(powNum) * pU[i])
    sumP = sum(ps)
    return ps[0] / sumP, ps[1] / sumP

def Estep(data, centers, covM, pU):
    EZij = []
    for row in data:
        p1, p2 = pdfGMM(centers, row, covM, pU)
        EZij.append([p1, p2])
    return np.array(EZij)

def Mstep(EZij):
    pU = [0, 0]
    pU[0] = sum(EZij[:,0]) / len(EZij)
    pU[1] = sum(EZij[:,1]) / len(EZij)
    return pU

def gmm(data):
    cIdx = random.sample(range(len(data)), 2)
    centers = [data[i] for i in cIdx]
    covM = np.cov(data.T)
    N, D = np.shape(data)
    covDet = np.linalg.det(covM)
    # divisor = np.power(np.power((2 * np.pi), D) * covDet, 0.5)
    pU = [0.5, 0.5]
    tolerance = 0.00001
    EZij = []
    while 1:
        EZij = Estep(data, centers, covM, pU)
        newPU = Mstep(EZij)
        if abs(newPU[0] - pU[0]) < tolerance:
            break
        pU = newPU
    
    clusters = [[], []]

    for i in range(len(data)):
        if EZij[i][0] > EZij[i][1]:
            clusters[0].append(data[i,:2])
        else:
            clusters[1].append(data[i,:2])

    twoDCenters = [c[:2] for c in centers]
    return twoDCenters, np.array(clusters)
    
