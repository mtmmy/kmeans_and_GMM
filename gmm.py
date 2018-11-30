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

def Mstep(EZij, data):
    newCenters = [[], []]
    newCenters[0] = sum(data * EZij[:,0][:, None]) / sum(EZij[:,0])
    newCenters[1] = sum(data * EZij[:,1][:, None]) / sum(EZij[:,1])

    pU = [0, 0]
    pU[0] = sum(EZij[:,0]) / len(EZij)
    pU[1] = sum(EZij[:,1]) / len(EZij)
    return pU, newCenters

def gmm(data):
    np.random.seed(10)
    cIdx = [np.random.randint(0, 127) for _ in range(2)]
    centers = [data[i] for i in cIdx]
    covM = np.cov(data.T)
    pU = [0.5, 0.5]
    tolerance = 0.0001
    EZij = []
    while 1:
        EZij = Estep(data, centers, covM, pU)
        newPU, centers = Mstep(EZij, data)        
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
    
