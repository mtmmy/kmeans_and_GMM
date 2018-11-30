#%%
# imports and reading data
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import loadCSV
from kmeans import kmeans
from pca import pca
from gmm import gmm

plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['grid.color'] = 'black'
plt.rcParams['figure.edgecolor'] = 'white'
plt.rcParams['savefig.transparent'] = False

data = loadCSV.loadFile("audioData.csv")

#%%
# Kmeans
ks = [k for k in range(2, 11)]
oriKmeans = kmeans(data)
plt.figure(1)
plt.plot(ks, oriKmeans)
plt.xlabel('Ks', color="black")
plt.ylabel('Object Function Value', color="black")
plt.title('Kmeans', color="black")
plt.savefig('Kmeans')

#%%
# Kmeans with PCA
pcaKmeans = kmeans(pca(data))
plt.figure(2)
plt.plot(ks, pcaKmeans)
plt.xlabel('Ks', color="black")
plt.ylabel('Object Function Value', color="black")
plt.title('PCA Kmeans', color="black")
plt.savefig('Kmeans with PCA')

#%%
# GMM
centers, clusters = gmm(data)
c1 = np.array(clusters[0])
c2 = np.array(clusters[1])

plt.figure(3)
plt.scatter(centers[0][0], centers[0][1], marker="*", c="r", s=500)
plt.scatter(centers[1][0], centers[1][1], marker="*", c="b", s=500)
plt.scatter(c1[:,0], c1[:,1], c="r")
plt.scatter(c2[:,0], c2[:,1], c="b")
plt.title("GMM", color="black")
plt.savefig('GMM')

#%%
#GMM PCA
centers, clusters = gmm(pca(data))
c1 = np.array(clusters[0])
c2 = np.array(clusters[1])

plt.figure(4)
plt.scatter(centers[0][0], centers[0][1], marker="*", c="r", s=500)
plt.scatter(centers[1][0], centers[1][1], marker="*", c="b", s=500)
plt.scatter(c1[:,0], c1[:,1], c="r")
plt.scatter(c2[:,0], c2[:,1], c="b")
plt.title("PCA GMM", color="black")
plt.savefig('GMM with PCA')