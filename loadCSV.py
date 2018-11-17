import csv
import numpy as np

def loadFile(filename):
    records = []
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            records.append(np.array(row).astype(float))
    return np.array(records)