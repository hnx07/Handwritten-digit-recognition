import numpy as np
import os
import model

def loadData(datasetPath):
    data = np.loadtxt(datasetPath, delimiter=",");
    label = data[:, 0].astype(int); #Y
    img = data[:,1:].T / 255; #X
    return img, label

def saveWeight(savePath, W1, b1, W2, b2, W3, b3):
    np.savez(savePath, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)

def executeTraining():
    datasetPath = "data/mnist_train.csv";
    savePath = "weights/weights.npz";
    X, Y = loadData(datasetPath);
    alpha = 0.15;
    iterations = 700;
    W1, b1, W2, b2, W3, b3 = model.performGradientDescent(X, Y, alpha, iterations);
    saveWeight(savePath,W1, b1, W2, b2, W3, b3);

if __name__ == '__main__':
    executeTraining();