import numpy as np
numsOfNeuron = 64
numsOfOutput = 10
imgSize = 784

def calRelu(Z):
    return np.maximum(0, Z);

def calReluDerivative(Z):
    return Z > 0;

def calSoftmax(Z):
    # use axis=0 to compute sample seperately (mini-batches)
    expZ = np.exp(Z - np.max(Z, axis=0));
    return expZ / np.sum(expZ, axis=0);

def encodeOneHot(Y):
    temp = np.zeros((Y.size, numsOfOutput));
    temp[np.arange(Y.size), Y] = 1;
    return temp.T;

def initializeParameters():
    # will optimize with He Initialization later
    W1 = np.random.rand(numsOfNeuron, imgSize) - 0.5;
    b1 = np.random.rand(numsOfNeuron, 1) - 0.5;
    W2 = np.random.rand(numsOfNeuron, numsOfNeuron) - 0.5;
    b2 = np.random.rand(numsOfNeuron, 1) - 0.5;
    W3 = np.random.rand(numsOfOutput, numsOfNeuron) - 0.5;
    b3 = np.random.rand(numsOfOutput, 1) - 0.5;
    return W1, b1, W2, b2, W3, b3;

def propagateForward(X, W1, b1, W2, b2, W3, b3):
    Z1 = W1.dot(X) + b1;
    A1 = calRelu(Z1); 
    Z2 = W2.dot(A1) + b2;
    A2 = calRelu(Z2);
    Z3 = W3.dot(A2) + b3;
    A3 = calSoftmax(Z3);
    return Z1, A1, Z2, A2, Z3, A3 

def propagateBackward(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    m = Y.size
    dZ3 = A3 - encodeOneHot(Y);
    dW3 = 1/m * dZ3.dot(A2.T);
    db3 = 1/m * np.sum(dZ3, axis=1, keepdims=True);

    dZ2 = W3.T.dot(dZ3) * calReluDerivative(Z2);
    dW2 = 1/m * dZ2.dot(A1.T);
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True);

    dZ1 = W2.T.dot(dZ2) * calReluDerivative(Z1);
    dW1 = 1/m * dZ1.dot(X.T);
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True);
    return dW1, db1, dW2, db2, dW3, db3;

def updateParameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1;
    b1 = b1 - alpha * db1;
    W2 = W2 - alpha * dW2;
    b2 = b2 - alpha * db2;
    W3 = W3 - alpha * dW3;
    b3 = b3 - alpha * db3;
    return W1, b1, W2, b2, W3, b3;

def getPredictions(A3):
    return np.argmax(A3, 0);

def calculateAccuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size;

def performGradientDescent(X, Y, alpha, iterations):
    W1, b1, W2, b2, W3, b3 = initializeParameters();
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = propagateForward(X, W1, b1, W2, b2, W3, b3);
        dW1, db1, dW2, db2, dW3, db3 = propagateBackward(Z1, A1, Z2, A2,Z3, A3, W1, W2, W3, X, Y);
        W1, b1, W2, b2, W3, b3 = updateParameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha);
        if (i % 10 == 0):
            print("Iteration: ", i, "| Acurracy: ", round(calculateAccuracy(getPredictions(A3), Y),4));
    return W1, b1, W2, b2, W3, b3