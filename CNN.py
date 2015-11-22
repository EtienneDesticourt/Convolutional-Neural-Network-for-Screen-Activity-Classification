from scipy.signal import convolve2d 
import numpy as np

def convolve(image, kernels):
    #Init convolutionnal layer
    side = image.shape[0] - kernels.shape[0] + 1 #We assume they're both square
    numKernels = len(kernels)
    shape = (side, side, numKernels)
    convolvedLayer = np.zeros(shape)
    #Fill it by convolving kernels with the image
    for k in range(numKernels):
        kernel = kernels[k]
        convolvedLayer[:,:,k] = convolve2d(image, kernel, 'valid')

    return convolvedLayer

def RELU(array):
    array[array < 0] = 0
    return array

def pool(convLayer, factor):
    if convLayer.shape[0] % factor != 0: raise ValueError("Convolutionnal layer's shape needs to be divisible by given factor.")
    side = int(convLayer.shape[0] / factor)
    downsampled = np.empty((side, side, convLayer.shape[2]))
    
    for c in range(side):
        columns = slice(c*factor, (c+1)*factor)
        for r in range(side):
            rows = slice(r*factor, (r+1)*factor)
            downsampled[c,r,:] = np.mean(convLayer[columns,rows,:])

    return downsampled

#Fully connected layer logic
def activaton(array):
    return 1. / (np.exp(-array) + 1)

def predict(inputLayer, weights):
    return activation(inputLayer.dot(weights))

def backpropagate(inputLayer, fcLayer, labels, weights, alpha=0.01):
    deltas = (labels - fcLayer) * (1 - fcLayer) * fcLayer
    weights += alpha * np.outer(deltas, inputLayer)
    return weights






                        

