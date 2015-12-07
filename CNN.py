from scipy.signal import convolve2d 
import numpy as np

def convolve(image, kernels):
    #Init convolutionnal layer
    side = image.shape[0] - kernels.shape[1] + 1 #We assume they're both square
    numKernels = len(kernels)
    shape = (side, side, numKernels)
    convolvedLayer = np.zeros(shape)
    #Fill it by convolving kernels with the image
    for k in range(numKernels):
        kernel = kernels[k]
        convolvedLayer[:,:,k] = convolve2d(image, kernel, 'valid')

    return convolvedLayer

def RELU(array):
    result = array.copy()
    result[array < 0] = 0
    return result

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
def activation(array):
    return 1. / (np.exp(-array) + 1)

def predict(inputLayer, weights):
    return activation(inputLayer.dot(weights))

def backpropagate(inputLayer, fcLayer, labels, weights, alpha=0.01):
    deltas = (labels - fcLayer) * (1 - fcLayer) * fcLayer
    weights += alpha * np.outer(deltas, inputLayer)
    return weights

def error(output, target):
    return sum(0.5 * (target - output) ** 2)
    

##def backprop(inputFC, output, target, weights, alpha=0.01):
##    delta = (output - target) * output  * (1 - output)
##    weights -= alpha * np.outer(delta2, inputFC)
##    return weights.T

def backpropFC(inputs, outputs, targets, weights, alpha=0.01):
    delta = (outputs - targets) * outputs  * (1 - outputs)
    #convDelta = (delta*weights).sum() * (1 - outputCONV) * outputConv
    weights -= alpha * np.outer(delta, inputs)
    return weights.T

def backpropRELU(inputs, outputs, targets, weights, alpha=0.01):
    pass

def calcDeltaFC(outputs, targets, alpha=0.01):
    return (outputs - targets) * outputs  * (1 - outputs)
    
def calcDeltaCONV(outputs, delta, weights, alpha=0.01):
    return (delta * weights).sum() * (1 - outputs) * outputs

def backprop(inputsFC, outputsFC, weightsFC, inputsCONV, outputsCONV, weightsCONV, label):
    deltaFC = calcDeltaFC(outputFC, label)
    deltaCONV = calcDeltaCONV(outputCONV, deltaFC, weightsFC)

    newWeightsFC = weightsFC - alpha * np.outer(deltaFC, inputsFC)
    #figure oout derivative of convolution here

def initTest():
    #Random 4*4 array with values between -1 and 1
    image = np.random.rand(4,4) * 2. - 1
    label = np.array([1,0])
    #4 random 2*2 convolution filters
    kernels = np.random.rand(4,2,2)
    #Put the image through the CONV and RELU layers
    outputCONV = convolve(image,kernels)
    inputCONV = outputCONV
    outputCONV = activation(outputCONV)
    outputRELU = RELU(outputCONV)
    #Create random weights with shape of output
    numWeights = np.ndarray.flatten(outputRELU).shape
    weightsFC = np.random.rand(numWeights[0],2)



    
##    inputFC = np.ndarray.flatten(outputRELU)
##    for i in range(1):
##        outputFC = predict(inputFC, weightsFC)
##        weightsFC = backprop(inputFC, outputFC, label, weigh
    print(outputFC)
    return image, kernels

a,b = initTest()

                        

