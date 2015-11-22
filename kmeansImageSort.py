import numpy as np
import os, time
from PIL import Image
from sklearn.cluster import KMeans
from shutil import copyfile

def featureExtractor(img):
    result = np.asarray(img)
    #Flatten all but last dimension (RGB)
    result = result.reshape(-1, result.shape[-1])
    return result

def featureGenerator(path, resize=None):
    for f in os.listdir(path):
        filePath = os.path.join(path, f)
        try:
            img = Image.open(filePath)
        except OSError: #Deals with corrupted images
            continue
        if resize:
            img = img.resize(resize, Image.ANTIALIAS)
        yield featureExtractor(img)

def buildArray(path, side):
    "Creates a numpy array of all the features of all the files in path"
    data = np.empty((0, side**2, 3))
    for features in featureGenerator(path, numFeatures, (side, side)):
        data = np.append(data, [features], axis=0)
    return data

PATH = "C:\\Users\\Etienne\\Desktop\\Captures"
SIDE = 25

data = buildArray(PATH, SIDE)
#Sum RGB channels
data = np.sum(data, axis=2)

KM = KMeans(n_clusters=20)
clusters = KM.fit_predict(data)
