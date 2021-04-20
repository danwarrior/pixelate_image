
import numpy as np
import cv2
import skimage
from sklearn.cluster import KMeans
from numpy import linalg as LA
from matplotlib import pyplot as plt


def load_image(pathToImage):
    image = cv2.imread(pathToImage)
    return image

def display_image(image: np.array) -> None:
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_image)

def display_images_pair(image1: np.array, image2: np.array) -> None:
    fig = plt.figure(figsize=(15,20))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

def simple_pixelation(image: np.array, w: int, h: int) -> np.array:
    # Save original image size
    height, width = image.shape[:2]

    # Collapse te image to the desired size using  a linear interpolation
    temp = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

    # Return the image to the original size but use NEAREST interpolation to adjust the pixels
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)


def colorClustering(idx, img, k):
    clusterValues = []
    for _ in range(0, k):
        clusterValues.append([])

    for r in range(0, idx.shape[0]):
        for c in range(0, idx.shape[1]):
            clusterValues[idx[r][c]].append(img[r][c])

    imgC = np.copy(img)

    clusterAverages = []
    for i in range(0, k):
        clusterAverages.append(np.average(clusterValues[i], axis=0))

    for r in range(0, idx.shape[0]):
        for c in range(0, idx.shape[1]):
            imgC[r][c] = clusterAverages[idx[r][c]]

    return imgC


def segmentImgClrRGB(img, k):
    imgC = np.copy(img)

    h = img.shape[0]
    w = img.shape[1]

    imgC.shape = (img.shape[0] * img.shape[1], 3)

    # 5. Run k-means on the vectorized responses X to get a vector of labels (the clusters);
    #
    kmeans = KMeans(n_clusters=k, random_state=0).fit(imgC).labels_

    # 6. Reshape the label results of k-means so that it has the same size as the input image
    #   Return the label image which we call idx
    kmeans.shape = (h, w)

    return kmeans

def kMeansImage(image, k):
    idx = segmentImgClrRGB(image, k)
    return colorClustering(idx, image, k)