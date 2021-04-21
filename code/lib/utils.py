
import numpy as np
import cv2
from sklearn.cluster import KMeans
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

def display_images_plt(image1: np.array, image2: np.array) -> None:
    fig = plt.figure(figsize=(15,20))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(image1)
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

def simple_pixelation(image: np.array, w: int, h: int) -> np.array:
    # Save original image size
    height, width = image.shape[:2]

    # Collapse te image to the desired size using  a linear interpolation
    temp = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

    # Return the image to the original size but use NEAREST interpolation to adjust the pixels
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)


def fill_parcelation(parcel_image: np.array, image: np.array, k: int) -> np.array:
    '''

    :param parcel_image:
    :param image:
    :param k:
    :return:
    '''
    clusterValues = []
    for _ in range(0, k):
        clusterValues.append([])

    for r in range(0, parcel_image.shape[0]):
        for c in range(0, parcel_image.shape[1]):
            clusterValues[parcel_image[r][c]].append(image[r][c])

    imageC = np.copy(image)

    clusterAverages = []
    for i in range(0, k):
        clusterAverages.append(np.average(clusterValues[i], axis=0))

    for r in range(0, parcel_image.shape[0]):
        for c in range(0, parcel_image.shape[1]):
            imageC[r][c] = clusterAverages[parcel_image[r][c]]

    return imageC


def parcelate_image(image: np.array, k: int) -> np.array:
    '''

    :param image:
    :param k:
    :return:
    '''
    imageC = np.copy(image)

    h = image.shape[0]
    w = image.shape[1]

    # vectorize image fom 2d to 1d
    imageC.shape = (image.shape[0] * image.shape[1], 3)

    # Find most-similar pixels using the 3 channels as features
    kmeans = KMeans(n_clusters=k, random_state=42).fit(imageC).labels_

    # Return the vector back to matrix form. Now we have a parcelation with k colors
    kmeans.shape = (h, w)

    return kmeans

def kMeans_pixelation(image: np.array, k: int) -> np.array:
    '''

    :param image:
    :param k:
    :return:
    '''
    idx = segmentImgClrRGB(image, k)
    return colorClustering(idx, image, k)