
import numpy as np
import cv2
from matplotlib import pyplot as plt


def load_image(pathToImage):
    img = cv2.imread(pathToImage)

def display_image(image: np.array) -> None:
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_image)

def display_images_pair(image1: np.array, image2: np.array) -> None:
    fig = plt.figure()
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



