
# coding=utf-8

import numpy as np
import cv2
from math import (floor, ceil)
import matplotlib.pyplot as plt


def interpolate(first_value: float, second_value: float, ratio: float) -> float:
    """Interpolate with a linear weighted sum."""
    return first_value * (1 - ratio) + second_value * ratio


def get_array_value(x: int, y: int, c: int, array: np.ndarray):
    """Returns the value of the array at position x,y."""
    return array[y, x, c]


def bilinear_interpolation(x: float, y: float, c: int,  img: np.ndarray) -> float:
    """Returns the bilinear interpolation of a pixel in the image.
    :param x: x-position to interpolate
    :param y: y-position to interpolate
    :param img: image, where the pixel should be interpolated
    :returns: value of the interpolated pixel
    """
    if x < 0 or y < 0:
        raise ValueError('x and y pixel position have to be positive!')
    if img.shape[1]-1 < x or img.shape[0]-1 < y:
        x = img.shape[1] - 1
        y = img.shape[0] - 1

    x_rounded_up = int(ceil(x))
    x_rounded_down = int(floor(x))
    y_rounded_up = int(ceil(y))
    y_rounded_down = int(floor(y))

    ratio_x = x - x_rounded_down
    ratio_y = y - y_rounded_down

    interpolate_x1 = interpolate(get_array_value(x_rounded_down, y_rounded_down, c, img),
                                 get_array_value(x_rounded_up, y_rounded_down,c, img),
                                 ratio_x)
    interpolate_x2 = interpolate(get_array_value(x_rounded_down, y_rounded_up,c, img),
                                 get_array_value(x_rounded_up, y_rounded_up, c, img),
                                 ratio_x)
    interpolate_y = interpolate(interpolate_x1, interpolate_x2, ratio_y)

    return interpolate_y

def bilinear_resize(filename):
    # Read all 3Channels
    original_image = cv2.imread(filename, -1)
    rows, cols, channels = original_image.shape
    print(rows, cols)
    resizing_factor = 4
    new_image = np.zeros(shape =(rows*resizing_factor, cols*resizing_factor, channels))
    print(new_image.shape)

    # Now iterate over all the pixels in the new image 
    # And fill corresponding values with bilinear_interpolated values

    for i in range(0, new_image.shape[0]):
        for j in range (0, new_image.shape[1]):
            for k in range (new_image.shape[2]):
                new_image[i][j][k] = bilinear_interpolation((j/resizing_factor) ,(i/resizing_factor), k, original_image)


    cv2.imwrite('Interpolated.jpg', new_image) 

    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     cap.release()
    #     cv2.destroyAllWindows()



if __name__ == '__main__':
    # image = np.arange(0, 9).reshape((3, 3))
    # print(bilinear_interpolation(0.5, 1.5, image))

    bilinear_resize('dogsmall.jpg')