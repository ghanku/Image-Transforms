#!env python
# -*- coding: utf-8 -*-
# Dominant Color Descriptor of 2D image.
# file created by A.Chabira
# original matlab implementaion by https://github.com/Molen1945
# License: Public Domain
#
# reference:
# Shao, H., Wu, Y., Cui, W., Zhang, J. (2008). Image retrieval based on MPEG-7 dominant color descriptor. Proceedings of the 9th International Conference for Young Computer Scientists, ICYCS 2008, 753–757. https://doi.org/10.1109/ICYCS.2008.89

import numpy as np
import imageio
import skimage.color


def DCD(img, img_type='RGB', order=8):
    ''' Calculates the Dominant Color Descriptor described in the MPEG-7 standard

            arguments
              img  : either HSV image, or RGB image
              img_type : input image type either 'HSV' or RGB
              order : n first dominant colors

            return
                  2 numpy arrays, one  with size equal to input image, and second ###72
          '''

    if img_type == 'RGB':
        img = skimage.color.rgb2hsv(img)

    # HSV component sorting
    H = 360 * img[:, :, 0]
    S = img[:, :, 1]
    V = img[:, :, 2]

    # HSV Component Quantization
    Hq = QunatizeH(H)
    Sq = QunatizeSV(S)
    Vq = QunatizeSV(V)

    # HSV matrix generation
    C = np.round(9 * Hq + 3 * Sq + Vq)

    m, n = C.shape
    color, _ = np.histogram(C, bins=72, range=(0, 72))
    Pi = color / (m * n)

    M = order

    # Pi values ​​are sorted in descending order and stored in Qj
    # I : the index of the Pi values ​​that have been sorted in descending order

    I = np.argsort(Pi)[::-1]  # indices of sorted elements
    Qj = np.sort(Pi)[::-1]

    # Take the first 8 values ​​of Qj
    Qj = Qj[0:M]

    Pi1 = np.zeros(72)
    I = I[0:M]
    Pi1[I] = Qj

    P = Pi1 / sum(Qj)

    return (P, C)


def QunatizeH(H):

    bins = np.array([20, 40, 75, 155, 190, 270, 295, 316])
    ix = np.digitize(H, bins=bins)
    return ix


def QunatizeSV(SV):

    bins = np.array([1, 0.7, 0.2])
    ix = np.digitize(SV, bins=bins, right=True)
    return ix


if __name__ == '__main__':
    input = 'https://raw.githubusercontent.com/mikolalysenko/lena/master/lena.png'
    I = imageio.imread(input, pilmode='RGB')
    P, C = DCD(I, img_type='RGB', order=8)
    print('done')
