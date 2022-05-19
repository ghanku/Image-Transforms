#!env python
# -*- coding: utf-8 -*-
# Slantlet Transform of 2D image.
# file created by A.Chabira
# original matlab implementaion by https://github.com/sanjumaramattam/Image-Transforms
# License: Public Domain
#
# reference:
# Selesnick, I. W. (1999). The Slantlet Transform. In IEEE TRANSACTIONS ON SIGNAL PROCESSING (Vol. 47, Issue 5)


import numpy as np
import imageio
import skimage.color
from numpy import sqrt, zeros, eye
from numpy import concatenate as con
from skimage.transform import resize


def slant_transform(img, order='default'):
    ''' Calculates Slantelet Transform of grayscale image

            arguments
              img   : input single channel only image or 2D array (preferably square of shape 2**n)
              order : order of slant transform, either 'default' calculated from image shape , or int

            return
                  a numpy 2D array with size equal to size 2**n, where n is the order of slant matrix required
          '''

    img = skimage.color.rgb2gray(img)
    N = img.shape[0]
    if (order == 'default'):
        n = np.ceil(np.log2(N)).astype(int)  # calculate order of slant
    elif (isinstance(order, int)):
        n = order

    if (np.log2(img.shape[0]) % 1 != 0):  # check if shape is in the form (2**n)
        print('image is not of shape 2**n, resizing ..')
        img = resize(img, (2**n, 2**n))

    S = slant_matrix(n)
    S_t = S @ img @ np.transpose(S)  # clalculate slant transform

    return S_t


def slant_matrix(n):
    ''' Calculates Slantelet matrices of order n

            arguments
              n  : order of slant matrix

            return
                  a numpy 2D array with size equal to 2**n
          '''
    # init S1
    S = 1 / sqrt(2) * np.array([[1, 1], [1, -1]])
    a = 1

    for i in range(2, n + 1):

        b = 1 / sqrt(1 + 4 * a**2)
        a = 2 * b * a

        q1 = np.array([[1, 0], [a, b]])
        q2 = np.array([[1, 0], [-a, b]])
        q3 = np.array([[0, 1], [-b, a]])
        q4 = np.array([[0, -1], [b, a]])

        Z = con([con([S, zeros(S.shape)], axis=1),
                 con([zeros(S.shape), S], axis=1)])

        if (i == 2):
            B1 = con([q1, q2], axis=1)  # block 1
            B2 = con([q3, q4], axis=1)  # block 2
            S = (1 / sqrt(2)) * con([B1, B2]) @ Z

        else:
            k = int((2**i - 4) / 2)
            B1 = con([q1, zeros([2, k]), q2, zeros([2, k])], axis=1)  # block 1
            B2 = con([zeros([k, 2]), eye(k), zeros(
                [k, 2]), eye(k)], axis=1)  # block 2
            B3 = con([q3, zeros([2, k]), q4, zeros([2, k])], axis=1)  # block 3
            B4 = con([zeros([k, 2]), eye(k), zeros(
                [k, 2]), -eye(k)], axis=1)  # block 4

            S = (1 / sqrt(2)) * con([B1, B2, B3, B4]) @ Z

    return S


if __name__ == '__main__':
    input = 'https://raw.githubusercontent.com/mikolalysenko/lena/master/lena.png'
    I = imageio.imread(input, pilmode='RGB')
    S_t = slant_transform(I)
    print('done')
