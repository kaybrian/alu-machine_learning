#!/usr/bin/env python3
"""
    A function def convolve_grayscale_same(images, kernel):
    that performs a same convolution on grayscale images:
"""


import numpy as np


def convolve_grayscale_same(images, kernel):
    '''
        A function def convolve_grayscale_same(images, kernel):

        Args:
            images is a numpy.ndarray with shape (m, h, w)
            containing multiple grayscale images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
            kernel is a numpy.ndarray with shape (kh, kw)
            containing the kernel for the convolution
            kh is the height of the kernel
            kw is the width of the kernel
        Returns:
            a numpy.ndarray containing
            the convolved images

    '''
    m, h, w = images.shape
    kh, kw = kernel.shape
    if (kh == kw):
        y = h - kh + 1
        x = w - kw + 1

        # Add padding to the images
        pad_y = kh // 2
        pad_x = kw // 2
        padded_images = np.pad(images, ((0, 0), (pad_y, pad_y),
                                        (pad_x, pad_x)), mode='constant')

        convolved_image = np.zeros((m, y, x))
        for i in range(y):
            for j in range(x):
                shadow_area = padded_images[:, i:i + kh, j:j + kw]
                convolved_image[:, i, j] = \
                    np.sum(shadow_area * kernel, axis=(1, 2))
        return convolved_image
