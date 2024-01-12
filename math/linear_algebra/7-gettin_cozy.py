#!/usr/bin/env python3
'''
    A function def cat_matrices2D(mat1, mat2, axis=0):
    that concatenates two matrices along a specific axis:
'''


def cat_matrices2D(mat1, mat2, axis=0):
    '''
        A function def cat_matrices2D(mat1, mat2, axis=0):
        that concatenates two matrices along a specific axis:
    '''
    if axis == 0:
        return mat1 + mat2
    else:
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
