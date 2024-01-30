#!/usr/bin/env python3
'''
    a function def poly_derivative(poly):
    that calculates the derivative of a polynomial
'''


def poly_derivative(poly):
    '''
        calculates the derivative of a polynomial
    '''
    if type(poly) is not list:
        return None
    elif len(poly) <= 1:
        return [0]
    return [poly[i] * i for i in range(1, len(poly))]