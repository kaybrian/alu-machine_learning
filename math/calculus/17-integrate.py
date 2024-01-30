#!/usr/bin/env python3
'''
    a function def poly_integral(poly, C=0):
    that calculates the integral of a polynomial:
'''


def poly_integral(poly, C=0):
    # Check if poly is a list and C is an integer
    if not isinstance(poly, list) or not all(isinstance(coeff, (int, float)) for coeff in poly) or not isinstance(C, int):
        return None

    integral = [C]

    for i, coeff in enumerate(poly):
        if not isinstance(coeff, (int, float)):
            return None

        integral.append(coeff / (i + 1))

    while integral[-1] == 0 and len(integral) > 1:
        integral.pop()

    return integral
