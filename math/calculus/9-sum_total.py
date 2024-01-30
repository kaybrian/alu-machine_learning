#!/usr/bin/env python3
'''
    a function def summation_i_squared(n):
    that calculates the summation
    of all numbers from 1 to n
'''


def summation_i_squared(n):
    '''
    calculates the summation
    of all numbers from 1 to n
    '''
    sum = 0
    for i in range(1, n + 1):
        sum += i ** 2
    return sum
