#!/usr/bin/env python3
'''
    Normal distribution
'''


class Normal:
    '''
        Class Normal that represents
        a normal distribution
    '''

    def __init__(self, data=None, mean=0., stddev=1.):
        '''
            Class constructor
        '''
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.stddev = float(stddev)
            if mean <= 0:
                raise ValueError('mean must be a positive value')
            self.mean = float(mean)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = float(sum(data) / len(data))
            self.stddev = float(
                (sum([(x - self.mean) ** 2 for x in data]) / len(data)) ** 0.5
            )
