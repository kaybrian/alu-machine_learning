#!/usr/bin/env python3
"""
    Defines function that calculates the cofactor matrix of a matrix
"""


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix

    Args:
        matrix (list of lists): Matrix whose cofactor
        matrix should be calculated

    Returns:
        list of lists: The cofactor matrix of matrix
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if not matrix:
        raise ValueError("matrix must be a non-empty square matrix")
    height = len(matrix)
    if any(len(row) != height for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if height == 1:
        return [[1]]

    cofactor_matrix = []
    for row_i in range(height):
        cofactor_row = []
        for column_i in range(height):
            sub_matrix = [row[:column_i] + row[column_i+1:]
                          for row in (matrix[:row_i] + matrix[row_i+1:])]
            determinant_value = determinant(sub_matrix)
            cofactor_row.append((-1) ** (row_i + column_i) * determinant_value)
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix


def determinant(matrix):
    """
    Calculates the determinant of a matrix

    Args:
        matrix (list of lists): Matrix whose determinant should be calculated

    Returns:
        int or float: The determinant of matrix
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")
    height = len(matrix)
    if not matrix:
        raise ValueError("matrix must be a non-empty square matrix")
    for row in matrix:
        if len(row) != height:
            raise ValueError("matrix must be a square matrix")

    if height == 1:
        return matrix[0][0]
    elif height == 2:
        a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
        return a * d - b * c

    det = 0
    for i, element in enumerate(matrix[0]):
        sub_matrix = [row[:i] + row[i+1:] for row in matrix[1:]]
        det += element * (-1) ** i * determinant(sub_matrix)
    return det
