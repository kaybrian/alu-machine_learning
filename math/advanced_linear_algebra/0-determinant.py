#!/usr/bin/env python3
"""
    a function def determinant(matrix):
    that calculates the determinant of a matrix:
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix

    Args:
        - matrix: list of lists whose determinant

    Returns:
        - the determinant of matrix

    """
    # Check if the input is a list of lists
    if isinstance(matrix, list) is False:
        raise TypeError("matrix must be a list of lists")
    height = len(matrix)
    if height == 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if isinstance(row, list) is False:
            raise TypeError("matrix must be a list of lists")
        if len(row) == 0 and height == 1:
            return 1
        if len(row) != height:
            raise ValueError("matrix must be a square matrix")
    if height == 1:
        return matrix[0][0]
    if height == 2:
        return (matrix[0][0] * matrix[0][1]) - (matrix[1][0] * matrix[1][1])

    multiplier = 1
    d = 0
    for i in range(height):
        element = matrix[0][i]
        sub_matrix = []
        for row in range(height):
            if row == 0:
                continue
            new_row = []
            for column in range(height):
                if column == i:
                    continue
                new_row.append(matrix[row][column])
            sub_matrix.append(new_row)
        d += element * multiplier * determinant(sub_matrix)
        multiplier *= -1
    return d
