import math


def signedDistToLine2Pts(pt, p0, p1):
    '''
    return a signed distance to a line where line is defined as two points
    positive sign refers to "above" the line or "left" of a vertical line
    to get the expected sign of "right" is positive, the vertical line will be inverted back under the "angle_sign" in _computeNodeHeights() of MergeTree.py

    pt: tuple
    p0: tuple
    p1: tuple
    '''
    return ((p0[0] - pt[0]) * (p1[1] - p0[1]) - (p1[0] - p0[0]) * (p0[1] - pt[1])) / math.dist(p0, p1)


def distToLine2Pts(pt, p0, p1):
    '''
    pt to line as defined by p0, p1 https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line

    pt: tuple
    p0: tuple
    p1: tuple
    '''
    return abs(signedDistToLine2Pts(pt, p0, p1))


def intersectionToLine2Pts(pt, p0, p1):
    '''
    intersection of a point to a line defined by two points

    pt: tuple
    p0: tuple
    p1: tuple
    '''
    a = p0[1] - p1[1]
    b = p1[0] - p0[0]
    c = p0[0] * p1[1] - p1[0] * p0[1]

    x = (b * (b * pt[0] - a * pt[1]) - a * c) / (a ** 2 + b ** 2)
    y = (a * (-b * pt[0] + a * pt[1]) - b * c) / (a ** 2 + b ** 2)

    return (x, y)
