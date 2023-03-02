"""
quadtree1.py

Region Quadtree

Original codes from GIS Algorithms by Ningchuan Xiao
2/13/2022 Modified by Atsushi Nara
"""

import numpy as np
import time
from matplotlib import pyplot as plt


# To Do - Q1
# updated quad class to include depth, values, mean attributes
# set default values as None to those attributes
class Quad():
    def __init__(self, value, dim, nw, ne, se, sw, depth=None, values=None, mean=None):
        self.value = value
        self.dim = dim
        self.nw = nw
        self.ne = ne
        self.se = se
        self.sw = sw
        self.depth = depth
        self.values = values
        self.mean = mean

    def __repr__(self):
        return str(self.value)

def homogeneity(d):
    """
    Check homogeneity
    :param d: ndarray
    :return: boolean
    """
    v = d[0,0]
    for i in d:
        for j in i:
            if j != v:
                return False
    return True

# To Do
# Q1: Modify def quadtree()

# Modifications I made:
# add input of depth, root depth is 0 so set as default
# assign values attribute to each Quad, just used the input data
# mean is mean of those values, or the homogeneous value
# depth adds one as recursion runs
def quadtree(data, depth = 0):
    """
    :param data:
    :param depth: the depth of the node
    :return: Quad
    """
    dim = data.shape[0]     #size (# of columns/rows)
    dim2 = int(dim/2)       #half-size
    if homogeneity(data):
        return Quad(value=data[0,0], dim=dim,
                    nw=None,
                    ne=None,
                    se=None,
                    sw=None,
                    depth=depth,
                    values=data,
                    mean=data[0,0])

    return Quad(value=None,
                dim=dim,
                nw = quadtree(data[0:dim2, 0:dim2], depth+1),
                ne = quadtree(data[0:dim2, dim2:,], depth+1),
                se = quadtree(data[dim2:, dim2:,], depth+1),
                sw = quadtree(data[dim2:, 0:dim2], depth+1),
                depth=depth,
                values=data,
                mean=np.mean(data))

# ------------------------------------------------------------------------
# Note: x, y start at the upper-left corner
#
# Child node quadrants are orderd as
# [q.nw, q.ne, q.sw, q.se].
# The side of cell (x, y) in each region is determined by dx and dy.
# Then the quadrant (x, y) is located in is calculated as dx+dy*2.
# where dx is 0 (left) or 1 (right), dy is 0 (upper) or 1 (lower)
#
#           dx
#   -----+-----+-----
#        |  0  |  1
#   -----+-----+-----
# dy  0  |  0  |  1
#   -----+-----+-----
#     1  |  3  |  2
#   -----+-----+-----
#
# ------------------------------------------------------------------------

def query_quadtree(q, x, y):
    if q.value is not None: # this is a leaf node
        return q.value      # return the value

    dim = q.dim
    dim2 = dim/2
    dx, dy = 0, 0
    # checking if x is left (0) or right (1)
    if x >= dim2:
        dx = 1
        x = x - dim2

    # checking if y is upper (0) or lower (1)
    if y >= dim2:
        dy = 1
        y = y - dim2
    qnum  = dx + dy * 2
    return query_quadtree([q.nw, q.ne, q.sw, q.se][qnum], x, y)


# TO Do - # Q2

# want to return a mean value at a specified depth
# copied code from query tree method to start with, these are my modifications:
# stop point is when query reaches desired/input tree depth (if q.depth == depth); at that point, return the mean value
# otherwise, go to next quad and run the query again; include the depth in that function call

def query_quadtree_depth(q, x, y, depth, flag_print=False):

    # make the stop point at the specified depth
    if q.depth == depth:  # this is the desired depth of node
        return q.mean  # return the mean value (can be from homogenous region or not)

    dim = q.dim
    dim2 = dim / 2
    dx, dy = 0, 0

    # checking if x is left (0) or right (1)
    if x >= dim2:
        dx = 1
        x = x - dim2

    # checking if y is upper (0) or lower (1)
    if y >= dim2:
        dy = 1
        y = y - dim2
    qnum = dx + dy * 2
    # qnum indicates which segment of quad to select in following argument

    return query_quadtree_depth([q.nw, q.ne, q.sw, q.se][qnum], x, y, depth)



def get_test_data():
    # 4x4
    data0 = np.array(
        [
            [0, 1, 2, 2],
            [2, 2, 2, 2],
            [0, 0, 2, 0],
            [0, 0, 1, 0]
        ]
    )

    # 16x16
    data1 = np.array(
        [
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],

            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],

            [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],

            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]
        ]
    )

    return [data0, data1]



def run_quadtree(data):
    # create a quadtree
    q = quadtree(data)

    # search each value at x, y
    for y in range(q.dim):
        s = ""
        for x in range(q.dim):
            res = query_quadtree(q, x , y)
            s += "{} ".format(res)
        print(s)
    return q


def compare_performance(n_start=3, n_iter=9, int_low=0, int_high=10, dtype=np.uint8):
    """
    A function to compare quadtree performance with a brute-force method
    :param n_start: the start exponent value of the powers of 2 (default=3)
    :param n_iter: the number of times to iterate over the list of the powers of 2
    :param int_low: the lowest integer value for a random value
    :param int_high: the highest integer value for a random value
    :param dtype: numpy data type
    :return: None
    """
    time_brute_force_time=[]
    time_quadtree_build=[]
    time_quadtree_search=[]
    time_quadtree_total=[]
    ns = []

    base2 = [2 ** j for j in range(n_start, n_iter + 1)]

    for i, n in enumerate(base2):
        print("num_iterations={}, size={}".format(i+1, n))
        data = np.random.randint(int_low, int_high, size=(n, n))

        rows = data.shape[0]
        cols = data.shape[1]

        #bruete force
        t = time.process_time()

        # for each value at y, x
        for y in range(rows):
            for x in range(cols):
                # search the value in a brute-force way
                for y2 in range(rows):
                    if y == y2:
                        for x2 in range(cols):
                            if x == x2:
                                val = data[y, x]
        dt = time.process_time() - t
        time_brute_force_time.append(dt)

        t = time.process_time()
        q = quadtree(data)
        dt = time.process_time() - t
        time_quadtree_build.append(dt)

        for y in range(rows):
            for x in range(cols):
                val = query_quadtree(q, x, y)
        dt2 = time.process_time() - t
        time_quadtree_total.append(dt2)

        time_quadtree_search.append(dt2-dt)
        ns.append(n)

    plt.gcf()
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.plot(ns, time_brute_force_time, label="Brute-Force", color='b')
    plt.plot(ns, time_quadtree_search, label="Quadtree-Search", color='wheat')
    plt.plot(ns, time_quadtree_build, label="Quadtree-Build", color='lightgreen')
    plt.plot(ns, time_quadtree_total, label="Quadtree-Total", color='r')
    ax.legend()
    plt.title("Performance Comparison - Quadtrees")
    plt.show()



if __name__ == '__main__':
    # get test data
    data = get_test_data()

    # test
    run_quadtree(data[0])

    run_quadtree(data[1])

    # performance test
    # compare_performance()


