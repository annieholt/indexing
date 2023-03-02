"""
Point quadtree. Part 1.

History:

  November 12, 2016

    The search_pqtree function now returns None if the point is found and
    is_find_only is set to False so that points will not be duplicated
    in the tree.

  November 19, 2015

      changed the two conditions in search_pqtree to:
         if p.x>=q.point.x
      and
         if p.y>=q.point.y

      This forces the consistency in how the four quads are determined in
      functions search_pqtree and insert_pqtree
      (Thanks to Hui Kong for examining the code!)

Contact:
Ningchuan Xiao
The Ohio State University
Columbus, OH
"""

"""
Updates by Atsushi Nara
- Updates: 02/13/2023
- Updates: 02/13/2023
-- Added get_depth_quadtree() 
-- Added self.count attribute to PQuadTreeNode()
-- Added supplemental functions to plot a point quadtree
"""



import sys
sys.path.append('../geom')
from geom.point import *
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

__author__ = "Ningchuan Xiao <ncxiao@gmail.com>"
__author__ = "Atsushi Nara <anara@sdsu.edu>"


class PQuadTreeNode():
    def __init__(self,point,nw=None,ne=None,se=None,sw=None):
        self.point = point
        self.nw = nw
        self.ne = ne
        self.se = se
        self.sw = sw

        self.count = {'nw': 0, 'ne': 0, 'se': 0, 'sw': 0}

    def __repr__(self):
        return str(self.point)
    def is_leaf(self):
        return self.nw==None and self.ne==None and \
            self.se==None and self.sw==None

def search_pqtree(q, p, is_find_only=True):
    if q is None:
        return
    if q.point == p:
        if is_find_only:
            return q
        else:
            return
    dx,dy = 0,0
    if p.x >= q.point.x:
        dx = 1
    if p.y >= q.point.y:
        dy = 1
    qnum = dx+dy*2
    child = [q.sw, q.se, q.nw, q.ne][qnum]
    if child is None and not is_find_only:
        return q
    return search_pqtree(child, p, is_find_only)

def insert_pqtree(q, p):
    n = search_pqtree(q, p, False)
    node = PQuadTreeNode(point=p)
    if p.x < n.point.x and p.y < n.point.y:
        n.sw = node
        q.count['sw'] += 1
    elif p.x < n.point.x and p.y >= n.point.y:
        n.nw = node
        q.count['nw'] += 1
    elif p.x >= n.point.x and p.y < n.point.y:
        n.se = node
        q.count['se'] += 1
    else:
        n.ne = node
        q.count['ne'] += 1

def pointquadtree(data):
    root = PQuadTreeNode(point = data[0])
    for p in data[1:]:
        insert_pqtree(root, p)
    return root


# Q4
# sort data by y variables, then create tree, so branches are always se or sw of parents

def pointquadtree_sesw(data):
    # sorting lambda function
    # sorting data greatest to least by y, so always south of starting point
    data.sort(key=lambda p: p.y, reverse=True)
    # then create nodes based on that data ordering
    root = PQuadTreeNode(point=data[0])
    for p in data[1:]:
        insert_pqtree(root, p)
    return root



# Q5

def pointquadtree_opt(data, depth=0):
    l = len(data)

    if l == 0:
        return
    # alternate by axis
    axis = l % 2
    # sort by x or y axis
    data.sort(key=lambda d: d[axis])
    # complete your codes here (hint: pivot)
    # get center point, from x or y sorting
    pivot = l//2

    # use pivot, determine which quadrant
    # based on x or y
    # creating pivot point
    p = data[pivot]

    # creating empty lists for quad datasets
    nw = []
    ne = []
    se = []
    sw = []

    # sort by x-axis
    if axis == 0:
        # complete codes here
        # when x axis, first split by east and west
        west = data[0:pivot]
        # print(west)
        east = data[pivot+1:]
        # print(east)

        # then split again north and south
        for i in west:
            if i.y >= p.y:
                nw.append(i)
            else:
                sw.append(i)
        for j in east:
            if i.y >= p.y:
                ne.append(j)
            else:
                se.append(j)

    # sort by y-axis
    # why y axis, first split by north and south
    elif axis == 1:
        north = data[pivot+1:]
        south = data[0:pivot]
        # then split east and west
        for i in north:
            if i.x >= p.x:
                ne.append(i)
            else:
                nw.append(i)
        for j in south:
            if i.x >= p.x:
                se.append(j)
            else:
                sw.append(j)

    # complete your codes here
    node = PQuadTreeNode(point=data[pivot],
                         nw=pointquadtree_opt(nw, depth + 1),
                         ne=pointquadtree_opt(ne, depth + 1),
                         se=pointquadtree_opt(se, depth + 1),
                         sw=pointquadtree_opt(sw, depth + 1))
    # also finish adding the count info
    node.count['nw'] += len(nw)
    node.count['ne'] += len(ne)
    node.count['se'] += len(se)
    node.count['sw'] += len(sw)

    return node


# supplemental functions
# get the depth of a point quadtree
def get_depth_quadtree(qt):
    if qt is None:
        return -1

    return max(get_depth_quadtree(qt.nw)+1,
               get_depth_quadtree(qt.ne)+1,
               get_depth_quadtree(qt.se)+1,
               get_depth_quadtree(qt.sw)+1)

# find all nodes in a quadtree
def find_nodes(t, points=[]):
    if t is None:
        return
    else:
        points.append(t.point)

    if t.ne:
        find_nodes(t.ne, points)
    if t.nw:
        find_nodes(t.nw, points)
    if t.se:
        find_nodes(t.se, points)
    if t.sw:
        find_nodes(t.sw, points)
    return points

# drawing a quad
def draw_rect(qt, p_min, p_max, margin, ax, depth=0, max_depth=None):

    if qt is None:
        return

    depth += 1

    if max_depth:
        if depth > max_depth:
            return

    l1 = [[qt.point.x, p_min[1] - margin[1][0]] ,[qt.point.x,p_max[1] + margin[1][1]]]
    l2 = [[p_min[0] - margin[0][0], qt.point.y],[p_max[0] + margin[0][1], qt.point.y]]

    line_segments = LineCollection([l1, l2], linewidths=0.5, color='gray', linestyle='solid')
    ax.add_collection(line_segments)

    if qt.ne:
        draw_rect(qt.ne, (qt.point.x, qt.point.y), p_max, [[0, margin[0][1]], [0, margin[1][1]]], ax, depth, max_depth)
    if qt.nw:
        draw_rect(qt.nw, (p_min[0], qt.point.y), (qt.point.x, p_max[1]), [[margin[0][0], 0], [0, margin[1][1]]], ax, depth, max_depth)
    if qt.se:
        draw_rect(qt.se, (qt.point.x, p_min[1]), (p_max[0], qt.point.y), [[0, margin[0][1]], [margin[1][0], 0]], ax, depth, max_depth)
    if qt.sw:
        draw_rect(qt.sw, p_min, (qt.point.x, qt.point.y), [[margin[0][0], 0], [margin[1][0], 0]], ax, depth, max_depth)

    return

# plot a quadtree
def plot_qt(qt, margin=[[1,1],[1,1]], max_depth=None, flag_plot_coordinates=False, pt_size=15):
    points = find_nodes(qt, points=[])

    fig, ax = plt.subplots(1,1)
    fig.figsize=(12,8)
    plt.title("Quadtree")

    p_min = Point(min([p.x for p in points]), min([p.y for p in points]))
    p_max = Point(max([p.x for p in points]), max([p.y for p in points]))

    #plot points & coordinates
    ax.scatter([p.x for p in points], [p.y for p in points], color='red', s=pt_size)

    if flag_plot_coordinates:
        for i, p in enumerate(points):
            ax.text(p.x + 0.1, p.y + 0.1, "({},{})".format(p.x, p.y), color='grey', fontsize=8)

    #draw quads
    depth=0
    draw_rect(qt, p_min, p_max, margin, ax, depth, max_depth)

    # x-y limit
    plt.xlim(p_min[0] - margin[0][0], p_max[0] + margin[0][1])
    plt.ylim(p_min[1] - margin[1][0], p_max[1] + margin[1][1])

    #plt.axis('off')
    plt.show()

# print a quadtree
def print_qt(qt, s_prefix="", depth=0):

    if qt is None:
        return
    elif depth == 0:
        s_prefix = "\nRT:"

    s = "{}{}{}".format('\t'*depth, s_prefix, qt.point)
    print(s)

    depth += 1
    if qt.ne:
        print_qt(qt.ne, "NE:", depth)
    if qt.nw:
        print_qt(qt.nw, "NW:", depth)
    if qt.se:
        print_qt(qt.se, "SE:", depth)
    if qt.sw:
        print_qt(qt.sw, "SW:", depth)

    return



if __name__ == '__main__':
    data1 = [(2,2), (0,5), (8,0), (9,8), (7,14), (13,12), (14,13)]
    points = [Point(d[0], d[1]) for d in data1]
    q = pointquadtree(points)
    print([search_pqtree(q, p) for p in points])

    print(q.count)
    plot_qt(q)

