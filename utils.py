import numpy as np
import math
from scipy.spatial import distance
import matplotlib.pyplot as plt




def get_bearing(p1,p2):
    lat1 = p1[1]
    long1 = p1[0]
    lat2 = p2[1]
    long2 = p2[0]
    # lat1 = p1[0]
    # long1 = p1[1]
    # lat2 = p2[0]
    # long2 = p2[1]

    dLon = (long2 - long1)
    x = math.cos(math.radians(lat2)) * math.sin(math.radians(dLon))
    y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dLon))
    brng = np.arctan2(x,y)
    brng = np.degrees(brng)

    return brng


def get_highest_contrast_neighbor(g, node, neighbors):
    """ TODO """
    c1 = g.nodes[node]['mean color']
    highest_contrast_neighbor = None
    highest_contrast = -np.inf
    for neighbor in neighbors:
        c2 = g.nodes[neighbor]['mean color']
        dist = distance.euclidean(c1, c2)
        if dist > highest_contrast:
            highest_contrast_neighbor = neighbor
            highest_contrast = dist                    
    return highest_contrast_neighbor




def get_neighbors(node, edges):
    """ TODO """
    return list(
        set(
            [edge[1] for edge in edges if edge[0]==node] + [edge[0] for edge in edges if edge[1]==node]
            ))





def plot_line(ax, center, slope, length=1):

    """ TODO """
    if slope > 30:
        slope = 30
    elif slope < -30:
        slope = -30
    else:
        slope = slope

    if -0.01 < slope < 0.01:
        length = 2
    elif (slope > 5) or (slope < -5):
        length = 0.2
    else:
        length=length

    b = center[1] - slope * center[0]
    pt1 = (center[0] - length, slope * (center[0] - length) + b)
    pt2 = (center[0] + length, slope * (center[0] + length) + b)

    # ax.plot((pt1[0], center[0]), (pt1[1], center[1]), color='red', linewidth=0.5)
    ax.plot((center[0], pt2[0]), (center[1], pt2[1]), color='red', linewidth=0.5)