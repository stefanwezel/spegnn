import numpy as np
import matplotlib.pyplot as plt
import math
# def angle_between(p1, p2):
#     ang1 = np.arctan2(*p1)
#     ang2 = np.arctan2(*p2)
#     # print((ang1 - ang2) % (2 * np.pi))
#     return np.rad2deg((ang1 - ang2) % (2 * np.pi))



def slope(p1,p2):
    # Assignments made purely for readability. One could opt to just one-line return them
    x0 = p1[0]
    y0 = p1[1]
    x1 = p2[0]
    y1 = p2[1]
    if (x1 - x0) == 0:
        return 0
    else:
        return (y1 - y0) / (x1 - x0)


def angle(slope):
    return np.rad2deg((np.arctan2(slope)))


# p1 = (5,1)
# # p2 = (1, 5)
# # print(slope(p1,p2))
# slope = slope(p2,p1)
# print(slope)
# # print(angle(slope))
# angle = np.rad2deg(np.arctan2(p1[-1] - p1[0], p2[-1] - p2[0]))
# print(angle)
# # print(angle_between(p1,p2))

# # plt.plot(p1)
# # plt.show()
def azimuthAngle( x1, y1, x2, y2):
  angle = 0.0;
  dx = x2 - x1
  dy = y2 - y1
  if x2 == x1:
    angle = math.pi / 2.0
    if y2 == y1 :
      angle = 0.0
    elif y2 < y1 :
      angle = 3.0 * math.pi / 2.0
  elif x2 > x1 and y2 > y1:
    angle = math.atan(dx / dy)
  elif x2 > x1 and y2 < y1 :
    angle = math.pi / 2 + math.atan(-dy / dx)
  elif x2 < x1 and y2 < y1 :
    angle = math.pi + math.atan(dx / dy)
  elif x2 < x1 and y2 > y1 :
    angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
  return (angle * 180 / math.pi)


# print(azimuthAngle(0,0, 0,-1))
import numpy
import math

def get_bearing(p1,p2):

    lat1 = p1[1]
    long1 = p1[0]
    lat2 = p2[1]
    long2 = p2[0]

    


    dLon = (long2 - long1)
    x = math.cos(math.radians(lat2)) * math.sin(math.radians(dLon))
    y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dLon))
    brng = np.arctan2(x,y)
    brng = np.degrees(brng)

    # if brng < 0:
    #     return 360 + brng



    return brng


p1 = (0,0)
p2 = (-0.001,1)


print(get_bearing(p1, p2))
