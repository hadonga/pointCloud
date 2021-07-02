import numpy as np
import math
from module.pcd2bin import points_to_cylinder

point=np.array([[-11,-0.0000001,1],[2,1,1],[-1,40,3]])
# point = np.array([x for x in point if 0 < x[0] + 51.2 < 102.4 and 0 < x[1] + 51.2 < 102.4])

def mySqrt(x):
    if x == 0:
        return 0
    cur = 1
    while True:
        pre = cur
        cur = (cur + x / cur) / 2
        if abs(cur - pre) < 1e-6:
            return int(cur)

point = np.array([x for x in point if mySqrt(np.power(x[0],2) + np.power(x[1],2)) < 51.2])
print(point)

coor = np.zeros(shape=(3,), dtype=np.int32)
print(coor)

segid_1=int(np.floor((math.atan2(point[0,1],point[0,0])/math.pi*50))+50)
segid_2=math.atan2(point[1,1],point[1,0])/math.pi*180
segid_3=math.atan2(point[2,1],point[2,0])/math.pi*180

print(segid_1,segid_2,segid_3)

pcd_data= np.fromfile("./003939.bin",dtype='float32').reshape(-1,4)[:100000,:3]

pcd_data = np.array([x for x in pcd_data if 0 < x[0] + 45 < 90 and 0 < x[1] + 45 < 90 and 0< x[2]+5 < 8])

a,b,c=points_to_cylinder(pcd_data)

print(a.shape, b.shape, c.shape)
print (pcd_data.shape)

