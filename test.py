import numpy as np

point=np.array([[30,-45,-10],[1,2,-80],[1,2,3]])
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