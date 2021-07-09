# to test dynamic seg
import numpy as np
import math
import time
# from module.pcd2cylinder import points_to_cylinder
#
# pcd_files= ["./000010.bin","./000011.bin","./000012.bin","./000013.bin","./000014.bin","./000003.bin"]
# for pcd_file in pcd_files:
#     pcd_data= np.fromfile(pcd_file,dtype='float32').reshape(-1,4)#[:100000,:]
#     print(pcd_data.shape)
#     pcd_squr = np.array([x for x in pcd_data if 0 < x[0] + 45 < 90 and 0 < x[1] + 45 < 90 and 0< x[2]+5 < 8])
#     print(pcd_squr.shape)
#     pcd_round =np.array([x for x in pcd_data if  2 < math.sqrt(x[0]**2 + x[1]**2) < 34.75 and 0 < x[2] + 5 < 8])
#     print("Radius- Number of points: {}".format(pcd_round.shape))
#
#     a,b,c =points_to_cylinder(pcd_round)
#     print(a.shape)

cyliner_shape= [3.6, 0.5, 8]
a=0.1
k=0.005
cyliner_shape[1]=[]
cyliner_shape[1].append(0)


for i in range(1, 101):
    temp = a + k*i
    temp += cyliner_shape[1][i-1]
    cyliner_shape[1].append(temp)

print(cyliner_shape[1])
print(len(cyliner_shape[1]))

radius_test=np.random.randn(1000000,1)
radius_test.tolist()
rad_id=[]
checkp1 =time.time()
for r in radius_test:
    for dis in cyliner_shape[1]:
        if r-dis < 0:
            rad_id.append(cyliner_shape[1].index(dis))
            break

print(time.time()-checkp1)

rad_id2=[]
checkp2 = time.time()
for r in radius_test:
    rad_id2.append(r/0.8)

print(time.time()-checkp2)

# print(rad_id)




