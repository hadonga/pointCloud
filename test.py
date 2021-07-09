import numpy as np
import math
from module.pcd2cylinder import points_to_cylinder_dynamic,points_to_cylinder_fixed
from module.pcd2pillar import points_to_voxel
import time
import matplotlib.pyplot as plt
import torch

# point_test= np.array([2,2,2],[2,1,2],[2,3,2])
# start=time.time()
# radius_test= math.sqrt(point_test[0][0]**2+point_test[0][1]**2+point_test[0][2]**2)
# print(time.time()-start)
#
# checkpoint1 =time.time()
# radius_test1 = np.linalg.norm(point_test)
# print(time.time()-checkpoint1)
# print(radius_test)
# print(radius_test1)

# point=np.array([[-11,-0.0000001,1],[2,1,1],[-1,40,3]])
# point = np.array([x for x in point if 0 < x[0] + 51.2 < 102.4 and 0 < x[1] + 51.2 < 102.4])

# def mySqrt(x):
#     if x == 0:
#         return 0
#     cur = 1
#     while True:
#         pre = cur
#         cur = (cur + x / cur) / 2
#         if abs(cur - pre) < 1e-6:
#             return int(cur)



# segid_1=int(np.floor((math.atan2(point[0,1],point[0,0])/math.pi*50))+50)
# segid_2=math.atan2(point[1,1],point[1,0])/math.pi*180
# segid_3=math.atan2(point[2,1],point[2,0])/math.pi*180
#
# print(segid_1,segid_2,segid_3)

pcd_data= np.fromfile("./000014.bin",dtype='float32').reshape(-1,4)[:100000,:]
pcd_data = np.array([x for x in pcd_data if 0 < x[0] + 30 < 60 and 0 < x[1] + 30 < 60 and 0< x[2]+5 < 8])

# Distribution test

time1=time.time()
a,b,c=points_to_cylinder_dynamic(pcd_data)
print("Cylinder Number: ")
print(a.shape, b.shape, c.shape)
print(time.time()-time1)

time3=time.time()
x,y,z =points_to_cylinder_fixed(pcd_data)
print("Pillar Number: ")
print(x.shape, y.shape, z.shape)
print(time.time()-time3)

time2=time.time()
u,v,w =points_to_voxel(pcd_data)
print("Pillar Number: ")
print(u.shape, v.shape, w.shape)
print(time.time()-time2)


print (pcd_data.shape)


num_cylinder=[]
for i in range(a.shape[0]):
    pts_num=0
    for j in range(a.shape[1]):
        pts_num += 1
        if a[i,j,1] == 0:
            break
    num_cylinder.append(pts_num)
print(num_cylinder)

num_cylinder_fix=[]
for i in range(x.shape[0]):
    pts_num=0
    for j in range(x.shape[1]):
        pts_num += 1
        if x[i,j,1] == 0:
            break
    num_cylinder_fix.append(pts_num)
print(num_cylinder_fix)

num_cube=[]
for i in range(u.shape[0]):
    pts_num=0
    for j in range(u.shape[1]):
        pts_num += 1
        if u[i,j,1] == 0:
            break
    num_cube.append(pts_num)
print(num_cube)
#
# features=torch.zeros((3,10,5))
#
# features_intensity = torch.unsqueeze(features[:,:,3],2)
# features_intensity_1 = features[:,:,3]
# print(features_intensity.shape)
# print(features_intensity_1.shape)
#
# # pc_file = './003939.bin'
# # pc_data= np.fromfile(pc_file,dtype='float32').reshape(-1,4)
# # print(pc_data.shape)
#
# # X= pc_data[:,0]
# # Y= pc_data[:,1]
# # Z= pc_data[:,2]
# # ity= pc_data[:,3]
# # r=[]
# #
# # Z=np.array(Z)
# # Z.sort()
# # print(Z[:100],Z[-20:])
# #
# #
# # print(min(Z),max(Z))
# #
# #
# # for i in range(pc_data.shape[0]):
# #     r.append(np.sqrt(pow(pc_data[i,0],2)+pow(pc_data[i,1],2)+pow(pc_data[i,2],2)))
#
n_bins =20
fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
#
# # axs[0].hist(X,bins=n_bins )
# # axs[0].set_title('X-axis')
# # axs[1].hist(Y,bins=n_bins)
# # axs[1].set_title('Y-axis')
# # axs[2].hist(Z,bins=n_bins)
# # axs[2].set_title('Z-axis')
# # axs[3].hist(ity,bins=n_bins)
# # axs[3].set_title('intensity')
# # axs[4].hist(r,bins=n_bins)
# # axs[4].set_title('depth')
#
axs[0].hist(num_cylinder,bins=n_bins )
axs[0].set_title('cylinder_dynamic')
axs[1].hist(num_cylinder_fix,bins=n_bins)
axs[1].set_title('cylinder_fixed')
axs[2].hist(num_cube,bins=n_bins)
axs[2].set_title('pillar')

plt.show()
