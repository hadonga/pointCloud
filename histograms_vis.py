import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from  matplotlib.ticker import PercentFormatter
import open3d as o3d
from pcd2pillar import points_to_voxel,points_to_pillar
import os
import tqdm
import time

#
# vol_cnt=0
# total_time =0
# # for i in range(10):
#     time_start =time.time()
#     pcd_dir= "H:/0.Dataset/0.semanticKitti/dataset/sequences/0"+str(i)+"/velodyne"
#     print(pcd_dir)
#     for pcd_file in tqdm.tqdm(os.listdir(pcd_dir)):
#         pcd_data=np.fromfile(os.path.join(pcd_dir,pcd_file), dtype=np.float32).reshape(-1,4)
#         points= pcd_data[:,:3]
#
#         # pcd=o3d.geometry.PointCloud()
#         # pcd.points=o3d.utility.Vector3dVector(points)
#         # o3d.visualization.draw_geometries([pcd])
#
#         # Raw points
#         # intensity= pcd_data[:,3]
#         # X= pcd_data[:,0]
#         # Y= pcd_data[:,1]
#         # Z=pcd_data[:,2]
#         # r= np.sqrt(np.sum(np.square(points),axis=1))
#         #
#         # Points in Pillar
#         voxels, coors, num_points_per_voxel =points_to_pillar(points=pcd_data,
#                             # voxel_size=(0.8,0.8,8),
#                             # coors_range=[-51.2, -51.2, -4, 51.2, 51.2, 4],
#                             voxel_size=(0.8, 0.8),
#                             coors_range=[-51.2, -51.2, 51.2, 51.2],
#                             max_points= 100,
#                             max_voxels= 10000)
#
#         if coors.shape[0]>6000:
#             vol_cnt +=1
#             print(vol_cnt)
#             print(voxels.shape, coors.shape, num_points_per_voxel.shape)
#     time_end = time.time()
#     time_loop= time_end-time_start
#     total_time += time_loop
#     print(time_loop)
#
# print(vol_cnt)
# print("totol time: {} s".format(total_time))



# num_big=0
# for i in num_points_per_voxel:
#     if i>900:
#         num_big +=1
# print("nums >50:{}".format(num_big))


pc_file = './003939.bin'
pc_data= np.fromfile(pc_file,dtype='float32').reshape(-1,4)
print(pc_data.shape)

X= pc_data[:,0]
Y= pc_data[:,1]
Z= pc_data[:,2]
ity= pc_data[:,3]
r=[]

for i in range(pc_data.shape[0]):
    r.append(np.sqrt(pow(pc_data[i,0],2)+pow(pc_data[i,1],2)+pow(pc_data[i,2],2)))

n_bins =50
fig, axs = plt.subplots(1, 5, sharey=True, tight_layout=True)

axs[0].hist(X,bins=n_bins )
axs[0].set_title('X-axis')
axs[1].hist(Y,bins=n_bins)
axs[1].set_title('Y-axis')
axs[2].hist(Z,bins=n_bins)
axs[2].set_title('Z-axis')
axs[3].hist(ity,bins=n_bins)
axs[3].set_title('intensity')
axs[4].hist(r,bins=n_bins)
axs[4].set_title('depth')

plt.show()

#
#
# coors_range=np.array([-51.2, -51.2, -4, 51.2, 51.2, 4])
# voxel_size = np.array([0.8,0.8,8])
# grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size  # [-51.2, -51.2, -4, 51.2, 51.2, 4] / [0.8,0.8,8?] -> [128,128,1]
# # np.round(grid_size)
#
# grid_size= np.round(grid_size,0).astype(np.int) #[128,128,1]
# coor_to_voxelidx = -np.ones(shape=(5,5,5))
# coor=[1,2,3]
# voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
# # plot
# n_bins = 30
# fig, axs = plt.subplots(3, 3, sharey=True, tight_layout=True)
#
# # axs[0][0].hist(X,bins=n_bins )
# # axs[0][0].set_title('X-axis')
# # axs[0][1].hist(Y,bins=n_bins)
# # axs[0][1].set_title('Y-axis')
# # axs[0][2].hist(Z,bins=n_bins)
# # axs[0][2].set_title('Z-axis')
# #
# # axs[1][0].hist(r, bins=n_bins)
# # axs[1][0].set_title('depth')
# # axs[1][1].hist(intensity, bins=n_bins)
# # axs[1][1].set_title('intensity')
# # axs[1][2].hist(r, bins=n_bins)
# # axs[1][2].set_title('depth')
#
# # axs[2][0].hist(num_points_per_voxel, bins=n_bins)
# # axs[2][0].set_title('num_points')
# # axs[2][1].hist(r, bins=n_bins)
# # axs[2][1].set_title('depth')
# # axs[2][2].hist(r, bins=n_bins)
# # axs[2][2].set_title('depth')
#
# # axs[0][0].hist(coors[:,0],bins=n_bins )
# # axs[0][0].set_title('pillar_X-axis')
# # axs[0][1].hist(coors[:,1],bins=n_bins)
# # axs[0][1].set_title('pillar_Y-axis')
# # axs[0][2].hist(coors[:,2],bins=n_bins)
# # axs[0][2].set_title('pillar_Z-axis')
# #
# # axs[1][0].hist(num_points_per_voxel, bins=n_bins)
# # axs[1][0].set_title('num_points_per_pillar')
# # axs[1][1].hist(intensity, bins=n_bins)
# # axs[1][1].set_title('intensity')
# # axs[1][2].hist(r, bins=n_bins)
# # axs[1][2].set_title('depth')
# #
# # axs[2][0].hist(num_points_per_voxel, bins=n_bins)
# # axs[2][0].set_title('num_points')
# # axs[2][1].hist(r, bins=n_bins)
# # axs[2][1].set_title('depth')
# # axs[2][2].hist(r, bins=n_bins)
# # axs[2][2].set_title('depth')
#
#
#
#
# plt.show()
