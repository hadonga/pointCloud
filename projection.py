# %%

import numpy as np
import os
import sys
# sys.path.append("..") # Adds higher directory to python modules path.

import open3d as o3d
import struct
import math
import matplotlib.pyplot as plt
from dataset_loader import kitti_loader
from torch.utils.data import DataLoader
import torch

# %%


def projection_points(x,y,z,i):
    H = 64  # No. of lasers
    W = 1024  # best image size
    fup = 2  # in deg
    fdown = -24.8

    # range of Point from Lidar
    r = np.sqrt((pow(x, 2)) + (pow(y, 2)) + (pow(z, 2)))

    fup = (fup / 180) * math.pi
    fdown = (fdown / 180) * math.pi
    ft = abs(fup) + abs(fdown)

    u = np.floor(0.5 * W * (1 - ((np.arctan2(y, x)) / math.pi)))
    v = np.floor(H * (1 - ((np.arcsin(z / r)) + abs(fdown)) / ft))
    # v=np.floor(W*(1-(((np.arcsin(z/r)+abs(fup))/ft)))

    for xx in range(u.shape[0]):
        u[xx] = min(W - 1, u[xx])
        u[xx] = max(0, u[xx])

    for yy in range(v.shape[0]):
        v[yy] = min(H - 1, v[yy])
        v[yy] = max(0, v[yy])

    spherical_img = np.zeros((H, W))
    # print(spherical_img.shape)

    for zz in range(len(i)):
        spherical_img[int(v[zz]), int(u[zz])] += i[zz]

    return spherical_img

# dataset = kitti_loader(data_dir=cfg.root_dir, point_cloud_files=cfg.point_cloud_files,
#                        data_type=args.dataset_type, labels_files=cfg.labels_files,
#                        train=True, skip_frames=1)
# dataloader = DataLoader(dataset, batch_size=cfg.batch_size * cfg.num_gpus, shuffle=True,
#                         num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
# test_dataset = kitti_loader(data_dir=cfg.root_dir, point_cloud_files=cfg.point_cloud_files,
#                             data_type=args.dataset_type, labels_files=cfg.labels_files,
#                             train=False)
# test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size * cfg.num_gpus, shuffle=False,
#                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
# dt = np.dtype(['x','y','z','i'])
dataset = kitti_loader()
# scan,_ =dataset.__getitem__(index=1)
dataloader = DataLoader(dataset,batch_size=8, shuffle= True, num_workers= 4,
                        pin_memory= True, drop_last=True)
for batch_idx, (scan, labels) in enumerate(dataloader):
    print(scan.shape)
    print(labels.shape)

    # convert it into x,y,z coordinates and i
    x = scan[0,:, 0]  # get x
    y = scan[0,:, 1]  # get x
    z = scan[0,:, 2]  # get x
    i = scan[0,:, 3]  # get intensity

    projection_points(x, y, z, i)



#
# # %%
#
# print(max(x))
# print(min(x))
# print(max(y))
# print(min(y))
# print(max(z))
# print(min(z))
# print(max(i))
# print(min(i))
#
#
# # %%
#
# def convert_kitti_bin_to_pcd(binFilePath):  # fork from HTLife/convert_kitti_bin_to_pcd.py
#     size_float = 4
#     list_pcd = []
#     with open(binFilePath, "rb") as f:
#         byte = f.read(size_float * 4)
#         while byte:
#             x, y, z, intensity = struct.unpack("ffff", byte)
#             list_pcd.append([x, y, z])
#             byte = f.read(size_float * 4)
#     np_pcd = np.asarray(list_pcd)
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(np_pcd)
#     return pcd
#
#
# # %%
#
# pcd = convert_kitti_bin_to_pcd(filename)
# print('coverting!')
# o3d.io.write_point_cloud('pcloud.pcd', pcd)
#
# # %%
#
# # cloud = o3d.io.read_point_cloud('pcloud.pcd')
# # o3d.visualization.draw_geometries([pcd],
# #                                   zoom=0.1,
# #                                   # front=[0.4257, -0.2125, -0.8795],
# #                                   lookat=[1, 1, 1],
# #                                   front=[0.5, 0.5, 0.5],
# #                                   up=[-1, -1, -1])
#
# # %%
#
# # def projection_points(fup,fdown,W,L):
# H = 64  # No. of lasers
# W = 1024  # best image size
# fup = 2  # in deg
# fdown = -24.8
#
# # range of Point from Lidar
# r = np.sqrt((pow(x, 2)) + (pow(y, 2)) + (pow(z, 2)))
#
# fup = (fup / 180) * math.pi
# fdown = (fdown / 180) * math.pi
# ft = abs(fup) + abs(fdown)
#
# u = np.floor(0.5 * W * (1 - ((np.arctan2(y, x)) / math.pi)))
# v = np.floor(H * (1 - ((np.arcsin(z / r)) + abs(fdown)) / ft))
# # v=np.floor(W*(1-(((np.arcsin(z/r)+abs(fup))/ft)))
#
# for xx in range(u.shape[0]):
#     u[xx] = min(W - 1, u[xx])
#     u[xx] = max(0, u[xx])
#
# for yy in range(v.shape[0]):
#     v[yy] = min(H - 1, v[yy])
#     v[yy] = max(0, v[yy])
#
# # %%
#
# print(min(u))
# print(max(u))
# print(min(v))
# print(max(v))
#
# # %%
#
# spherical_img = np.zeros((H, W))
# print(spherical_img.shape)
#
# for zz in range(len(i)):
#     spherical_img[int(v[zz]), int(u[zz])] += i[zz]
#
# # %%
#
# import matplotlib.cm as cm
#
# plt.figure(figsize=(50, 10))
# plt.imshow(spherical_img, cmap=cm.RdYlGn, )
# plt.show()
