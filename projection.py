# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 留出前几个GPU跑其他程序, 需要在导入模型前定义


import open3d as o3d
import struct
import math
import matplotlib.pyplot as plt
from dataset_loader import kitti_loader
from torch.utils.data import DataLoader
import torch
from network import U_Net
import numpy as np
import torch

import torch.nn as nn




def projection_points(x,y,z,i,label):
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

    spherical_img_int = np.zeros((H, W))
    spherical_img_lab = np.zeros((H, W))
    # print(spherical_img.shape)

    for zz in range(len(i)):
        spherical_img_int[int(v[zz]), int(u[zz])] += i[zz]
        spherical_img_lab[int(v[zz]), int(u[zz])] = label[zz]

    return spherical_img_int,spherical_img_lab

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
dataloader = DataLoader(dataset,batch_size=1, shuffle= True, num_workers= 4,
                        pin_memory= True, drop_last=True)

model= U_Net()
print("Model has {} paramerters in total".format(sum(x.numel() for x in model.parameters())))

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.cuda()
# loss_crs = ClsLoss(ignore_index=0, reduction='mean')
loss_crs = nn.CrossEntropyLoss(ignore_index=0, reduction='mean').cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.35, patience=5, verbose=True,
                                                       threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                       eps=1e-08)
def train():
    for batch_idx, (scan, labels) in enumerate(dataloader):
        # convert it into x,y,z coordinates and i
        x = scan[0,:, 0]  # get x
        y = scan[0,:, 1]  # get x
        z = scan[0,:, 2]  # get x
        i = scan[0,:, 3]  # get intensity
        label= labels[0,:]
        for i in range(len(label)):
            if label[i] == "10":
                label[i] = 1
            else:
                label[i] = 0
        img,lab = projection_points(x, y, z, i, label)
    optimizer.zero_grad()
    out=model(img)
    loss=loss_crs(out,lab)

    loss.backward()
    optimizer.step()


def main():
    for epoch in range(50):
        train()

if __name__ == '__main__':
    main()


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
