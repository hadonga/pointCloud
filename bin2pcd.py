# covert bin files in Kitti to pcd files
# use open3d
import open3d as o3d
import os
import struct
import numpy as np

def convert_kitti_bin_to_pcd(binFilePath): # fork from HTLife/convert_kitti_bin_to_pcd.py
    size_float = 4
    list_pcd = []
    with open(binFilePath, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    return pcd

bin_path= '/velo_bin/' #set bin path
pcd_path='/velo_pcd/' #set pcd path
if not os.path.exists('/velo_pcd'):
    os.mkdir('/velo_pcd')

bin_files=os.listdir(bin_path)
for file in bin_files:
    pcd=convert_kitti_bin_to_pcd(os.path.join(bin_path,file))
    print(file,'is coverting!')
    o3d.io.write_point_cloud(os.path.join(pcd_path,file[:-4]+'.pcd'), pcd)
