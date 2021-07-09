import numpy as np
import open3d as o3d

pcd_file="./000014.bin"

pcd=np.fromfile(pcd_file,dtype=np.float32).reshape(-1,4)[:100000,:3]
pcd = np.array([x for x in pcd if 0 < x[0] + 50 < 100 and 0 < x[1] + 50 < 100 and 0< x[2]+5 < 8])

pcd_vis = o3d.geometry.PointCloud()

pcd_vis.points=o3d.utility.Vector3dVector(pcd)
o3d.visualization.draw_geometries([pcd_vis])


