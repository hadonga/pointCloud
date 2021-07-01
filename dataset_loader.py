
import os
import numpy as np
from torch.utils.data import Dataset


# def pcd_normalize(pcd):
#     pcd = pcd.copy()
#     pcd[:, 0] = pcd[:, 0] / 70
#     pcd[:, 1] = pcd[:, 1] / 70
#     pcd[:, 2] = pcd[:, 2] / 3
#     pcd = np.clip(pcd, -1, 1)
#     return pcd
#
#
# def pcd_unnormalize(pcd):
#     pcd = pcd.copy()
#     pcd[:, 0] = pcd[:, 0] * 70
#     pcd[:, 1] = pcd[:, 1] * 70
#     pcd[:, 2] = pcd[:, 2] * 3
#     return pcd
# part_length = {'00': 4540, '01': 1100, '02': 4660, '03': 800, '04': 270, '05': 2760,
#                '06': 1100, '07': 1100, '08': 4070, '09': 1590, '10': 1200}

class kitti_loader(Dataset):
    def __init__(self, data_dir='/root/dataset/kitti/sequences/',
                 train=True, skip_frames=1, npoints=100000):
        self.train = train
        self.npoints = npoints
        self.pointcloud_path = []
        self.label_path = []
        if self.train:
            seq = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
            for seq_num in seq:
                folder_pc = os.path.join(data_dir, seq_num, 'velodyne')
                folder_lb = os.path.join(data_dir, seq_num, 'labels')

                file_pc = os.listdir(folder_pc)
                file_pc.sort(key=lambda x: str(x[:-4]))
                file_lb = os.listdir(folder_lb)
                file_lb.sort(key=lambda x: str(x[:-4]))

                for index in range(0, len(file_pc), skip_frames):
                    self.pointcloud_path.append('%s/%s' % (folder_pc, file_pc[index]))
                    self.label_path.append('%s/%s' % (folder_lb, file_lb[index]))
        else:
            seq = '08'
            folder_pc = os.path.join(data_dir, seq, 'velodyne')
            folder_lb = os.path.join(data_dir, seq, 'labels')
            file_pc = os.listdir(folder_pc)
            file_pc.sort(key=lambda x: str(x[:-4]))
            file_lb = os.listdir(folder_lb)
            file_lb.sort(key=lambda x: str(x[:-4]))
            for index in range(0, len(file_pc), skip_frames):
                self.pointcloud_path.append('%s/%s' % (folder_pc, file_pc[index]))
                self.label_path.append('%s/%s' % (folder_lb, file_lb[index]))

    def get_data(self, pointcloud_path, label_path):
        # points = np.load(pointcloud_path) # for npy files
        points = np.fromfile(pointcloud_path, dtype=np.float32).reshape(-1, 4)
        labels = np.fromfile(label_path, dtype=np.float32)
        return points, labels

    def mySqrt(x):
        if x == 0:
            return 0
        cur = 1
        while True:
            pre = cur
            cur = (cur + x / cur) / 2
            if abs(cur - pre) < 1e-6:
                return int(cur)

    def __len__(self):
        return len(self.pointcloud_path)

    def __getitem__(self, index):
        point, label = self.get_data(self.pointcloud_path[index], self.label_path[index])
        # square
        point = np.array([x for x in point if 0 < x[0] + 51.2 < 102.4 and 0 < x[1] + 51.2 < 102.4 and 0< x[2]+5 < 8])
        # round
        # point = np.array([x for x in point if self.mySqrt((np.power(x[0], 2) + np.power(x[1], 2))) < 51.2 and 0< x[2]+5 < 8])

        if len(point) >= self.npoints:
            choice = np.random.choice(len(point), self.npoints, replace=False)
        else:
            choice = np.random.choice(len(point), self.npoints, replace=True)
        point = point[choice]
        label = label[choice]
        return point, label
