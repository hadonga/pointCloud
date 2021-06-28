
import os
import numpy as np
from torch.utils.data import Dataset


def pcd_normalize(pcd):
    pcd = pcd.copy()
    pcd[:, 0] = pcd[:, 0] / 70
    pcd[:, 1] = pcd[:, 1] / 70
    pcd[:, 2] = pcd[:, 2] / 3
    pcd = np.clip(pcd, -1, 1)
    return pcd


def pcd_unnormalize(pcd):
    pcd = pcd.copy()
    pcd[:, 0] = pcd[:, 0] * 70
    pcd[:, 1] = pcd[:, 1] * 70
    pcd[:, 2] = pcd[:, 2] * 3
    return pcd


class kitti_loader(Dataset):
    def __init__(self, data_dir='/root/dataset/kitti/sequences/',
                 train=True, skip_frames=1, npoints=100000):
        self.train = train
        self.npoints = npoints
        # part_length = {'00': 4540, '01': 1100, '02': 4660, '03': 800, '04': 270, '05': 2760,
        #                '06': 1100, '07': 1100, '08': 4070, '09': 1590, '10': 1200}
        self.pointcloud_path = []
        self.label_path = []
        if self.train:
            seq = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
            for seq_num in seq:
                folder_pc = os.listdir(os.path.join(data_dir, seq_num, 'velodyne'))
                folder_pc.sort(key=lambda x: str(x[:-4]))
                folder_lb = os.listdir(os.path.join(data_dir, seq_num, 'labels'))
                folder_lb.sort(key=lambda x: str(x[:-4]))
                for index in range(0, len(folder_lb), skip_frames):
                    self.pointcloud_path.append('%s/%s/%s/%s' % (data_dir, seq_num, 'velodyne', folder_pc[index]))
                    self.label_path.append('%s/%s/%s/%s' % (data_dir, seq_num, 'labels', folder_lb[index]))
        # else:
        #     seq = ['08']
        #     for seq_num in seq:
        #         length_d = os.listdir(os.path.join(data_dir, seq_num, point_cloud_files))
        #         length_d.sort(key=lambda x: str(x[:-4]))
        #         length_l = os.listdir(os.path.join(data_dir, seq_num, labels_files))
        #         length_l.sort(key=lambda x: str(x[:-4]))
        #         for index in range(0, len(length_l), skip_frames):
        #             self.pointcloud_path.append('%s/%s/%s/%s' % (data_dir, seq_num, point_cloud_files, length_d[index]))
        #             self.label_path.append('%s/%s/%s/%s' % (data_dir, seq_num, labels_files, length_l[index]))

    def get_data(self, pointcloud_path, label_path):
        # points = np.load(pointcloud_path)
        points = np.fromfile(pointcloud_path, dtype=np.float32).reshape(-1, 4)
        label_map = np.fromfile(label_path, dtype=np.float32)
        return points, label_map

    def __len__(self):
        return len(self.pointcloud_path)

    def __getitem__(self, index):
        cloud, label = self.get_data(self.pointcloud_path[index], self.label_path[index])
        cloud = np.array([x for x in cloud if 0 < x[0] + 51.2 < 102.4 and 0 < x[1] + 51.2 < 102.4])
        if len(cloud) >= self.npoints:
            choice = np.random.choice(len(cloud), self.npoints, replace=False)

        else:
            choice = np.random.choice(len(cloud), self.npoints, replace=True)
        cloud = cloud[choice]
        label = label[choice]
        return cloud, label
