import torch
from torch import nn
from torch.nn import functional as F


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]

    Returns:
        [type]: [description]
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


class SPN(nn.Module):  # SimplePointNet: local feature extractor
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, inputs):
        x = self.linear(inputs)
        x = self.bn(x)
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]  # Maxpooling
        return x_max


class LPN(nn.Module):  # To test?! LightPointNet: local feature extractor from paper LightPointNet
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size= 4 * 1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=1)
        self.linear = nn.Linear(128, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        x_max = torch.max_pool1d(x)
        return x_max


class Conv1x1(nn.Module):  # spatial feature extractor
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)
        return out


class FeatureExtractor(nn.Module):
    def __init__(self,
                 num_input_features,
                 num_filters=(64,),
                 voxel_size=(0.8, 0.8, 8),
                 pc_range=(-50, -50, -3, 50, 50, 8)):
        super().__init__()
        assert len(num_filters) > 0
        num_input_features = 12  # modify then

        self.pfn_layers_1 = SPN(3, 64)
        self.pfn_layers_2 = SPN(1, 64)
        self.pfn_layers_3 = SPN(3, 64)
        self.pfn_layers_4 = SPN(5, 64)
        self.pfn_layers_0 = SPN(12, 64)
        # self.pfn_layers = nn.ModuleList(pfn_layers)
        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]  #
        self.y_offset = self.vy / 2 + pc_range[1]  # ?

    def forward(self, features, num_points, coors):  # input:
        # pc->  [voxels:nx100x7, num_points: n  , coors : nx4]
        # non-empty pillar number x 35 点 x 4 特征，c是voxel的坐标 voxel_num x3，D 是 pillar内的points数 - voxel_numx1
        # voxels from two batches are concatenated and coord have information corrd [num_voxels, (batch, x,y)]
        # pdb.set_trace()

        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_points.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean  # f_cluster: nx100x3

        # Find distance of x, y, and z from pillar center
        f_center = torch.zeros_like(features[:, :, :2])  # f_center: nx100x2  ; coors 前两列都是0
        f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].float().unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].float().unsqueeze(1) * self.vy + self.y_offset)

        # Combine together feature decorations
        features_pillar = [f_cluster, f_center]  # features: 5
        features_intensity = torch.unsqueeze(features[:, :, 3], 2)
        features_xyz = features[:, :, :3]
        features_norm = features[:, :, 4:]
        # print(features_norm.shape,features_xyz.shape,features_intensity.shape)

        features_pillar = torch.cat(features_pillar, dim=-1)
        org_feature = torch.cat([features, f_cluster, f_center], dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        # voxel_count = features.shape[1]
        # mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        # mask = torch.unsqueeze(mask, -1).type_as(features)
        # features *= mask

        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features_xyz *= mask
        features_norm *= mask
        features_pillar *= mask
        features_intensity *= mask
        org_feature *= mask

        # Forward pass through PFNLayers
        features_xyz = self.pfn_layers_1(
            features_xyz)  # here they are considering num of voxels as batch size for linear layer
        features_intensity = self.pfn_layers_2(features_intensity)
        features_norm = self.pfn_layers_3(features_norm)
        features_pillar = self.pfn_layers_4(features_pillar)
        org_feature = self.pfn_layers_0(org_feature)
        # features_res= torch.transpose(torch.cat([features_point,features_pillar],dim=2),2,0)
        # print(org_feature.shape, features_point.shape, features_pillar.shape)

        return features_xyz.squeeze(), features_intensity.squeeze(), \
               features_norm.squeeze(), features_pillar.squeeze(), org_feature.squeeze()
