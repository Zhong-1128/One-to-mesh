import torch
# import torch.nn as nn
from knn_cuda import KNN
import pytorch_lightning as pl
from torch.nn import functional as f

#PointNetfeat相关
class AttentionPoco(pl.LightningModule):
    # self-attention for feature vectors
    # adapted from POCO attention
    # https://github.com/valeoai/POCO/blob/4e39b5e722c82e91570df5f688e2c6e4870ffe65/networks/decoder/interp_attention.py

    def __init__(self, net_size_max=1024, reduce=True):
        super(AttentionPoco, self).__init__()

        self.fc_query = torch.nn.Conv2d(net_size_max, 1, 1)
        self.fc_value = torch.nn.Conv2d(net_size_max, net_size_max, 1)
        self.reduce = reduce

    def forward(self, feature_vectors: torch.Tensor):
        # [feat_len, batch, num_feat] expected -> feature dim to dim 0
        feature_vectors_t = torch.permute(feature_vectors, (1, 0, 2))

        query = self.fc_query(feature_vectors_t).squeeze(0)  # fc over feature dim -> [batch, num_feat]
        value = self.fc_value(feature_vectors_t).permute(1, 2, 0)  # -> [batch, num_feat, feat_len]

        weights = torch.nn.functional.softmax(query, dim=-1)  # softmax over num_feat -> [batch, num_feat]
        if self.reduce:
            feature_vector_out = torch.sum(value * weights.unsqueeze(-1).broadcast_to(value.shape), dim=1)
        else:
            feature_vector_out = (weights.unsqueeze(2) * value).permute(0, 2, 1)
        return feature_vector_out
class STN(pl.LightningModule):
    def __init__(self, net_size_max=1024, num_scales=1, num_points=500, dim=3, sym_op='max'):
        super(STN, self).__init__()

        self.net_size_max = net_size_max
        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.net_size_max, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)

        self.fc1 = torch.nn.Linear(self.net_size_max, int(self.net_size_max / 2))
        self.fc2 = torch.nn.Linear(int(self.net_size_max / 2), int(self.net_size_max / 4))
        self.fc3 = torch.nn.Linear(int(self.net_size_max / 4), self.dim*self.dim)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(self.net_size_max)
        self.bn4 = torch.nn.BatchNorm1d(int(self.net_size_max / 2))
        self.bn5 = torch.nn.BatchNorm1d(int(self.net_size_max / 4))

        if self.num_scales > 1:
            self.fc0 = torch.nn.Linear(self.net_size_max * self.num_scales, self.net_size_max)
            self.bn0 = torch.nn.BatchNorm1d(self.net_size_max)

    def forward(self, x):
        batch_size = x.size()[0]
        x = f.relu(self.bn1(self.conv1(x)))
        x = f.relu(self.bn2(self.conv2(x)))
        x = f.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            x_scales = x.new_empty(x.size(0), self.net_size_max * self.num_scales, 1)
            for s in range(self.num_scales):
                x_scales[:, s*self.net_size_max:(s+1)*self.net_size_max, :] = \
                    self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, self.net_size_max*self.num_scales)

        if self.num_scales > 1:
            x = f.relu(self.bn0(self.fc0(x)))

        x = f.relu(self.bn4(self.fc1(x)))
        x = f.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.dim, dtype=x.dtype, device=x.device).view(1, self.dim*self.dim).repeat(batch_size, 1)
        x = x + iden
        x = x.view(-1, self.dim, self.dim)
        return x
class QSTN(pl.LightningModule):
    def __init__(self, net_size_max=1024, num_scales=1, num_points=500, dim=3, sym_op='max'):
        super(QSTN, self).__init__()

        self.net_size_max = net_size_max
        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.net_size_max, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = torch.nn.Linear(self.net_size_max, int(self.net_size_max / 2))
        self.fc2 = torch.nn.Linear(int(self.net_size_max / 2), int(self.net_size_max / 4))
        self.fc3 = torch.nn.Linear(int(self.net_size_max / 4), 4)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(self.net_size_max)
        self.bn4 = torch.nn.BatchNorm1d(int(self.net_size_max / 2))
        self.bn5 = torch.nn.BatchNorm1d(int(self.net_size_max / 4))

        if self.num_scales > 1:
            self.fc0 = torch.nn.Linear(self.net_size_max*self.num_scales, self.net_size_max)
            self.bn0 = torch.nn.BatchNorm1d(self.net_size_max)

    def forward(self, x):
        x = f.relu(self.bn1(self.conv1(x)))
        x = f.relu(self.bn2(self.conv2(x)))
        x = f.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            x_scales = x.new_empty(x.size(0), self.net_size_max*self.num_scales, 1)
            for s in range(self.num_scales):
                x_scales[:, s*self.net_size_max:(s+1)*self.net_size_max, :] = \
                    self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, self.net_size_max*self.num_scales)

        if self.num_scales > 1:
            x = f.relu(self.bn0(self.fc0(x)))

        x = f.relu(self.bn4(self.fc1(x)))
        x = f.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # add identity quaternion (so the network can output 0 to leave the point cloud identical)
        iden = x.new_tensor([1, 0, 0, 0])
        x_quat = x + iden

        # convert quaternion to rotation matrix
        x = batch_quat_to_rotmat(x_quat)

        return x, x_quat

#K_Norm相关
class GroupFeature(torch.nn.Module):  # FPS + KNN
    def __init__(self, group_size):
        super().__init__()
        self.group_size = group_size  # the first is the point itself
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, feat):
        '''
            input:
                xyz: B N 3
                feat: B N C
            ---------------------------
            output:
                neighborhood: B N K 3
                feature: B N K C
        '''
        batch_size, num_points, _ = xyz.shape  # B N 3 : 1 128 3
        C = feat.shape[-1]

        center = xyz
        # knn to get the neighborhood
        _, idx = self.knn(xyz, xyz)  # B N K : get K idx for every center
        assert idx.size(1) == num_points  # N center
        assert idx.size(2) == self.group_size  # K knn group
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]  # B N K 3
        neighborhood = neighborhood.view(batch_size, num_points, self.group_size, 3).contiguous()  # 1 128 8 3
        neighborhood_feat = feat.contiguous().view(-1, C)[idx, :]  # BxNxK C 128x8 384   128*26*8
        assert neighborhood_feat.shape[-1] == feat.shape[-1]
        neighborhood_feat = neighborhood_feat.view(batch_size, num_points, self.group_size,
                                                   feat.shape[-1]).contiguous()  # 1 128 8 384
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        return neighborhood, neighborhood_feat
# K_Norm
class K_Norm(torch.nn.Module):
    def __init__(self, out_dim, k_group_size, alpha, beta):
        super().__init__()
        self.group_feat = GroupFeature(k_group_size)
        self.affine_alpha_feat = torch.nn.Parameter(torch.ones([1, 1, 1, out_dim]))
        self.affine_beta_feat = torch.nn.Parameter(torch.zeros([1, 1, 1, out_dim]))

    def forward(self, lc_xyz, lc_x):
        # get knn xyz and feature
        knn_xyz, knn_x = self.group_feat(lc_xyz, lc_x)  # B G K 3, B G K C

        # Normalize x (features) and xyz (coordinates)
        mean_x = lc_x.unsqueeze(dim=-2)  # B G 1 C
        std_x = torch.std(knn_x - mean_x)

        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz)  # B G 1 3

        knn_x = (knn_x - mean_x) / (std_x + 1e-5)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)  # B G K 3

        B, G, K, C = knn_x.shape

        # Feature Expansion
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)  # B G K 2C

        # Affine
        knn_x = self.affine_alpha_feat * knn_x + self.affine_beta_feat

        # Geometry Extraction
        knn_x_w = knn_x.permute(0, 3, 1, 2)  # B 2C G K

        return knn_x_w

# Pooling
class K_Pool(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        e_x = torch.exp(knn_x_w) # B 2C G K
        up = (knn_x_w * e_x).mean(-1) # # B 2C G
        down = e_x.mean(-1)
        lc_x = torch.div(up, down)
        # lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1) # B 2C G K -> B 2C G
        return lc_x

#K_Norm  K_Pool  串联
class K_Norm_Pool(torch.nn.Module):
    def __init__(self, out_dim, k_group_size, alpha, beta):
        super(K_Norm_Pool, self).__init__()
        # 实例化 K_Norm 和 K_Pool 模块
        self.k_norm = K_Norm(out_dim, k_group_size, alpha, beta)
        self.k_pool = K_Pool()

    def forward(self, lc_xyz, lc_x):
        # 首先调用 K_Norm 模块
        knn_x_w = self.k_norm(lc_xyz, lc_x)
        # 然后调用 K_Pool 模块
        lc_x_pooled = self.k_pool(knn_x_w)
        return lc_x_pooled

# 示例：假设输入数据 lc_xyz 和 lc_x
# lc_xyz = torch.randn(8, 128, 3)  # (batch_size, num_groups, coord_dim)
# lc_x = torch.randn(8, 128, 64)   # (batch_size, num_groups, feature_dim)
#
# # 初始化 K_Norm_Pool 模块
# out_dim = 64
# k_group_size = 16
# alpha = 1.0
# beta = 0.0
# model = K_Norm_Pool(out_dim, k_group_size, alpha, beta)
#
# # 前向传播
# output = model(lc_xyz, lc_x)
# print(output.shape)  # 输出的形状 (B, 2C, G)


class PointNetfeat(pl.LightningModule):
    def __init__(self, net_size_max=1024, num_scales=1, num_points=500,
                 polar=False, use_point_stn=True, use_feat_stn=True,
                 output_size=100, sym_op='max', dim=3):
        super(PointNetfeat, self).__init__()

        self.net_size_max = net_size_max
        self.num_points = num_points
        self.num_scales = num_scales
        self.polar = polar
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.sym_op = sym_op
        self.output_size = output_size
        self.dim = dim

        if self.use_point_stn:
            self.stn1 = QSTN(net_size_max=net_size_max, num_scales=self.num_scales,
                             num_points=num_points, dim=dim, sym_op=self.sym_op)

        if self.use_feat_stn:
            self.stn2 = STN(net_size_max=net_size_max, num_scales=self.num_scales,
                            num_points=num_points, dim=64, sym_op=self.sym_op)

        self.conv0a = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv0b = torch.nn.Conv1d(64, 64, 1)
        self.bn0a = torch.nn.BatchNorm1d(64)
        self.bn0b = torch.nn.BatchNorm1d(64)
        self.conv1 = torch.nn.Conv1d(64, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, output_size, 1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(output_size)

        if self.num_scales > 1:
            self.conv4 = torch.nn.Conv1d(output_size, output_size*self.num_scales, 1)
            self.bn4 = torch.nn.BatchNorm1d(output_size*self.num_scales)


        #对称操作选择
        # max：选择每个点的最大特征值。
        # sum：求和操作，聚合所有点的特征。
        # wsum：加权求和操作，权重由pts_weights决定。
        # att：使用注意力机制进行聚合，可以更有效地捕捉重要的点特征
        if self.sym_op == 'max':
            self.mp1 = torch.nn.MaxPool1d(num_points)
        elif self.sym_op == 'sum':
            pass
        elif self.sym_op == 'wsum':
            pass
        elif self.sym_op == 'att':
            self.att = AttentionPoco(output_size)
        else:
            raise ValueError('Unsupported symmetric operation: {}'.format(self.sym_op))

    def forward(self, x, pts_weights):
        print("################这里")
        print(x.shape)
        # input transform
        if self.use_point_stn:
            trans, trans_quat = self.stn1(x[:, :3, :])  # transform only point data
            # an error here can mean that your input size is wrong (e.g. added normals in the point cloud files)
            x_transformed = torch.bmm(trans, x[:, :3, :])  # transform only point data
            x = torch.cat((x_transformed, x[:, 3:, :]), dim=1)
        else:
            trans = None
            trans_quat = None

        if bool(self.polar):
            x = torch.permute(x, (0, 2, 1))
            x = cartesian_to_polar(pts_cart=x)
            x = torch.permute(x, (0, 2, 1))

        # mlp (64,64)
        x = f.relu(self.bn0a(self.conv0a(x)))
        x = f.relu(self.bn0b(self.conv0b(x)))

        # feature transform
        if self.use_feat_stn:
            trans2 = self.stn2(x)
            x = torch.bmm(trans2, x)
        else:
            trans2 = None

        # mlp (64,128,output_size)
        x = f.relu(self.bn1(self.conv1(x)))
        x = f.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # mlp (output_size,output_size*num_scales)
        if self.num_scales > 1:
            x = self.bn4(self.conv4(f.relu(x)))

        # symmetric max operation over all points
        if self.num_scales == 1:
            if self.sym_op == 'max':
                x = self.mp1(x)
            elif self.sym_op == 'sum':
                x = torch.sum(x, 2, keepdim=True)
            elif self.sym_op == 'wsum':
                pts_weights_bc = torch.broadcast_to(torch.unsqueeze(pts_weights, 1), size=x.shape)
                x = x * pts_weights_bc
                x = torch.sum(x, 2, keepdim=True)
            elif self.sym_op == 'att':
                x = self.att(x)
            else:
                raise ValueError('Unsupported symmetric operation: {}'.format(self.sym_op))

        else:
            x_scales = x.new_empty(x.size(0), self.output_size*self.num_scales**2, 1)
            if self.sym_op == 'max':
                for s in range(self.num_scales):
                    x_scales[:, s*self.num_scales*self.output_size:(s+1)*self.num_scales*self.output_size, :] = \
                        self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            elif self.sym_op == 'sum':
                for s in range(self.num_scales):
                    x_scales[:, s*self.num_scales*self.output_size:(s+1)*self.num_scales*self.output_size, :] = \
                        torch.sum(x[:, :, s*self.num_points:(s+1)*self.num_points], 2, keepdim=True)
            else:
                raise ValueError('Unsupported symmetric operation: %s' % self.sym_op)
            x = x_scales

        x = x.view(-1, self.output_size * self.num_scales ** 2)

        return x, trans, trans_quat, trans2

if __name__ == '__main__':
    vit = PointNetfeat()
    print(vit)