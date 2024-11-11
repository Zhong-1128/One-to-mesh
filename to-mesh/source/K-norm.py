import torch
# import torch.nn as nn
from knn_cuda import KNN
import pytorch_lightning as pl
from torch.nn import functional as f

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

# 示例：假设输入数据 lc_xyz 和 lc_x
lc_xyz = torch.randn(8, 128, 3)  # (batch_size, num_groups, coord_dim)
lc_x = torch.randn(8, 128, 64)   # (batch_size, num_groups, feature_dim)

# 确保 CUDA 可用
if torch.cuda.is_available():
    lc_xyz = lc_xyz.cuda()  # 将 lc_xyz 放置到 GPU
    lc_x = lc_x.cuda()  # 将 lc_x 放置到 GPU

# 初始化 K_Norm_Pool 模块
model = K_Norm(out_dim=64, k_group_size=16, alpha=1.0, beta=0.0)

# 将模型也放置到 GPU
model = model.cuda()

# 前向传播
output = model(lc_xyz, lc_x)
print(output.shape)  # 输出的形状 (B, 2C, G)


# 使用示例
# if __name__ == '__main__':
#     block = K_Norm(kernel_size=3)  # 实例化K_Norm模块，指定核大小为3
#     input = torch.rand(1, 64, 64, 64)  # 生成一个随机输入
#     output = block(input)  # 将输入通过K_Norm模块处理
#     print(block)  # 打印输入和输出的尺寸，验证模块的作用