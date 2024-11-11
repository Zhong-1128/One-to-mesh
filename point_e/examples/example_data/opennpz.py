# import numpy as np
# import open3d as o3d
#
# # Load the point cloud data from the .npy file
# point_cloud_data = np.load("p2s_02.npy")  # Replace with your file path
#
# # Check the structure of the point cloud data
# print("Shape of the point cloud:", point_cloud_data.shape)
# print("First 5 points:\n", point_cloud_data[:10])
#
# # Create an Open3D PointCloud object
# pcd = o3d.geometry.PointCloud()
#
# # Assign the points to the PointCloud object
# pcd.points = o3d.utility.Vector3dVector(point_cloud_data)
#
# # Visualize the point cloud
# o3d.visualization.draw_geometries([pcd], window_name='Point Cloud Visualization')


# import numpy as np
# import matplotlib.pyplot as plt
#
# #打开.npz文件
# file_path='pc_corgi.npz'
# file_path2='output_point_cloud.npz'
#
# data=np.load(file_path)
# data2=np.load(file_path2)
#
#
# print(data.files)  #['coords', 'R', 'G', 'B']
# print(data2.files) #['coords', 'R', 'G', 'B']
#
# print(data['coords'])
# print(data['R'])
# print(data['G'])
# # print(data.shape)
# a=data['R']
# b=data['coords']
# c=data['G']
# d=data['B']
# print(b.shape)
# print(a.shape)
# print(c.shape)
# print(d.shape)
#
# #可视化点云
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
#
# def load_and_print_npz(file_path):
#     # 加载 .npz 文件
#     with np.load(file_path, allow_pickle=True) as data:
#         # 打印所有键（即存储的数组名）
#         print("Available keys in the .npz file:")
#         for key in data.keys():
#             print(key)
#             array = data[key]
#             print(f"Shape of {key}: {array.shape}")
#
#         # 获取点云坐标和颜色数据
#         points = data['coords']  # 坐标点数据，形状为 (N, 3)
#         r = data['R']  # 红色通道，形状为 (N,)
#         g = data['G']  # 绿色通道，形状为 (N,)
#         b = data['B']  # 蓝色通道，形状为 (N,)
#
#         # 将 R, G, B 通道组合成颜色数组，确保形状为 (N, 3)
#         colors = np.stack((r, g, b), axis=-1)  # 形状为 (N, 3)
#
#         # 创建一个 3D 图形
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#
#         # 确保 points 的形状为 (N, 3) 和 colors 的形状为 (N, 3)
#         x, y, z = points[:, 0], points[:, 1], points[:, 2]
#
#         # 使用颜色数组绘制散点图
#         ax.scatter(x, y, z, c=colors, marker='o')  # 使用 (N, 3) 颜色数组
#
#         # 设置图形标题和坐标轴标签
#         ax.set_title('3D Scatter Plot of Points with RGB Colors')
#         ax.set_xlabel('X Axis')
#         ax.set_ylabel('Y Axis')
#         ax.set_zlabel('Z Axis')
#
#         # 显示图形
#         plt.show()
#
#
# #替换为你的 .npz 文件路径
# npz_file_path = 'output_point_cloud.npz'
# print(npz_file_path)
# load_and_print_npz(npz_file_path)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_and_print_npz(file_path):
    # 加载.npz文件
    with np.load(file_path, allow_pickle=True) as data:
        # 打印所有键（即存储的数组名）
        print("Available keys in the .npz file:")
        # temp=[]
        for key in data.keys():
            print(key)

            # 打印对应数组的形状
            array = data[key]
            # temp.append(array)

            print(f"Shape of {key}: {array.shape}")
            #print(array)
            # 如果需要，你还可以打印数组的数据（注意：这可能会输出大量数据）
            # print(array)
        points=data['coords']
        r=data['R']
        g=data['G']
        b=data['B']


        # 将 R, G, B 通道组合成颜色数组，范围归一化为 [0, 1]
        colors = np.stack((r, g, b), axis=-1)

        # 创建一个3D图形
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 绘制points数组中的点
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        ax.scatter(x, y, z, c=colors, marker='o')

        # 设置图形标题和坐标轴标签
        ax.set_title('3D Scatter Plot of Points')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')



        # 显示图形
        plt.show()

# 替换为你的.npz文件路径
npz_file_path = 'pc_cube_stack.npz'
# npz_file_path='output_point_cloud.npz'
print(npz_file_path)
print("66666666666666666")
load_and_print_npz(npz_file_path)

