import numpy as np
# import open3d as o3d

# def npz_to_obj(npz_file_path,obj_file_path):
#     #加载.npz文件
#     data = np.load(npz_file_path)
#
#     # 获取坐标数据
#     coords = data['coords']
#
#     # 检查coords是否是三维坐标
#     if coords.shape[1] != 3:
#         raise ValueError("坐标必须是三维（xyz）.")
#
#     # # 写入.obj文件
#     # with open(obj_file_path, 'w') as obj_file:
#     #     for point in coords:
#     #         obj_file.write(f"v {point[0]} {point[1]} {point[2]}\n")
#
#     print(f"Conversion complete: {obj_file_path} created.")
#
# #示例调用
# npz_file_path = 'example_data/output_point_cloud.npz'
# # ply_file_path = '../../ppsurf-main/datasets/abc_minimal/04_pts_vis/point_cloud.obj'
# ply_file_path = 'example_data/pointcloud_npz_to_ply.ply'
# npz_to_obj(npz_file_path,ply_file_path)


import numpy as np

def npz_to_ply_manual(npz_file_path, ply_file_path):
    # 加载 npz 文件
    data = np.load(npz_file_path)

    # 提取坐标数据
    coords = data['coords']  # 假设 'coords' 是 npz 文件中的键，包含 (N, 3) 的坐标数据

    # 确保 coords 是 (N, 3) 的形状
    assert coords.shape[1] == 3, "坐标数据应包含3列 (x, y, z)."

    # 写入 PLY 文件头
    header = f"""ply
format ascii 1.0
element vertex {coords.shape[0]}
property float x
property float y
property float z
end_header
"""

    # 打开目标文件写入头部和坐标数据
    with open(ply_file_path, 'w') as ply_file:
        ply_file.write(header)
        np.savetxt(ply_file, coords, fmt='%f %f %f')  # 写入数据

    print(f"成功将 {npz_file_path} 转换为 {ply_file_path}.")

#示例调用
npz_file_path = 'example_data/pc_corgi.npz'
# ply_file_path = '../../ppsurf-main/datasets/abc_minimal/04_pts_vis/point_cloud.obj'
ply_file_path = '../../ppsurf-main/datasets/abc_minimal/04_pts_vis/corgi.ply'
npz_to_ply_manual(npz_file_path,ply_file_path)
