from PIL import Image
import numpy as np
import torch
from tqdm.auto import tqdm
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('creating base model...')
base_name = 'base1B' # use base300M or base1B for better results
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

print('creating upsample model...')
upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

print('downloading base checkpoint...')
base_model.load_state_dict(load_checkpoint(base_name, device))

print('downloading upsampler checkpoint...')
upsampler_model.load_state_dict(load_checkpoint('upsample', device))

sampler = PointCloudSampler(
    device=device,
    models=[base_model, upsampler_model],
    diffusions=[base_diffusion, upsampler_diffusion],
    num_points=[1024, 8192 - 1024],
    aux_channels=['R', 'G', 'B'],
    guidance_scale=[3.0, 3.0],
)

# Load an image to condition on.
img = Image.open('example_data/picture/oneyingtao.jpg')

# Produce a sample from the model.
samples = None
for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
    samples = x


pc = sampler.output_to_point_clouds(samples)[0]
fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))


coords=np.array(pc.coords)
colors=np.array(pc.channels)
np.savez('example_data/zhuanhuan_point_cloud.npz', coords=coords, R=pc.channels['R'], G=pc.channels['G'], B=pc.channels['B'])

# npz_to_obj
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
npz_file_path = 'example_data/zhuanhuan_point_cloud.npz'
ply_file_path = '../../ppsurf-main/datasets/abc_minimal/04_pts_vis/oneyingtao1B.ply'
npz_to_ply_manual(npz_file_path,ply_file_path)