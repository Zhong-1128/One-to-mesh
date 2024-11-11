# One-to-mesh
Official Pytorch implementation for the paper titled "Single-Image to High-Quality 3D Mesh: A Deep Learning-based Automatic Modeling Approach"

# Abstract
Automated 3D modeling from a single image holds significant potential in reducing modeling time and labor costs. However, existing methods often struggle with finegrained local features and consume substantial memory. In this paper, we propose a novel framework called One-to-mesh that efficiently generates high-quality 3D mesh models from single images. Our approach fine-tunes a pre-trained model to represent a single image as a 3D point cloud, which is then processed by a parallel encoder to extract both global and local features. The global encoder utilizes an interpolation function to capture coarse global features, and the local encoder utilizes a joint operation of propagation and aggregation based on self-attention to capture detailed local features. These features are decoded into an occupied field, which is subsequently meshed using the Marching Cubes algorithm. Experiments demonstrate that our method can rapidly generate accurate 3D mesh models on a single GPU, outperforming state-of-the-art single-image generation methods in terms of accuracy and memory efficiency. This work paves the way for broader applications of automated 3D modeling in fields such as automatic driving, virtual reality, and smart cities.

# Envirement:
   sys.platform: linux
   IDE: PyCharm 2022.3.3 Professional Edition.
   Python: 3.10.14
   pytorch: 2.1.0
   cuda: 11.7
   trimesh: 3.23
