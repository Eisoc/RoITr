import onnx
import onnxruntime as ort
import torch
import numpy as np
import os
import torch
from tqdm import tqdm
from lib.trainer import Trainer
from lib.utils import to_o3d_pcd
from visualizer.visualizer import Visualizer, create_visualizer
from visualizer.feature_space import visualize_feature_space
import open3d as o3d
import numpy as np
from dataset.common import normal_redirect

def generate_tgt_pcd(src_pcd, rotation_angle=None, translation=None, noise_sigma=0.01):
    """
    根据源点云 src_pcd 生成目标点云 tgt_pcd，通过旋转、平移和噪声。
    
    :param src_pcd: 源点云，形状 [N, 3] 的 numpy 数组
    :param rotation_angle: 旋转角度（弧度），如果为 None，则随机生成
    :param translation: 平移向量，形状 [3]，如果为 None，则随机生成
    :param noise_sigma: 噪声标准差，用于加入噪声，默认为 0.01
    :return: 生成的目标点云 tgt_pcd
    """
    N = src_pcd.shape[0]
    
    # 如果未提供旋转角度，则随机生成一个 0 到 2*pi 之间的角度
    if rotation_angle is None:
        rotation_angle = np.random.uniform(0, 2 * np.pi)
    
    # 生成绕 Z 轴的旋转矩阵
    R = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                  [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                  [0, 0, 1]])
    
    # 如果未提供平移向量，则随机生成一个
    if translation is None:
        translation = np.random.uniform(-0.5, 0.5, size=(3,))
    
    # 对源点云进行旋转和平移
    tgt_pcd = np.dot(src_pcd, R.T) + translation
    
    # 加入噪声
    noise = np.random.normal(scale=noise_sigma, size=tgt_pcd.shape)
    tgt_pcd += noise
    tgt_pcd = tgt_pcd.astype(np.float32)
    return tgt_pcd

num = 16649

src_pcd = torch.randn(num, 3).cpu().numpy()  # n=500，形状 [500, 3]
tgt_pcd = generate_tgt_pcd(src_pcd)
# tgt_pcd = torch.randn(num, 3).cpu().numpy()  # n=500，形状 [500, 3]
src_feats = np.ones(shape=(src_pcd.shape[0], 1)).astype(np.float32)
tgt_feats = np.ones(shape=(tgt_pcd.shape[0], 1)).astype(np.float32)

o3d_src_pcd = to_o3d_pcd(src_pcd)
o3d_tgt_pcd = to_o3d_pcd(tgt_pcd)
o3d_src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
src_normals = np.asarray(o3d_src_pcd.normals).astype(np.float32)
view_point = np.array([0., 0., 0.])
src_normals = normal_redirect(src_pcd, src_normals, view_point=view_point)
o3d_tgt_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
tgt_normals = np.asarray(o3d_tgt_pcd.normals)
tgt_normals = normal_redirect(tgt_pcd, tgt_normals, view_point=view_point).astype(np.float32)


# src_normals = torch.randn(num, 3).cpu().numpy()  # n=500，形状 [500, 3]
# tgt_normals = torch.randn(num, 3).cpu().numpy()  # n=500，形状 [500, 3]
rot = torch.randn(3, 3).cpu().numpy()  # 形状 [3, 3]
trans = torch.randn(3, 1).cpu().numpy()  # 形状 [3, 1]
src_raw_pcd = src_pcd

input_feed = {
    "src_pcd": src_pcd,
    "tgt_pcd": tgt_pcd,
    "src_feats": src_feats,
    "tgt_feats": tgt_feats,
    "src_normals": src_normals,
    "tgt_normals": tgt_normals,
    "rot": rot,
    "trans": trans,
    "src_raw_pcd": src_raw_pcd
}
model = onnx.load("RoITr16649.onnx")

# 检查模型输入形状
for input in model.graph.input:
    input_name = input.name
    input_shape = []
    for dim in input.type.tensor_type.shape.dim:
        if dim.dim_param:
            input_shape.append(dim.dim_param)
        else:
            input_shape.append(dim.dim_value)
    print(f"Input name: {input_name}, shape: {input_shape}")

# 检查模型输出形状
for output in model.graph.output:
    output_name = output.name
    output_shape = []
    for dim in output.type.tensor_type.shape.dim:
        if dim.dim_param:
            output_shape.append(dim.dim_param)
        else:
            output_shape.append(dim.dim_value)
    print(f"Output name: {output_name}, shape: {output_shape}")

# 加载 ONNX 模型
ort_session = ort.InferenceSession("RoITr16649.onnx")

# 推理
outputs = ort_session.run(
    None, 
    input_feed
)
print("done")