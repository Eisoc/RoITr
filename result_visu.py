import open3d as o3d
import numpy as np
import os

def read_nth_matrix(file_path, n):
    """读取第 n 个 4x4 变换矩阵"""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 每个矩阵的结构为 6 行（标题+4行矩阵+1行空行）
    matrices_per_block = 6
    start_idx = n * matrices_per_block + 1 
    
    # 读取 4x4 矩阵
    matrix_lines = lines[start_idx:start_idx+4]
    matrix = np.array([[float(num) for num in line.split()] for line in matrix_lines])
    
    return matrix

def apply_transform(matrix, point_cloud):
    """对点云应用4x4的变换矩阵"""
    points = np.asarray(point_cloud.points)
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = np.dot(matrix, points_homogeneous.T).T
    point_cloud.points = o3d.utility.Vector3dVector(transformed_points[:, :3])

def load_pcd_files(pcd_dir):
    """加载并排序所有的 .pcd 文件"""
    pcd_files = sorted([f for f in os.listdir(pcd_dir) if f.endswith('.pcd')])
    return pcd_files

def process_pcd(pcd_dir, matrix_file, n):

    pcd_files = load_pcd_files(pcd_dir)

    src_file = pcd_files[2*n]  # 第 n 个 src
    tgt_file = pcd_files[2*n + 1]  # 第 n 个 tgt

    src_pcd = o3d.io.read_point_cloud(os.path.join(pcd_dir, src_file))
    tgt_pcd = o3d.io.read_point_cloud(os.path.join(pcd_dir, tgt_file))

    matrix = read_nth_matrix(matrix_file, n)
    apply_transform(matrix, src_pcd)

    src_pcd.paint_uniform_color([1, 0, 0])  # 红色
    tgt_pcd.paint_uniform_color([0, 1, 0])  # 绿色

    combined_pcd = src_pcd + tgt_pcd

    output_file = (f"./results_visu/combined_{n}.pcd")
    o3d.io.write_point_cloud(output_file, combined_pcd)
    print(f"第 {n} 个拼接后的点云保存为: {output_file}")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pts', type=int, nargs='+', help='Specify one or more pts. Index starts from 0')
args = parser.parse_args()

pcd_dir = "/home/rhino/RoITr/data/hx/test"
matrix_file = "/home/rhino/RoITr/est_traj/test/10000/transforms.txt" 
for n in args.pts:
    process_pcd(pcd_dir, matrix_file, n)

