import os
import random
import torch
import pickle
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, default='test')
args = parser.parse_args()

# 定义文件夹路径
data_dir = f"./data/hx/{args.name}"

# 获取所有 .pcd 文件并按文件名排序
pcd_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pcd')])

data_dict = {
    "src": [],
    "tgt": [],
    "rot": [],
    "trans": [],
    "overlap": []
}

# 遍历文件，交替添加到 src 和 tgt 中
for i, pcd_file in enumerate(pcd_files):
    file_path = f"{args.name}/{pcd_file}"
    if i % 2 == 0:
        data_dict["src"].append(file_path)
    else:
        data_dict["tgt"].append(file_path)

num_src = len(data_dict["src"])

if args.name == "dummy":
    for _ in range(num_src):
        data_dict["rot"].append(np.zeros((3, 3)))
        data_dict["trans"].append(np.zeros((3, 1)))
        data_dict["overlap"].append(0)
else:
    pose_txt = "data/hx/pose/enu_pose.txt"
    with open(pose_txt, 'r') as file:
        for line in file:

            elements = line.split()
            transform_values = list(map(float, elements[2:]))  # 提取后12个元素

            rot = np.array(transform_values[:9]).reshape(3, 3)
            trans = np.array(transform_values[9:]).reshape(3, 1)

            data_dict["rot"].append(rot)
            data_dict["trans"].append(trans)
            data_dict["overlap"].append(0)

# 将结果保存为 .pkl 文件
output_file = f"./configs/tdmatch/{args.name}.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(data_dict, f)

print(f"结果已保存为 {output_file}")
