import torch
import open3d as o3d
import numpy as np
import os

# data_path = "/home/bing/RoITr/data/indoor/train/7-scenes-chess"
# data_name = "cloud_bin_0"
# data = os.path.join(data_path, data_name + ".pth")
# data = torch.load(data)
# print(len(data))

# data_path2 = "/home/bing/RoITr/data/indoor/test/7-scenes-redkitchen"
# data_name2 = "cloud_bin_1"
# data2 = os.path.join(data_path2, data_name2 + ".pth")
# data2 = torch.load(data2)

result_path = "/home/bing/RoITr/snapshot/HX_ripoint_transformer_test/3DLoMatch"
result_name = "1"
result = os.path.join(result_path, result_name + ".pth")
result = torch.load(result)
result = {key: torch.tensor(value).cpu() if not isinstance(value, torch.Tensor) else value.cpu()
        for key, value in result.items()}
print(len(result))
src_points = np.array(result['src_pcd'], dtype=np.float64)
tgt_points = np.array(result['tgt_pcd'], dtype=np.float64)
rot = np.array(result['rot'], dtype=np.float64)
trans = np.array(result['trans'], dtype=np.float64) 

transformed_src_points = (rot @ src_points.T).T + trans.T

src_pcd = o3d.geometry.PointCloud()
tgt_pcd = o3d.geometry.PointCloud()

src_pcd.points = o3d.utility.Vector3dVector(transformed_src_points)
tgt_pcd.points = o3d.utility.Vector3dVector(tgt_points)

src_pcd.paint_uniform_color([1, 0, 0]) 
tgt_pcd.paint_uniform_color([0, 1, 0]) 

combined_pcd = src_pcd + tgt_pcd
o3d.io.write_point_cloud(f"{result_name}.pcd", combined_pcd)
print(f"点云已保存{result_name}")


# 加载 .pth 
# data_path = "/home/rhino/RoITr/data/indoor/test/7-scenes-redkitchen"
# data_name = "cloud_bin_0"
# data = os.path.join(data_path, data_name + ".pth")
# data = torch.load(data)
# print(len(data))

# data_path2 = "/home/rhino/RoITr/data/indoor/test/7-scenes-redkitchen"
# data_name2 = "cloud_bin_1"
# data2 = os.path.join(data_path2, data_name2 + ".pth")
# data2 = torch.load(data2)
# print(len(data2))


# # 创建 open3d 点云对象
# pcd = o3d.geometry.PointCloud()
# pcd3 = o3d.geometry.PointCloud()

# pcd.points = o3d.utility.Vector3dVector(data)
# pcd3.points = o3d.utility.Vector3dVector(data2)

# # 保存点云为 .pcd 文件
# o3d.io.write_point_cloud(f"{data_name}.pcd", pcd)
# o3d.io.write_point_cloud(f"{data_name2}.pcd", pcd3)
# print(f"点云已保存{data_name},{result_name,data_name2}")