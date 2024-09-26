import open3d as o3d

# 定义 .pcd 文件路径
pcd_file = "/home/rhino/RoITr/data/hx/test/DP9675_1712445522.624953.pcd"

# 读取 .pcd 文件
pcd = o3d.io.read_point_cloud(pcd_file)
print("222")