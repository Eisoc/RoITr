import os
import json
import numpy as np
import pickle
import argparse

# 定义数据存储结构
def create_data_dict():
    return {
        "src": [],
        "tgt": [],
        "rot": [],
        "trans": [],
        "overlap": []
    }

# 计算逆矩阵
def inverse_matrix(matrix):
    return np.linalg.inv(matrix)

# 处理每个文件夹的数据
def process_folder(folder_path):
    json_file = None
    pcd_files = []
    
    # 获取文件夹中的 JSON 文件和 PCD 文件
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".json"):
            json_file = filename
        elif filename.endswith(".pcd"):
            pcd_files.append(filename)

    # 如果 JSON 或 PCD 文件不符合要求，报错
    if not json_file or len(pcd_files) < 2:
        print(f"no pose or < 2 pcd files! Check the folder {folder_path} ")
        return None

    # 读取 JSON 文件
    with open(os.path.join(folder_path, json_file), 'r') as f:
        poses = json.load(f)

    # 初始化数据结构
    data_dict = create_data_dict()
    folder_name = os.path.basename(folder_path)
    
    # 计算每对点云的变换信息
    for i in range(len(pcd_files) - 1):
        src_pcd = f"{folder_name}/{pcd_files[i]}"
        tgt_pcd = f"{folder_name}/{pcd_files[i + 1]}"
        
        data_dict["src"].append(src_pcd)
        data_dict["tgt"].append(tgt_pcd)
        
        pose_1 = np.array(poses[i]["pose"]).reshape(4, 4)
        pose_2 = np.array(poses[i + 1]["pose"]).reshape(4, 4)
        
        # 计算旋转矩阵和平移向量
        relative_transform = inverse_matrix(pose_2) @ pose_1
        rot_matrix = relative_transform[:3, :3]
        trans_vector = relative_transform[:3, 3].reshape(3, 1)
        
        data_dict["rot"].append(rot_matrix)
        data_dict["trans"].append(trans_vector)
        data_dict["overlap"].append(0)

    return data_dict


def main():
    base_path = './data/hx'
    all_folders = sorted([os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

    train_dict = create_data_dict()
    val_dict = create_data_dict()
    test_dict = create_data_dict()

    # 划分训练集、验证集和测试集（80%, 9%, 11%）
    num_folders = len(all_folders)
    train_end = int(0.8 * num_folders)
    val_end = int(0.89 * num_folders)

    for i, folder in enumerate(all_folders):
        data_dict = process_folder(folder)
        if data_dict is None:
            continue

        if i <= train_end:
            for key in train_dict:
                train_dict[key].extend(data_dict[key])
        elif i <= val_end:
            for key in val_dict:
                val_dict[key].extend(data_dict[key])
        else:
            for key in test_dict:
                test_dict[key].extend(data_dict[key])

    # PKL
    PKL_folder = f"./configs/tdmatch"
    with open(f'{PKL_folder}/train_hx.pkl', 'wb') as f:
        pickle.dump(train_dict, f)
    with open(f'{PKL_folder}/val_hx.pkl', 'wb') as f:
        pickle.dump(val_dict, f)
    with open(f'{PKL_folder}/test_hx.pkl', 'wb') as f:
        pickle.dump(test_dict, f)

    print("PKL generated: train.pkl, val.pkl, test.pkl")

if __name__ == "__main__":
    main()