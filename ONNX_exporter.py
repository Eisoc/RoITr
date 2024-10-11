import torch
import torch.onnx
import copy
import os, argparse, json, shutil
os.environ["CUDA_VISIBLE_DEVICES"] = " 1, 0, 2, 3" # for Pytorch DistributedDataParallel(DDP) training
import torch
from torch import optim
from torch.utils.data.distributed import DistributedSampler # for Pytorch DistbutedDataParallel(DDP) training
from lib.utils import setup_seed
from configs.utils import load_config
from easydict import EasyDict as edict
from dataset.dataloader import get_dataset, get_dataloader
from model.RIGA_v2 import create_model
from lib.loss import OverallLoss, Evaluator
from lib.tester import get_trainer


#########################################################
# load config
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument("--local_rank", type=int, default=-1) # for DDP training
args = parser.parse_args()
config = load_config(args.config)
config['local_rank'] = args.local_rank
#########################################################
#set cuda devices for both DDP training and single-GPU training
if config['local_rank'] > -1:
    torch.cuda.set_device(config['local_rank'])
    config['device'] = torch.device('cuda', config['local_rank'])
    torch.distributed.init_process_group(backend='nccl')

else:
    torch.cuda.set_device(0)
    config['device'] = torch.device('cuda', 0)

setup_seed(42) # fix the seed


config = edict(config)

# create model
#config.model = RIGA(config=config).to(config.device)
config.model = create_model(config).to(config.device)

class ModelWrapper(torch.nn.Module):
    def __init__(self, original_model):
        super(ModelWrapper, self).__init__()
        self.model = original_model

    def forward(self, *args, **kwargs):
        output_dict = self.model(*args, **kwargs)
        # 按照固定的顺序提取字典中的值
        outputs_values = []
        for key, value in output_dict.items():
            outputs_values.append(value)
        outputs_values = tuple(outputs_values)
        return outputs_values


# 加载权重文件（.pth）
state = torch.load('weights/model_3dmatch.pth')
config.model.load_state_dict({k.replace('module.', ''): v for k, v in state['state_dict'].items()})
config.model.eval()  # 切换到评估模式

n_input = 16649
n_input2 = 16649

src_pcd, tgt_pcd, src_feats, tgt_feats, src_normals, tgt_normals,rot, trans, src_raw_pcd = torch.randn(n_input, 3).cuda(), torch.randn(n_input2, 3).cuda(), torch.randn(n_input, 1).cuda(), torch.randn(n_input2, 1).cuda(), torch.randn(n_input, 3).cuda(), torch.randn(n_input2, 3).cuda(), torch.randn(3, 3).cuda(), torch.randn(3, 1).cuda(), torch.randn(n_input, 3).cuda()
dummy_input = src_pcd, tgt_pcd, src_feats, tgt_feats, src_normals, tgt_normals,rot, trans, src_raw_pcd  

# 注册未支持算子
torch.onnx.register_custom_op_symbolic("aten::lift_fresh", lambda g, x: x, 13)

output_names = ['src_points', 'tgt_points', 'src_nodes', 'tgt_nodes', 'src_point_feats', 'tgt_point_feats', 'src_node_feats', 'tgt_node_feats', 'gt_node_corr_indices', 'gt_node_corr_overlaps', 'gt_tgt_node_occ', 'gt_src_node_occ', 'src_node_corr_indices', 'tgt_node_corr_indices', 'src_node_corr_knn_points', 'tgt_node_corr_knn_points', 'src_node_corr_knn_masks', 'tgt_node_corr_knn_masks', 'matching_scores', 'tgt_corr_points', 'src_corr_points', 'corr_scores',"test_raw"]

dynamic_axes = {
    'src_pcd': {0: 'n'},
    'tgt_pcd': {0: 'n'},
    'src_feats': {0: 'n'},
    'tgt_feats': {0: 'n'},
    'src_normals': {0: 'n'},
    'tgt_normals': {0: 'n'},
    'src_raw_pcd': {0: 'n'},
}

for name in output_names:
    dynamic_axes[name] = {0: 'n'}

wrapped_model = ModelWrapper(config.model)

# 将模型导出为 ONNX 格式
torch.onnx.export(wrapped_model,                   # 要转换的模型
                  dummy_input,             
                  f"RoITr{n_input}.onnx",       # 输出文件名
                  export_params=True,      # 导出模型权重
                  opset_version=13,        
                  do_constant_folding=True,  # 常量折叠优化
                  input_names = ["src_pcd", "tgt_pcd", "src_feats", "tgt_feats", "src_normals", "tgt_normals","rot", "trans", "src_raw_pcd"], # 输入名
                  output_names = output_names, # 输出名 
                  # dynamic_axes=dynamic_axes,
                  verbose=True
                                )

import onnxruntime as ort
import onnx

# 加载 ONNX 模型
ort_session = ort.InferenceSession("RoITr16649.onnx")

num = 16649
num2 = 16649

src_pcd = torch.randn(num, 3).cpu().numpy()  # n=500，形状 [500, 3]
tgt_pcd = torch.randn(num2, 3).cpu().numpy()  # n=500，形状 [500, 3]
src_feats = torch.randn(num, 1).cpu().numpy()  # n=500，形状 [500, 1]
tgt_feats = torch.randn(num2, 1).cpu().numpy()  # n=500，形状 [500, 1]
src_normals = torch.randn(num, 3).cpu().numpy()  # n=500，形状 [500, 3]
tgt_normals = torch.randn(num2, 3).cpu().numpy()  # n=500，形状 [500, 3]
rot = torch.randn(3, 3).cpu().numpy()  # 形状 [3, 3]
trans = torch.randn(3, 1).cpu().numpy()  # 形状 [3, 1]
src_raw_pcd = torch.randn(num, 3).cpu().numpy() 


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
    input_shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
    print(f"Input name: {input_name}, shape: {input_shape}")


# 推理
outputs = ort_session.run(
    None, 
    input_feed
)
print("done")