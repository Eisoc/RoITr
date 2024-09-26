import torch
import torch.onnx
import copy
import os, argparse, json, shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3" # for Pytorch DistributedDataParallel(DDP) training
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

# 加载权重文件（.pth）
state = torch.load('weights/model_3dmatch.pth')
config.model.load_state_dict({k.replace('module.', ''): v for k, v in state['state_dict'].items()})
config.model.eval()  # 切换到评估模式

src_pcd, tgt_pcd, src_feats, tgt_feats, src_normals, tgt_normals,rot, trans, src_raw_pcd = torch.randn(1000, 3).cuda(), torch.randn(1000, 3).cuda(), torch.randn(1000, 1).cuda(), torch.randn(1000, 1).cuda(), torch.randn(1000, 3).cuda(), torch.randn(1000, 3).cuda(), torch.randn(3, 3).cuda(), torch.randn(3, 1).cuda(), torch.randn(1000, 3).cuda()
dummy_input = src_pcd, tgt_pcd, src_feats, tgt_feats, src_normals, tgt_normals,rot, trans, src_raw_pcd  
# 将模型导出为 ONNX 格式
torch.onnx.export(config.model,                   # 要转换的模型
                  dummy_input,             
                  "RoITr.onnx",       # 输出文件名
                  export_params=True,      # 导出模型权重
                  opset_version=13,        
                  do_constant_folding=True,# 常量折叠优化
                  input_names = ["src_pcd", "tgt_pcd", "src_feats", "tgt_feats", "src_normals", "tgt_normals","rot", "trans", "src_raw_pcd"], # 输入名
                  output_names = ['output_dict'],# 输出名 
                  # dynamic_axes={'input': {0: 'batch_size'},  # 动态批量处理 
                                # 'output': {0: 'batch_size'}}
                                )

import onnxruntime as ort

# 加载 ONNX 模型
ort_session = ort.InferenceSession("RoITr.onnx")

# 推理
outputs = ort_session.run(
    None, 
    dummy_input
)
print("done")