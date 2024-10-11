import os
import torch
from tqdm import tqdm
from lib.trainer import Trainer
from lib.utils import to_o3d_pcd
from visualizer.visualizer import Visualizer, create_visualizer
from visualizer.feature_space import visualize_feature_space
import open3d as o3d
import numpy as np

class Tester(Trainer):
    '''
    Tester
    '''

    def __init__(self, config):
        Trainer.__init__(self, config)

    def test(self):
        print('Starting to evaluate on test datasets...')
        os.makedirs(f'{self.snapshot_dir}/{self.config.benchmark}', exist_ok=True)


        num_iter = len(self.loader['test'])

        c_loader_iter = self.loader['test'].__iter__()
        self.model.eval()
        with torch.no_grad():
            for idx in tqdm(range(num_iter)):
                torch.cuda.synchronize()
                # inputs = c_loader_iter.next()
                inputs = next(c_loader_iter)

                #######################################
                # Load inputs to device
                for k, v in inputs.items():
                    if v is None:
                        pass
                    elif type(v) == list:
                        inputs[k] = [items.to(self.device) for items in v]
                    else:
                        inputs[k] = v.to(self.device)
                ##################
                # forward pass
                ##################
                rot, trans = inputs['rot'][0], inputs['trans'][0]
                src_pcd, tgt_pcd = inputs['src_points'].contiguous(), inputs['tgt_points'].contiguous()
                src_normals, tgt_normals = inputs['src_normals'].contiguous(), inputs[
                    'tgt_normals'].contiguous()
                src_feats, tgt_feats = inputs['src_feats'].contiguous(), inputs['tgt_feats'].contiguous()
                src_raw_pcd = inputs['raw_src_pcd'].contiguous()

                ### for onnx
                import onnx
                import onnxruntime as ort

                input_feed = {
                "src_pcd": src_pcd.cpu().numpy() ,
                "tgt_pcd": tgt_pcd.cpu().numpy() ,
                "src_feats": src_feats.cpu().numpy() ,
                "tgt_feats": tgt_feats.cpu().numpy() ,
                "src_normals": src_normals.cpu().numpy() ,
                "tgt_normals": tgt_normals.cpu().numpy() ,
                "rot": rot.cpu().numpy() ,
                "trans": trans.cpu().numpy() ,
                "src_raw_pcd": src_raw_pcd.cpu().numpy() 
                }   

                onnx_file = "debug_16649.onnx"
                model = onnx.load(onnx_file)
                print("onnx load done")

                # 检查模型输入形状
                # for input in model.graph.input:
                #     input_name = input.name
                #     input_shape = []
                #     for dim in input.type.tensor_type.shape.dim:
                #         if dim.dim_param:
                #             input_shape.append(dim.dim_param)
                #         else:
                #             input_shape.append(dim.dim_value)
                #     print(f"Input name: {input_name}, shape: {input_shape}")

                # # 检查模型输出形状
                # for output in model.graph.output:
                #     output_name = output.name
                #     output_shape = []
                #     for dim in output.type.tensor_type.shape.dim:
                #         if dim.dim_param:
                #             output_shape.append(dim.dim_param)
                #         else:
                #             output_shape.append(dim.dim_value)
                #     print(f"Output name: {output_name}, shape: {output_shape}")

                ort_session = ort.InferenceSession(onnx_file)

                # 推理
                output_names = [output.name for output in ort_session.get_outputs()]
                # outputs_onnx = ort_session.run(
                #     output_names, 
                #     input_feed
                # )

                onnx_dict = dict(zip(output_names, ort_session.run(
                    output_names, 
                    input_feed
                ) ))
                
                print("dict done")
                # with open("outputs.txt", "w") as file:
                #     print("writing.......")
                #     for name, value in onnx_dict.items():
                #         # 将输出名称和值写入文件
                #         file.write(f"{name}: shape: {value.shape}{value}\n")
                # print(f"saved outputs.txt")

                ### onnx
                outputs = onnx_dict
                data = dict()
                data['src_raw_pcd'] = src_raw_pcd
                data['src_pcd'], data['tgt_pcd'] = src_pcd, tgt_pcd
                data['src_nodes'], data['tgt_nodes'] = outputs['src_nodes'], outputs['tgt_nodes']
                data['src_node_desc'], data['tgt_node_desc'] = outputs['src_node_feats'], outputs['tgt_node_feats']
                data['src_point_desc'], data['tgt_point_desc'] = outputs['src_point_feats'], outputs['tgt_point_feats']
                data['src_corr_pts'], data['tgt_corr_pts'] = outputs['src_corr_points'], outputs['tgt_corr_points']
                data['confidence'] = outputs['corr_scores']
                data['gt_tgt_node_occ'] = outputs['gt_tgt_node_occ']
                data['gt_src_node_occ'] = outputs['gt_src_node_occ']
                data['rot'], data['trans'] = rot, trans
                if self.config.benchmark == '4DMatch' or self.config.benchmark == '4DLoMatch':
                    data['metric_index_list'] = inputs['metric_index']
                torch.save(data, f'{self.snapshot_dir}/{self.config.benchmark}/onnx_{idx}.pth')
                print(f'saved in: {self.snapshot_dir}/{self.config.benchmark}/onnx_{idx}.pth')

                ### 正常engine输出
                # outputs = self.model.forward(src_pcd, tgt_pcd, src_feats, tgt_feats, src_normals, tgt_normals,
                #                              rot, trans, src_raw_pcd)
                # data = dict()
                # data['src_raw_pcd'] = src_raw_pcd.cpu()
                # data['src_pcd'], data['tgt_pcd'] = src_pcd.cpu(), tgt_pcd.cpu()
                # data['src_nodes'], data['tgt_nodes'] = outputs['src_nodes'].cpu(), outputs['tgt_nodes'].cpu()
                # data['src_node_desc'], data['tgt_node_desc'] = outputs['src_node_feats'].cpu().detach(), outputs['tgt_node_feats'].cpu().detach()
                # data['src_point_desc'], data['tgt_point_desc'] = outputs['src_point_feats'].cpu().detach(), outputs['tgt_point_feats'].cpu().detach()
                # data['src_corr_pts'], data['tgt_corr_pts'] = outputs['src_corr_points'].cpu(), outputs['tgt_corr_points'].cpu()
                # data['confidence'] = outputs['corr_scores'].cpu().detach()
                # data['gt_tgt_node_occ'] = outputs['gt_tgt_node_occ'].cpu()
                # data['gt_src_node_occ'] = outputs['gt_src_node_occ'].cpu()
                # data['rot'], data['trans'] = rot.cpu(), trans.cpu()
                # if self.config.benchmark == '4DMatch' or self.config.benchmark == '4DLoMatch':
                #     data['metric_index_list'] = inputs['metric_index']


                # torch.save(data, f'{self.snapshot_dir}/{self.config.benchmark}/{idx}.pth')
                # print(f'saved in: {self.snapshot_dir}/{self.config.benchmark}/{idx}.pth')
                # print("test")
                ###########################################################

    





def get_trainer(config):
    '''
    Get corresponding trainer according to the config file
    :param config:
    :return:
    '''

    if config.dataset == 'tdmatch' or config.dataset == 'fdmatch' or config.dataset == 'hx':
        return Tester(config)
    else:
        raise NotImplementedError
