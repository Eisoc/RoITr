import onnx
import onnxruntime as ort
import torch



num = 12340

src_pcd = torch.randn(num, 3).cpu().numpy()  # n=500，形状 [500, 3]
tgt_pcd = torch.randn(num, 3).cpu().numpy()  # n=500，形状 [500, 3]
src_feats = torch.randn(num, 1).cpu().numpy()  # n=500，形状 [500, 1]
tgt_feats = torch.randn(num, 1).cpu().numpy()  # n=500，形状 [500, 1]
src_normals = torch.randn(num, 3).cpu().numpy()  # n=500，形状 [500, 3]
tgt_normals = torch.randn(num, 3).cpu().numpy()  # n=500，形状 [500, 3]
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
model = onnx.load("RoITr12340.onnx")

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
ort_session = ort.InferenceSession("RoITr12340.onnx")

# 推理
outputs = ort_session.run(
    None, 
    input_feed
)
print("done")