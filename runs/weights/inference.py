import torch

from models.yolo import Model

# 创建一个空的YOLOv5模型
model = Model()

# 加载权重文件
weights_path = 'runs/expX/weights/best.pt'  # 替换为你的权重文件路径
checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))

# 将权重文件中的参数加载到模型中
model.load_state_dict(checkpoint['model'])

# 设置模型为评估模式
model.eval()