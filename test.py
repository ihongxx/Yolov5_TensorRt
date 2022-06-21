from operator import mod
from pyexpat import model
import torch

pt_file_path = './model/pth/cell.pt'
state_dict = torch.load(pt_file_path, map_location='cuda:0')
model = state_dict['model'].float().fuse().eval()
for m in model.modules():
    print('m:', m, 'm.type:', type(m))
    print('######################')
# for k,v in state_dict.items():
#     print('k:', k, 'v:', v)