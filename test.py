from pyexpat import model
import torch

pt_file_path = './model/pth/cell.pt'
state_dict = torch.load(pt_file_path, map_location='cuda:0')
print(state_dict['model']) 
# for k,v in state_dict.items():
#     print('k:', k, 'v:', v)