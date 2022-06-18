from ast import arg
from turtle import st
from matplotlib.pyplot import cla
from numpy import isin
import onnx
import argparse
import torch.nn as nn
import torch
import models
import onnxsim

from utils.general import check_img_size

class SiLu(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def export_onnx(args):
    checkpoint = torch.load(args.torch_file_path)
    model = checkpoint['model'].float().fuse().eval()

    from models.yolo import Detect, Model
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = True
            if t is Detect:
                if not isinstance(m.anchor_grid, list):
                    delattr(m, 'anchor_grid')
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)  # nl 检测层数量
    
    nc, names = 4, model.names
    print('model.nc:', nc, 'model.names:', names)

    img_size = args.img_size
    gs =int(max(model.stride))
    img_size = [check_img_size(x, gs) for x in img_size]
    im = torch.zeros(args.batch_size, 3, *img_size).to(args.device)

    for k,m in model.named_modules():
        if isinstance(m, models.common.Conv):
            if isinstance(m.act, nn.SiLU):
                m.act = SiLu()
        elif isinstance(m, models.yolo.Detect):
            m.inplace = False
            m.onnx_dynamic = args.dynamic
            if hasattr(m, 'forward_export'):
                m.forward = m.forward_export
    
    for _ in range(2):
        y = model(im)
    
    grid = model.model[-1].anchor_grid
    model.model[-1].anchor_grid = [a[..., :1, :1, :] for a in grid]

    dynamic = args.dynamic
    print('start export')
    torch.onnx.export(model, im, args.onnx_file_path, verbose=False, opset_version=args.opset,
                    training=torch.onnx.TrainingMode.EVAL, do_constant_folding=True,
                    input_names=['images'],
                    output_names=['output'],
                    dynamic_axes={'images': {0:'batch', 2:'height', 3:'width'},
                                  'output': {0:'batch', 1:'anchors'}
                                    } if dynamic else None)
    
    print(' export model')
    model_onnx = onnx.load(args.onnx_file_path)
    onnx.checker.check_model(model_onnx)

    model_onnx, check = onnxsim.simplify(
                        model_onnx,
                        dynamic_input_shape=args.dynamic,
                        input_shapes={'images': list(im.shape)} if args.dynamic else None)
    assert check, 'assert check faild'
    print('export sim model')
    onnx.save(model_onnx, args.onnx_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default='1', help='batch size')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640,640], help='image size')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--torch_file_path', type=str, default='./model/pth/cell.pt')
    parser.add_argument('--onnx_file_path', type=str, default='./model/onnx/cell.onnx')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version')


    args = parser.parse_args()

    export_onnx(args)