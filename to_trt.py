import argparse
from tkinter import N
from cv2 import calibrateCamera
import torch
import onnx
import tensorrt as trt
# from Resnet50_myCalibrator import Yolov5EntropyCalibrator

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='fp16')
    parser.add_argument('--onnx_file_path', default='./model/onnx/cell.onnx', help='onnx file path')
    parser.add_argument('--engine_file_path', default='./model/trt/cell_fp16.engine', help='onnx file path')
    opt = parser.parse_args()
    return opt

def export_engine(opt, calib=None):

    f = './model/trt/cell_fp16.engine'
    # int8_mode = opt.int8

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.max_workspace_size = 4 << 30
    
    parser = trt.OnnxParser(network, logger)

    with open(opt.onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_error):
                print(parser.get_error(error))
            return None
    
    # try:
    #     engine_bytes  = builder.build_serialized_network(network, config)
    # except AttributeError:
    #     engine = builder.build_engine(network, config)
    #     engine_bytes = engine.serialize()
    #     del engine
    # with open(opt.engine_file_path, 'wb') as f:
    #     f.write(engine_bytes)

    # if not parser.parse_from_file(str(onnx)):
    #     raise RuntimeError(f'failed to load ONNX file:{onnx}')

    # inputs = [network.get_input(i) for i in range(network.num_inputs)]
    # outputs = [network.get_output(i) for i in range(network.num_outputs)]

    if opt.mode.lower() == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif opt.mode.lower() == 'int8':
        config.set_flag(trt.BuilderFlage.INT8)
        config.int8_calibrator = calib


    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        t.write(engine.serialize())

def main():
    opt = parse_opt()

    if opt.mode.lower() == 'int8':
        calib = Yolov5EntropyCalibrator(opt)
    else:
        calib = None

    export_engine(opt, calib)

if __name__ == '__main__':
    main()