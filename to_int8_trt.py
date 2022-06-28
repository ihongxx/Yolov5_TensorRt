import argparse
from tkinter import N
import tensorrt as trt
from myCalibrator import Yolov5EntropyCalibrator

def export_engine(opt, calib=None):

    f = opt.engine_file_path

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

    if opt.mode.lower() == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif opt.mode.lower() == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calib

    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        t.write(engine.serialize())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--channel', type=int, default=3)
    parser.add_argument('--height', type=int, default=640)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--cache_file', type=str, default='model/yolov5s_calibration.cache')
    parser.add_argument('--mode', type=str, default='int8')
    parser.add_argument('--CALIB_IMG_DIR', type=str, default='E:\\Datasets\\IMAGES\\Object_Detection\\Red_Cell\\images')
    parser.add_argument('--onnx_file_path', default='./model/onnx/cell.onnx', help='onnx file path')
    parser.add_argument('--engine_file_path', default='./model/trt/cell_int8_process.engine', help='onnx file path')
    args = parser.parse_args()

    if args.mode.lower() == 'int8':
        calib = Yolov5EntropyCalibrator(args)
    else:
        calib = None

    export_engine(args, calib)