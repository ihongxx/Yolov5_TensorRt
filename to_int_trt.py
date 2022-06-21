import argparse
from multiprocessing.spawn import import_main_path
from tkinter import N
from cv2 import calibrateCamera
import torch
import onnx
import tensorrt as trt
# from myCalibrator import Yolov5EntropyCalibrator
from myCalibrator import Calibrator
import numpy as np
import cv2
import os
import glob


BATCH_SIZE = 25
BATCH = 10
height = 640
width = 640
# CALIB_IMG_DIR = '/home/willer/yolov5-3.1/data/coco/images/train2017'
CALIB_IMG_DIR = 'E:\\Datasets\\IMAGES\\Object_Detection\\Red_Cell\\images'
onnx_model_path = './model/onnx/cell.onnx'
# onnx_model_path = "/home/willer/yolov5-4.0/models/models_silu/yolov5s-simple.onnx"
def preprocess_v1(image_raw):
    h, w, c = image_raw.shape
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    # Calculate widht and height and paddings
    r_w = width / w
    r_h = height / h
    if r_h > r_w:
        tw = width
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((height - th) / 2)
        ty2 = height - th - ty1
    else:
        tw = int(r_h * w)
        th = height
        tx1 = int((width - tw) / 2)
        tx2 = width - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th))
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
    )
    image = image.astype(np.float32)
    # Normalize to [0,1]
    image /= 255.0
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])
    # CHW to NCHW format
    #image = np.expand_dims(image, axis=0)
    # Convert the image to row-major order, also known as "C order":
    #image = np.ascontiguousarray(image)
    return image


def preprocess(img):
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    return img

class DataLoader:
    def __init__(self):
        self.index = 0
        self.length = BATCH
        self.batch_size = BATCH_SIZE
        # self.img_list = [i.strip() for i in open('calib.txt').readlines()]
        self.img_list = glob.glob(os.path.join(CALIB_IMG_DIR, "*.jpg"))
        assert len(self.img_list) > self.batch_size * self.length, '{} must contains more than '.format(CALIB_IMG_DIR) + str(self.batch_size * self.length) + ' images to calib'
        print('found all {} images to calib.'.format(len(self.img_list)))
        self.calibration_data = np.zeros((self.batch_size,3,height,width), dtype=np.float32)

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                assert os.path.exists(self.img_list[i + self.index * self.batch_size]), 'not found!!'
                img = cv2.imread(self.img_list[i + self.index * self.batch_size])
                img = preprocess_v1(img)
                self.calibration_data[i] = img

            self.index += 1

            # example only
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.length

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='int8')
    parser.add_argument('--onnx_file_path', default='./model/onnx/cell.onnx', help='onnx file path')
    parser.add_argument('--engine_file_path', default='./model/trt/cell_int8.engine', help='onnx file path')
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

    if opt.mode.lower() == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif opt.mode.lower() == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calib


    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        t.write(engine.serialize())

def main():
    opt = parse_opt()

    if opt.mode.lower() == 'int8':
        # calib = Yolov5EntropyCalibrator(opt)
        calibration_table = 'model/yolov5s_calibration.cache'
        calibration_stream = DataLoader()
        calib = Calibrator(calibration_stream, calibration_table)
    else:
        calib = None

    export_engine(opt, calib)

if __name__ == '__main__':
    main()