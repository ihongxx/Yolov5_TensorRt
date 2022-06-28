# *** tensorrt校准模块  ***

from ast import arg
import os
import cv2
import torch
import torch.nn.functional as F
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import ctypes
import logging
import glob
logger = logging.getLogger(__name__)
# ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_char_p
# ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    print('shape:', shape)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    # return im, ratio, (dw, dh)
    return im

class Yolov5EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, args):
        trt.IInt8EntropyCalibrator2.__init__(self)       

        self.cache_file = args.cache_file
        self.batch_size = args.batch_size
        self.Channel = args.channel
        self.Height = args.height
        self.Width = args.width

        # self.img_list = [i.strip() for i in open('calib.txt').readlines()]
        self.img_list = glob.glob(os.path.join(args.CALIB_IMG_DIR, "*.jpg"))
        self.max_batch_idx = len(self.img_list) // self.batch_size  # max batch idx
        self.calibration_data = np.zeros((self.batch_size, 3, self.Height, self.Width), dtype=np.float32)
        self.device_input = cuda.mem_alloc(self.calibration_data.nbytes)
        self.current_idx = 0  # batch idx

    def get_batch_size(self):
        return self.batch_size

    @staticmethod   
    def img_process_v1(img):
        img = cv2.resize(img, (640, 640))
        # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
        img = img.transpose((2, 0, 1)).astype(np.float32)  #
        img /= 255.0
        return img
    
    @staticmethod
    def img_process_v2(img):
        img = letterbox(img)
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # img = np.ascontiguousarray(img)
        img /= 255.0
        return img

    def next_batch(self):
        if self.current_idx < self.max_batch_idx:
            # batch_imgs = np.zeros((self.batch_size, self.Channel, self.Height, self.Width))
            for i in range(self.batch_size):
                img = cv2.imread(self.img_list[i + self.current_idx * self.batch_size])
                img = self.img_process_v1(img)
                self.calibration_data[i] = img
            self.current_idx += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def get_batch(self, names):
        if self.current_idx + self.batch_size > len(self.img_list):
            return None
        
        current_batch = int(self.current_idx / self.batch_size)  # batch idx
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))
        
        # get batch images
        try:
            batch_imgs = self.next_batch()
            if batch_imgs.size == 0 or batch_imgs.size != self.batch_size*self.Channel*self.Height*self.Width:
                return None
            cuda.memcpy_htod(self.device_input, batch_imgs.astype(np.float32))  # host to device
            # self.current_idx += 
            return [int(self.device_input)]
        except:
            return None

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                logger.info("Using calibration cache to save time: {:}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            logger.info("Caching calibration data for future use: {:}".format(self.cache_file))
            f.write(cache)
