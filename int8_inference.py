from mimetypes import init
import re
import time
from typing import List
import cv2
from matplotlib.pyplot import cla
import numpy as np
import torch
import pandas as pd
import yaml
import torch.nn as nn
import pycuda.driver as cuda
import pycuda.autoinit
# from pathlib import Path
import tensorrt as trt
from collections import OrderedDict, namedtuple
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages
from utils.plots import Annotator, colors
from utils.general import non_max_suppression, check_suffix, check_version, scale_coords, check_img_size

def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def loadEngine2TensorRT(filepath):
    G_LOGGER = trt.Logger(trt.Logger.INFO)
    # 反序列化引擎
    with open(filepath, 'rb') as f, trt.Runtime(G_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    return engine

def normalize_image(image):
    # 调整大小、平滑、转换图像为CHW
    c, h, w = 3, 640, 640
    im = cv2.resize(image, (h,w))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.transpose((2, 0, 1)).astype(np.float32)
    im = np.ascontiguousarray(im)  # 数据在内存中存储变得连续
    im /= 255 
    if len(im.shape) == 3:
        im = im[None]
    return im

class Backend(nn.Module):
    def __init__(self, weights='yolov5s.pt', device=torch.device('cuda:0'), dnn=False, data=None, fp16=False):
        super().__init__()

        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        engine = loadEngine2TensorRT(weights)
        bindings = OrderedDict()
        for index in range(engine.num_bindings):
            # print('index:', index)
            name = engine.get_binding_name(index)
            # print('name:', name)
            dtype = trt.nptype(engine.get_binding_dtype(index))
            shape = tuple(engine.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
            bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            if engine.binding_is_input(index) and dtype == np.float16:
                fp16 = True
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        context = engine.create_execution_context()
        batch_size = bindings['images'].shape[0]
        self.__dict__.update(locals())  # assign all variables to self
    
    def forward(self, im, augment=False, visualize=False, val=False):
        b, ch, h, w = im.shape  # batch, channel, height, width
        self.binding_addrs['images'] = int(im.data_ptr())
        assert im.shape == self.bindings['images'].shape, (im.shape, self.bindings['images'].shape)
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = self.bindings['output'].data
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=self.device)
        return (y, []) if val else y
    
    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        # if any((self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb)):  # warmup types
            # if self.device.type != 'cpu':  # only warmup GPU models
        im = torch.zeros(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
        for _ in range(1):  #
            self.forward(im)  # warmup

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        '''
        # print("engine.get_binding_shape(binding):", engine.get_binding_shape(binding))
        # engine.get_binding_shape(binding): (1, 3, 640, 640)   images
        # engine.get_binding_shape(binding): (1, 3, 80, 80, 9)  onnx:Sigmoid_390
        # engine.get_binding_shape(binding): (1, 3, 40, 40, 9)  onnx:Sigmoid_442
        # engine.get_binding_shape(binding): (1, 3, 20, 20, 9)  onnx::Sigmoid_494
        # engine.get_binding_shape(binding): (1, 25200, 9)      output
        '''
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)  # 创建host锁业内存
        device_mem = cuda.mem_alloc(host_mem.nbytes) # 创建device内存
        bindings.append(int(device_mem))  # bindings 存放device内存地址
        if engine.binding_is_input(binding):
            inputs.append({'host':host_mem, 'device': device_mem})  # input存放input的device地址
        else:
            outputs.append({'host':host_mem, 'device': device_mem})  # output存放除了input外的device地址
    return inputs, outputs, bindings, stream

def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp['device'], inp['host'], stream) for inp in inputs]
    # Run inference.
    # print('bindings:', bindings)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out['host'], out['device'], stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out['host'] for out in outputs]

def post_process(im, im0, pred, image_output_file, dt):
    names = ['cells', 'Platelets', 'RBC', 'WBC']
    s = "f'image {0}/{self.nf} {path}: '"
    t4 = time_sync()
    # dt[1] += t3 - t2
    # print('inference time:', dt[1])

    # NMS
    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
    dt[2] += time_sync() - t4
    print('nms time: {:.5f} ms'.format(dt[2]))

    for i, det in enumerate(pred):  # per image
        
        im0 = im0.copy()

        # p = Path(p)  # to Path
        # save_path = str(save_dir / p.name)  # im.jpg
        s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if False else im0  # for save_crop
        annotator = Annotator(im0, line_width=3, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            # print('det:', det)

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # if save_txt:  # Write to file
                #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format

                # if save_img or save_crop or view_img:  # Add bbox to image
                if True:
                    c = int(cls)  # integer class
                    label = None if False else (names[c] if False else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

        # Stream results
        im0 = annotator.result()

        cv2.imwrite(image_output_file, im0)

def python_tensorrt_predict(img_file, engine_file, image_output_file):
    '''通官方的源码export.py生成tensorrt模型'''
    device = torch.device("cuda:0")
    # data = "./data/cell.yaml"
    model = Backend(weights=engine_file, device=device)
    # model = DetectMultiBackend(weights=engine_file, device=device, fp16=half)

    stride, pt = 32, False
    imgsz = (640,640)
    imgsz = check_img_size(imgsz, s=stride)
    source = img_file
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    # Run inference
    # model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt = [0.0, 0.0, 0.0]
    for path, im, im0, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        # model.fp16 = False
        fp16 = False
        im = im.half() if fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1
        print('image process time: {:.5f} ms'.format(dt[0]))

        # Inference
        pred = model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2
        print('inference time: {:.5f} ms'.format(dt[1]))
        # print('pred1:', pred)

    post_process(im, im0, pred, image_output_file, dt)

def my_tensorrt_predict(img_file, engine_file, image_output_file):
    '''通过一般的数据处理进行模型推理'''

    dt = [0.0, 0.0, 0.0]

    # 加载engine引擎
    engine = loadEngine2TensorRT(engine_file)

    # 分配内存空间
    inputs, outputs, bindings, stream = allocate_buffers(engine)  

    # 创建上下文执行
    context = engine.create_execution_context()

    # 将图片加载到inputs中
    t1 = time_sync()
    im0 = cv2.imread(img_file)
    im = normalize_image(im0)
    t2 = time_sync()
    dt[0] += t2 - t1
    print('image process time: {:.5f} ms'.format(dt[0]))
    # im.shape  im: (1, 3, 640, 640)

    # 将图片内容放到inputs地址中，这边和yolov5官方不同在于官方是直接将图片的地址赋值给inputs
    inputs[0]['host'] = im  # 将图片数组放到inputs[0]['host']中，要求im是连续的内存空间

    # 输出是list，list中0、1、2都不是真正的输出，只有[3]才是output，
    outputs = do_inference_v2(context, bindings, inputs, outputs, stream)
    t3 = time_sync()
    dt[1] += t3 - t2
    print('inference time: {:.5f} ms'.format(dt[1]))

    # outputs[3]里面是array（226800）， 将他进行维度转换变成1，25200，9
    output = outputs[3].reshape(1, -1, 9)
    # 将numpy.ndarray转为tensor， 因为后面nms需要to device
    pred = torch.from_numpy(output)  
    # print('output data_ptr:', pred.data_ptr())
    # output.shape: torch.Size([1, 25200, 9])

    post_process(im, im0, pred, image_output_file, dt)

def my_letterbox_tensorrt_predict(img_file, engine_file, image_output_file):
    device = torch.device("cuda:0")

    # 加载engine引擎
    engine = loadEngine2TensorRT(engine_file)

    # 分配内存空间
    inputs, outputs, bindings, stream = allocate_buffers(engine)  

    # 创建上下文执行
    context = engine.create_execution_context()

    # 将图片加载到inputs中
    t1 = time_sync()

    stride, names, pt = 32, ['cells', 'Platelets', 'RBC', 'WBC'], False
    imgsz = (640,640)
    imgsz = check_img_size(imgsz, s=stride)
    dataset = LoadImages(img_file, img_size=imgsz, stride=stride, auto=pt)

    # Run inference
    dt = [0.0, 0.0, 0.0]
    for path, im, im0, vid_cap, s in dataset:
        # im.dtype :unit8 ,如需要转化成float32
        im = im.astype(np.float32)
        # im = torch.from_numpy(im)
        im = im / 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1
        print('image process time: {:.5f} ms'.format(dt[0]))

        # print('im.shape:', im.shape)

        # 将图片内容放到inputs地址中，这边和yolov5官方不同在于官方是直接将图片的地址赋值给inputs
        inputs[0]['host'] = im  # 将图片数组放到inputs[0]['host']中，要求im是连续的内存空间

        # 输出是list，list中0、1、2都不是真正的输出，只有[3]才是output，
        outputs = do_inference_v2(context, bindings, inputs, outputs, stream)
        t3 = time_sync()
        dt[1] += t3 - t2
        print('inference time: {:.5f} ms'.format(dt[1]))

        # outputs[3]里面是array（226800）， 将他进行维度转换变成1，25200，9
        output = outputs[3].reshape(1, -1, 9)
        # 将numpy.ndarray转为tensor， 因为后面nms需要to device
        pred = torch.from_numpy(output)  
        # print('output data_ptr:', pred.data_ptr())
        # output.shape: torch.Size([1, 25200, 9])

    post_process(im, im0, pred, image_output_file, dt)

if __name__ == '__main__':
    # model_path = r"/data/kile/202204/yolov5/log/2.engine"
    engine_file = "./model/trt/cell_fp16.engine"
    img_file = './data/1.jpg'
    save_image_file_1 = './runs/hxx/test_1.jpg'
    save_image_file_2 = './runs/hxx/test_2.jpg'
    save_image_file_3 = './runs/hxx/test_3.jpg'
    python_tensorrt_predict(img_file, engine_file, save_image_file_1)
    my_tensorrt_predict(img_file, engine_file, save_image_file_2)
    my_letterbox_tensorrt_predict(img_file, engine_file, save_image_file_3)