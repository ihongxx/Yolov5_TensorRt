import imp
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
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import tensorrt as trt
from collections import OrderedDict, namedtuple
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages
from utils.plots import Annotator, colors
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, check_suffix, check_version, scale_coords, check_img_size

def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def export_formats():
    # YOLOv5 export formats
    x = [['PyTorch', '-', '.pt', True],
         ['TorchScript', 'torchscript', '.torchscript', True],
         ['ONNX', 'onnx', '.onnx', True],
         ['OpenVINO', 'openvino', '_openvino_model', False],
         ['TensorRT', 'engine', '.engine', True],
         ['CoreML', 'coreml', '.mlmodel', False],
         ['TensorFlow SavedModel', 'saved_model', '_saved_model', True],
         ['TensorFlow GraphDef', 'pb', '.pb', True],
         ['TensorFlow Lite', 'tflite', '.tflite', False],
         ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', False],
         ['TensorFlow.js', 'tfjs', '_web_model', False]]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'GPU'])


def loadEngine2TensorRT(filepath):
    G_LOGGER = trt.Logger(trt.Logger.INFO)
    # 反序列化引擎
    with open(filepath, 'rb') as f, trt.Runtime(G_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    return engine

class Backend(nn.Module):
    def __init__(self, weights='yolov5s.pt', device=torch.device('cuda:0'), dnn=False, data=None, fp16=False):

        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs = self.model_type(w)  # get backend
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        w = attempt_download(w)  # download if not local
        fp16 &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16
        if data:  # data.yaml path (optional)
            with open(data, errors='ignore') as f:
                names = yaml.safe_load(f)['names']  # class names

        check_version(trt.__version__, '7.0.0', hard=True)
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        engine = loadEngine2TensorRT(weights)
        bindings = OrderedDict()
        for index in range(engine.num_bindings):
            print('index:', index)
            name = engine.get_binding_name(index)
            print('name:', name)
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

    @staticmethod
    def model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        # from export import export_formats
        suffixes = list(export_formats().Suffix) + ['.xml']  # export suffixes
        check_suffix(p, suffixes)  # checks
        p = Path(p).name  # eliminate trailing separators
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, xml2 = (s in p for s in suffixes)
        xml |= xml2  # *_openvino_model or *.xml
        tflite &= not edgetpu  # *.tflite
        return pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs
# class HostDeviceMem(object):
#     def __init__(self, host_mem, device_mem):
#         self.host = host_mem
#         self.device = device_mem

#     def __str__(self):
#         return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

#     def __repr__(self):
#         return self.__str__()

# def allocate_buffers(engine):
#     inputs = []
#     outputs = []
#     bindings = []
#     stream = cuda.Stream()

#     Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
#     for index in range(engine.num_bindings):
#         print('index:', index)
#         name = engine.get_binding_name(index)
#         print('name:', name)
#         dtype = trt.nptype(engine.get_binding_dtype(index))
#         shape = tuple(engine.get_binding_shape(index))
#         data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to('cuda:0')
#         bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
#         if engine.binding_is_input(index) and dtype == np.float16:
#             fp16 = True
#     binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
#     context = engine.create_execution_context()
#     batch_size = bindings['images'].shape[0]

#     for binding in engine:
#         print('binding:', binding)
#         size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
#         dtype = trt.nptype(engine.get_binding_dtype(binding))
#         host_mem = cuda.pagelocked_empty(size, dtype)
#         device_mem = cuda.mem_alloc(host_mem.nbytes)
#         bindings.append(int(device_mem))
#         if engine.binding_is_input(binding):
#             inputs.append(HostDeviceMem(host_mem, device_mem))
#         else:
#             outputs.append(HostDeviceMem(host_mem, device_mem))
#     return inputs, outputs, bindings, stream

# def load_normalized_test_case(test_image, pagelocked_buffer):
#     # 将输入图片转换为CHW numpy数组
#     def normalize_image(image):
#         # 调整大小、平滑、转换图像为CHW
#         c, h, w = 3, 640, 640
#         img = cv2.resize(image, (h,w))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = img.transpose((2, 0, 1)).astype(np.float32).ravel()  # .ravel将3*640*640铺平1228800
#         # img /= 255.0
#         return img
    
#     # 规范化图像，并将图像复制到锁页内存中
#     np.copyto(pagelocked_buffer, normalize_image(cv2.imread(test_image)))
#     return test_image

# def do_inference_v2(context, bindings, inputs, outputs, stream):
#     # Transfer input data to the GPU.
#     [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
#     # Run inference.
#     context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
#     # Transfer predictions back from the GPU.
#     [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
#     # Synchronize the stream
#     stream.synchronize()
#     # Return only the host outputs.
#     return [out.host for out in outputs]

def python_tensorrt_predict(img_file, engine_file):
    # engine = loadEngine2TensorRT(model_path)
    # inputs, outputs, bindings, stream = allocate_buffers(engine)  # ??
    device = torch.device("cuda:0")
    half = False
    # data = "./data/cell.yaml"
    model = Backend(weights=engine_file, device=device)
    
    # model = DetectMultiBackend(weights=engine_file, device=device, fp16=half)

    stride, names, pt = 32, ['cells', 'Platelets', 'RBC', 'WBC'], False
    imgsz = (640,640)
    imgsz = check_img_size(imgsz, s=stride)
    source = img_file
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    bs = 1  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        # model.fp16 = False
        fp16 = False
        im = im.half() if fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()

        # Inference
        pred = model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2
        print('inference time:', dt[1])
        # print('pred1:', pred)

        # NMS
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
        dt[2] += time_sync() - t3
        print('nms time: {:.5f} ms'.format(dt[2]))
        # print('pred2:', pred)
        # dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        # Process predictions
        for i, det in enumerate(pred):  # per image
            
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
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
            # if view_img:
            #     cv2.imshow(str(p), im0)
            #     cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # if save_img:
            if True:
                if dataset.mode == 'image':
                    cv2.imwrite('./runs/hxx/test_process_6_29_1.jpg', im0)
    # context = engine.create_execution_context()

    # image = load_normalized_test_case(img_file, inputs[0].host)
    # t_begine = time.time()
    # trt_outputs = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    # t_end = time.time()
    # print('int8 inference time:', (t_end - t_begine))
    # print(len(trt_outputs))

# def drawImage(image, class_list):
#     font = ImageFont.truetype(font='/data/kile/other/yolov3/font/FiraMono-Medium.otf',
#                               size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
#     thickness = (image.size[0] + image.size[1]) // 300
#     for i in class_list[0]:
#         if not isinstance(i, List):
#             i = list(i)
#         label = str(i[-1])+"_"+str(i[-2])
#         box = i[:-2]
#         left, top, right, bottom = box
#         top = int(top.numpy())
#         left = int(left.numpy())
#         bottom = int(bottom.numpy())
#         right = int(right.numpy())
#         draw = ImageDraw.Draw(image)
#         label_size = draw.textsize(label, font)

#         top = max(0, np.floor(top + 0.5).astype('int32'))
#         left = max(0, np.floor(left + 0.5).astype('int32'))
#         bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
#         right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

#         if top - label_size[1] >= 0:
#             text_origin = np.array([left, top - label_size[1]])
#         else:
#             text_origin = np.array([left, top + 1])
#         for i in range(thickness):
#             draw.rectangle(
#                 [left + i, top + i, right - i, bottom - i],
#                 outline=(0x27, 0xC1, 0x36))
#         draw.rectangle(
#             [tuple(text_origin), tuple(text_origin + label_size)],
#             fill=(128, 0, 128))
#         draw.text(text_origin, label, fill=(0, 0, 0), font=font)
#         del draw
#     return image


if __name__ == '__main__':
    # 通官方的源码export.py生成tensorrt模型
    # model_path = r"/data/kile/202204/yolov5/log/2.engine"
    engine_file = "./model/trt/cell_int8_v1.engine"
    img_file = './data/1.jpg'
    python_tensorrt_predict(img_file, engine_file)