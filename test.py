# from operator import mod
# from pyexpat import model
# import torch

# pt_file_path = './model/pth/cell.pt'
# state_dict = torch.load(pt_file_path, map_location='cuda:0')
# model = state_dict['model'].float().fuse().eval()
# for m in model.modules():
#     print('m:', m, 'm.type:', type(m))
#     print('######################')
# # for k,v in state_dict.items():
# #     print('k:', k, 'v:', v)

import time
from typing import List
import numpy as np
import tensorrt
import torch
from pycuda import driver
import pycuda.autoinit
from PIL import Image, ImageDraw, ImageFont

from utils.general import non_max_suppression, scale_coords

def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def trt_pre(batch, context, d_size,
            d_type):  # Need to set both input and output precisions to FP16 to fully enable FP16
    output = np.empty(d_size, dtype=d_type)
    batch = batch.reshape(-1)
    d_input = driver.mem_alloc(1 * batch.nbytes)
    d_output = driver.mem_alloc(output.nbytes)
    bindings = [int(d_input), int(d_output)]
    stream = driver.Stream()
    # Transfer input data to device
    driver.memcpy_htod_async(d_input, batch, stream)
    # Execute model
    context.execute_async_v2(bindings, stream.handle, None)
    # Transfer predictions back
    driver.memcpy_dtoh_async(output, d_output, stream)
    # Syncronize threads
    stream.synchronize()
    return output


def python_tensorrt_predict(model_path):
    # 加载模型A
    trt_model = tensorrt.Runtime(tensorrt.Logger(tensorrt.Logger.WARNING))
    # 反序列化模型
    engine = trt_model.deserialize_cuda_engine(open(model_path, "rb").read())
    # 创建推理上下文
    context = engine.create_execution_context()
    for binding in engine:
        if not engine.binding_is_input(binding):
            size = tensorrt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = tensorrt.nptype(engine.get_binding_dtype(binding))
        else:
            input_w = engine.get_binding_shape(binding)[-1]
            input_h = engine.get_binding_shape(binding)[-2]
    from utils.datasets import LoadImages
    source = "./data/1.jpg"
    start = time.perf_counter()
    dataset = LoadImages(source, img_size=[input_w, input_h], stride=32, auto=False)
    for path, im, im0s, vid_cap, s in dataset:
        image = Image.open(path)
        im = torch.from_numpy(im).to("cuda").float()
        im /= 255
        start1 = time.perf_counter()
        outputs = trt_pre(np.asarray(im.cpu(), dtype=np.float32), context, size, dtype)
        end1 = time.perf_counter()
        print(f"inference {end1 - start1}")
        # with open("log/outputs.txt", "w") as w:
        #     for i in outputs:
        #         w.write(str(i))
        #         w.write("\n")
        outputs = torch.as_tensor(outputs).reshape((-1, 7)).unsqueeze(0)
        pred = non_max_suppression(outputs, 0.25, 0.45)
        pred[0][:,:4] = scale_coords(im.shape[1:], pred[0][:,:4], im0s.shape)
        image = drawImage(image, list(pred))
        image.save("./runs/hxx/test_inference.jpg")
    end = time.perf_counter()
    print(f"{end -start}")

def drawImage(image, class_list):
    font = ImageFont.truetype(font='/data/kile/other/yolov3/font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300
    for i in class_list[0]:
        if not isinstance(i, List):
            i = list(i)
        label = str(i[-1])+"_"+str(i[-2])
        box = i[:-2]
        left, top, right, bottom = box
        top = int(top.numpy())
        left = int(left.numpy())
        bottom = int(bottom.numpy())
        right = int(right.numpy())
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=(0x27, 0xC1, 0x36))
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=(128, 0, 128))
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
    return image


if __name__ == '__main__':
    # 通官方的源码export.py生成tensorrt模型
    # model_path = r"/data/kile/202204/yolov5/log/2.engine"
    model_path = './model/trt/cell_int8_v1.engine'
    python_tensorrt_predict(model_path)
    from PIL import Image

    # path = r"/data/kile/data/oridata_100/n0942195117838/n0942195117838.jpeg"
    # image = cv2.imread(path)
    # im, _, _ = letterbox(image, (416,416), auto=False)
    # cv2.imwrite("/data/1.jpg", im)