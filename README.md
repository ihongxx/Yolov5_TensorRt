采用yolov5官方提供的推理方式：(采用预热、同步推理)
# yolov5 torch inference
    python inference.py --weights ./model/pth/cell.pt
Speed: 1.0ms pre-process, 14.0ms inference, 8.0ms NMS per image at shape (1, 3, 640, 640)

# yolov5 onnx inference
    python inference.py --weights ./model/onnx/cell.onnx
Speed: 1.0ms pre-process, 11.0ms inference, 2.0ms NMS per image at shape (1, 3, 640, 640)

# yolov5 fp32_engine inference
    python inference.py --weights ./model/trt/cell_fp32.engine
Speed: 0.0ms pre-process, 6.0ms inference, 7.0ms NMS per image at shape (1, 3, 640, 640)

# yolov5 fp16_engine inference
    python inference.py --weights ./model/trt/cell_fp16.engine  --half True
Speed: 0.0ms pre-process, 3.0ms inference, 5.0ms NMS per image at shape (1, 3, 640, 640)

# yolov5 int8_engine inference
    python inference.py --weights ./model/trt/cell_int8.engine
Speed: 0.0ms pre-process, 2.0ms inference, 6.0ms NMS per image at shape (1, 3, 640, 640)


采用自定义yolov5 TensorRT推理方式（没有预热，因此process image耗时，并且采用异步推理）
# 采用官方简化版本：python_tensorrt_predict（）
    python int8_inference.py --weights ./model/trt/cell_fp32.engine
image process time: 0.76500 ms, inference time: 0.00699 ms, nms time: 0.00601 ms

    # 这边推理还是采用的fp32，因为onnx模型数据类型就是fp32, 所以就算更改了img数据类型为fp16，但是模型还是fp32类型
    python int8_inference.py --weights ./model/trt/cell_fp16.engine --half True
image process time: 0.75011 ms, inference time: 0.00300 ms, nms time: 0.00100 ms
    
    python int8_inference.py --weights ./model/trt/cell_fp32.engine
image process time: 0.75000 ms, inference time: 0.00300 ms, nms time: 0.00500 ms


# 采用自定义版本：my_tensorrt_predict（），该方法没有采用letterbox图片处理
    python int8_inference.py --weights ./model/trt/cell_fp32.engine
image process time: 0.00801 ms, inference time: 0.01099 ms, nms time: 0.00500 ms

    # 这边推理还是采用的fp32，因为onnx模型数据类型就是fp32
    python int8_inference.py --weights ./model/trt/cell_fp16.engine --half True
image process time: 0.00601 ms, inference time: 0.00396 ms, nms time: 0.00300 ms
    
    python int8_inference.py --weights ./model/trt/cell_fp32.engine
image process time: 0.00700 ms, inference time: 0.00700 ms, nms time: 0.00500 ms

# 采用自定义版本：my_letterbox_tensorrt_predict（），该方法采用letterbox图片处理
    python int8_inference.py --weights ./model/trt/cell_fp32.engine
image process time: 0.00701 ms, inference time: 0.01000 ms, nms time: 0.00399 ms

    # 这边推理还是采用的fp32，因为onnx模型数据类型就是fp32, 所以就算更改了img数据类型为fp16，但是模型还是fp32类型
    python int8_inference.py --weights ./model/trt/cell_fp16.engine --half True
image process time: 0.00900 ms, inference time: 0.00505 ms, nms time: 0.00300 ms
    
    python int8_inference.py --weights ./model/trt/cell_fp32.engine
image process time: 0.00801 ms, inference time: 0.00399 ms, nms time: 0.00300 ms


