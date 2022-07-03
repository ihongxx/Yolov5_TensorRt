采用yolov5提供的推理方式：
# yolov5 torch inference
    python inference.py --weights ./model/pth/cell.pt

# yolov5 onnx inference
    python inference.py --weights ./model/onnx/cell.onnx

# yolov5 fp32_engine inference
    python inference.py --weights ./model/trt/cell_fp32.engine
Speed: 0.0ms pre-process, 4.0ms inference, 7.9ms NMS per image at shape (1, 3, 640, 640)

# yolov5 fp16_engine inference
    python inference.py --weights ./model/trt/cell_fp16.engine
Speed: 1.0ms pre-process, 2.0ms inference, 4.0ms NMS per image at shape (1, 3, 640, 640)

# yolov5 int8_engine inference
    python inference.py --weights ./model/trt/cell_int8.engine
Speed: 1.0ms pre-process, 1.0ms inference, 1.9ms NMS per image at shape (1, 3, 640, 640)


采用自定义yolov5推理方式：(适配fp32、fp16类型推理)
# yolov5 int8_engine inference
    python int8_inference.py

将上面的做benchmark