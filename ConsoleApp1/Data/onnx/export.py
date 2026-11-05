from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

model.export(format="onnx")