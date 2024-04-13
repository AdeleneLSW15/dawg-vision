
"""
Use yolov8 to perform object detection. 

Make it -> detect a bottle and send it over to serial (shoot a bottle function)
"""
#need to install ultralytics package - pip install ultralytics.
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
import cv2
import torch
import torch.onnx
import torchvision.models as models

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8s.pt", device = device)  # load a pretrained model (recommended for training)
#model.to('cuda')
#For Nvidia Jetson Nano: TensorFlow RT for better inference time
model.export(format='engine')
#load to TensorRT mode




# Use the model - running inference
#results = trt_model.predict(source="0", show=True, stream = True, conf=0.4, save=True)
#print (results)
#results = model.predict(source='video source', task='detect', stream=True, save=False, verbose=False, conf = 0.5

#condition: do X when detect one or two person?