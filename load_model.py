import torch
from ultralytics import YOLO

#Load a model
model = YOLO("yolo11n.pt")
depth_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
depth_model.eval()