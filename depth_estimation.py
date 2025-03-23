import torch 
import torchvison.transforms as transforms
from torchvision.transforms import Compose
import cv2
import numpy as np

# Depth Estimation Modules
def depth_preprocess_image(image):
    # Resize the input image to dimensions divisible by 32
    h, w, _ = image.shape
    new_h = h - (h % 32)
    new_w = w - (w % 32)
    resized_image = cv2.resize(image, (new_w, new_h))

    input_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return input_transform(resized_image).unsqueeze(0)

def postprocess_depth_map(depth_tensor, original_size):
    depth_map = depth_tensor.squeeze().cpu().numpy()
    depth_map = cv2.resize(depth_map, original_size)
    depth_map = depth_map / depth_map.max()
    return depth_map

def extract_depth_map(model, image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = depth_preprocess_image(img_rgb)
    with torch.no_grad():
        depth_tensor = model(input_tensor)
        depth_map = postprocess_depth_map(depth_tensor, (img_rgb.shape[1], img_rgb.shape[0]))
    return depth_map

#Function to extract depth from bounding boxes
def extract_depth_from_boxes(boxes, depth_map):
    object_depths = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        if x1 < 0 or y1 < 0 or x2 > depth_map.shape[1] or y2 > depth_map.shape[0]:
            print(f"Invalid bouding box: {box}")
            object_depths.append(float('nan'))
            continue
        object_depth_map = depth_map[y1:y2, x1:x2]
        if object_depth_map.size == 0:
            print(f"Empty depth map for bounding box: {box}")
            object_depths.append(float('nan'))
            continue
        median_depth = np.median(object_depth_map) * 1000.0
        object_depths.append(median_depth)
    return object_depths

#Function to extract bounding boxes and depth
def extract_bounding_boxes_and_depth(detected_boxes, detected_labels, depths):
    objects = []
    for i, box in enumerate(detected_boxes):
        x1, y1, x2, y2 = map(int, box)
        obj = {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'class': detected_labels[i],
            'depth': depths[i]
        }
        objects.append(obj)
    return objects

def find_center_coordinates_with_depth(obj):
    x1 = obj['x1']
    y1 = obj['y1']
    x2 = obj['x2']
    y2 = obj['y2']
    depth = obj['depth']
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y, depth