import argparse
import numpy as np
import sys
import cv2
import time
import os
import subprocess
import aria.sdk as aria
from projectaria_tools.core.sensor_data import ImageDataRecord
from ultralytics import YOLO

import torch
import torchvision.transforms as transforms
from torchvision.transforms import Compose

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        dest="streaming_interface",
        type=str,
        required=True,
        help="Type of interface to use for streaming. Options are usb or wifi.",
        choices=["usb", "wifi"],
    )
    parser.add_argument(
        "--profile",
        dest="profile_name",
        type=str,
        default="profile18",
        required=False,
        help="Profile to be used for streaming.",
    )
    parser.add_argument(
        "--device_ip", help="IP address to connect to the device over wifi"
    )

    return parser.parse_args()

class StreamingClientObserver():
    def __init__(self):
        self.images = {}

    def on_image_received(self, image: np.array, record: ImageDataRecord):
        self.images[record.camera_id] = image

def quit_keypress():
    key = cv2.waitKey(1)
    # Press ESC, 'q'
    return key == 27 or key == ord("q")

def update_iptables() -> None:
    """
    Update firewall to permit incoming UDP connections for DDS
    """
    update_iptables_cmd = [
        "sudo",
        "iptables",
        "-A",
        "INPUT",
        "-p",
        "udp",
        "-m",
        "udp",
        "--dport",
        "7000:8000",
        "-j",
        "ACCEPT",
    ]
    print("Running the following command to update iptables:")
    print(update_iptables_cmd)
    subprocess.run(update_iptables_cmd)

# Load a model
model = YOLO("yolo11n.pt")

# Detect objects using YOLO
def detect_objects(img, net, output_layers):
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return outputs

def draw_labels_and_boxes(img, boxes, confidences, class_ids, classes):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 1, color, 2)
    return img

# Process the detections
def process_detections(results, width, height):
    boxes = []
    confidences = []
    class_ids = []
    for result in results:
        for detection in result.boxes:
            x1, y1, x2, y2 = detection.xyxy[0]
            confidence = detection.conf[0]
            class_id = detection.cls[0]
            if confidence > 0.5:  # Confidence threshold
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))
    return boxes, confidences, class_ids

def device_stream(args):
    if sys.platform.startswith("linux"):
        update_iptables()
    # Set debug level
    aria.set_log_level(aria.Level.Info)
    # Create DeviceClient instance, setting the IP address if specified
    device_client = aria.DeviceClient()
    client_config = aria.DeviceClientConfig()
    if args.device_ip:
        client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)
    # Connect to the device
    device = device_client.connect()
    # Retrieve the streaming_manager and streaming_client
    streaming_manager = device.streaming_manager
    streaming_client = streaming_manager.streaming_client
    # Set custom config for streaming
    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = args.profile_name
    # Streaming type
    if args.streaming_interface == "usb":
        streaming_config.streaming_interface = aria.StreamingInterface.Usb
    # Use ephemeral streaming certificates
    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config
    # Start streaming
    streaming_manager.start_streaming()
    # Get streaming state
    streaming_state = streaming_manager.streaming_state
    print(f"Streaming state: {streaming_state}")
    return streaming_manager, streaming_client, device_client, device


def device_subscribe(streaming_client):
    # Configure subscription
    config = streaming_client.subscription_config
    config.subscriber_data_type = (aria.StreamingDataType.Rgb)
    # Take most recent frame
    config.message_queue_size[aria.StreamingDataType.Rgb] = 1
    # Set the security options
    # @note we need to specify the use of ephemeral certs as this sample app assumes
    # aria-cli was started using the --use-ephemeral-certs flag
    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config
    # Set the observer
    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)
    # Start listening
    print("Start listening to image data")
    streaming_client.subscribe()
    return observer

depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small") #DPT_Hybrid #MiDaS_small #DPT_Large
depth_model.eval()

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

def postprocess_depth(depth_tensor, original_size):
    depth_map = depth_tensor.squeeze().cpu().numpy()
    depth_map = cv2.resize(depth_map, original_size)
    depth_map = depth_map / depth_map.max()
    return depth_map

def extract_depth_map(model, image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = depth_preprocess_image(img_rgb)
    with torch.no_grad():
        depth_tensor = model(input_tensor)
        depth_map = postprocess_depth(depth_tensor, (img_rgb.shape[1], img_rgb.shape[0]))
    return depth_map

# Function to extract depth from bounding boxes
def extract_depth_from_boxes(boxes, depth_map):
    object_depths = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        object_depth_map = depth_map[y1:y2, x1:x2]
        median_depth = np.mean(object_depth_map) * 1000.0
        object_depths.append(median_depth)
    return object_depths

# Function to extract bounding boxes and depth
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

def find_center_coordinates_with_depth(box):
    x1, y1, x2, y2, depth = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y, depth

# Main function
def main():
    args = parse_args()
    streaming_manager, streaming_client, device_client, device = device_stream(args)
    observer = device_subscribe(streaming_client)

    rgb_window = "Aria RGB"
    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_window, 1080, 1080)
    cv2.moveWindow(rgb_window, 50, 50)

    while not quit_keypress():
        if aria.CameraId.Rgb in observer.images:
            rgb_image = np.rot90(observer.images[aria.CameraId.Rgb], -1)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb_image.shape

            # YOLO object detection
            results = model(rgb_image)
            boxes, confidences, class_ids = process_detections(results, width, height)
            img_with_boxes = draw_labels_and_boxes(rgb_image.copy(), boxes, confidences, class_ids, model.names)
            cv2.imshow(rgb_window, img_with_boxes)
            del observer.images[aria.CameraId.Rgb]
            
            # Depth estimation
            depth_map = extract_depth_map(depth_model, rgb_image)
            depth_boxes = extract_depth_from_boxes(boxes, depth_map)
            detected_objects = extract_bounding_boxes_and_depth(boxes, class_ids, depth_boxes)
            for obj in detected_objects:
                print(obj)
                
            # Center coordinates with depth
            for obj in detected_objects:
                center_x, center_y, depth = find_center_coordinates_with_depth(obj)
                print(f"Center coordinates: ({center_x}, {center_y}), Depth: {depth} mm")
    cv2.destroyAllWindows()
    

    print("Stop listening to image data")
    streaming_client.unsubscribe()
    streaming_manager.stop_streaming()
    device_client.disconnect(device)

main()
