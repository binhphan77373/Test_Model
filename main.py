import cv2
import time
 
from depth_estimation import extract_depth_map, extract_depth_from_boxes, extract_bounding_boxes_and_depth, find_center_coordinates_with_depth
from load_model import model, depth_model
from object_detection import process_detections, draw_labels_and_boxes

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def quit_keypress():
    key = cv2.waitKey(1)
    return key == 27 or key == ord('q')

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            'aria/rgb_image',
            self.listener_callback,
            10) 
        self.bridge = CvBridge()
        self.rgb_image = None

    def listener_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()

    rgb_window = "Aria RGB Image"
    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_window, 640, 480)
    cv2.moveWindow(rgb_window, 50, 50)

    last_time = time.time()

    while rclpy.ok() and not quit_keypress():
        rclpy.spin_once(node)
        if node.rgb_image is not None:
            current_time = time.time()
            fps = 1.0 / (current_time - last_time)
            last_time = current_time

            rgb_image = node.rgb_image
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb_image.shape

            # Object Detection
            results = model(rgb_image)
            boxes, confidences, class_ids = process_detections(results, height, width)
            img_with_boxes = draw_labels_and_boxes(rgb_image.copy(), boxes, confidences, class_ids, model.names)
            cv2.imshow(rgb_window, img_with_boxes)

            # Depth Estimation
            depth_map = extract_depth_map(depth_model, rgb_image)
            depth_boxes = extract_depth_from_boxes(boxes, depth_map)
            detected_objects = extract_bounding_boxes_and_depth(boxes, class_ids, depth_boxes)
            for obj in detected_objects:
                print(obj)

            # Center coordinates with depth
            for obj in detected_objects:
                center_x, center_y, depth = find_center_coordinates_with_depth(obj)
                print(f"Center: ({center_x}, {center_y}), Depth: {depth:.2f} mm")
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()