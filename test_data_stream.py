# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2

# class ImageSubscriber(Node):
#     def __init__(self):
#         super().__init__('image_subscriber')
#         self.subscription = self.create_subscription(
#             Image,
#             'aria/rgb_image',
#             self.listener_callback,
#             10) 
#         self.bridge = CvBridge()

#     def listener_callback(self, msg):
#         cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
#         cv2.imshow('Received Image', cv_image)
#         cv2.waitKey(1)

# def main(args=None):
#     rclpy.init(args=args)
#     node = ImageSubscriber()
#     rclpy.spin(node)
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2
# import time

# class ImageSubscriber(Node):
#     def __init__(self):
#         super().__init__('image_subscriber')
#         self.subscription = self.create_subscription(
#             Image,
#             'aria/rgb_image',
#             self.listener_callback,
#             10) 
#         self.bridge = CvBridge()
#         self.last_time = time.time()

#     def listener_callback(self, msg):
#         curent_time = time.time()
#         fps = 1.0 / (curent_time - self.last_time)
#         self.last_time = curent_time

#         cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
#         cv2.putText(cv_image, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#         cv2.imshow('Received Image', cv_image)
#         cv2.waitKey(1)

# def main(args=None):
#     rclpy.init(args=args)
#     node = ImageSubscriber()
#     rclpy.spin(node)
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.rgb_subscription = self.create_subscription(
            Image,
            'aria/rgb_image',
            self.rgb_image_callback,
            10) 
        self.slam_subscription = self.create_subscription(
            Image,
            'aria/slam_image',
            self.slam_image_callback,
            10)
        # self.eye_camera_subscription = self.create_subscription(
        #     Image,
        #     'aria/eye_camera_image',
        #     self.eyetrack_image_callback,
        #     10)
        self.bridge = CvBridge()
        self.last_time = time.time()

    def rgb_image_callback(self, msg):
        curent_time = time.time()
        fps = 1.0 / (curent_time - self.last_time)
        self.last_time = curent_time

        cv_rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        cv2.putText(cv_rgb_image, f'RGB FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('RGB Image', cv_rgb_image)
        cv2.waitKey(1)

    def slam_image_callback(self, msg):
        curent_time = time.time()
        fps = 1.0 / (curent_time - self.last_time)
        self.last_time = curent_time

        cv_slam_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        cv2.putText(cv_slam_image, f'SLAM FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('SLAM Image', cv_slam_image)
        cv2.waitKey(1)

    # def eyetrack_image_callback(self, msg):
    #     curent_time = time.time()
    #     fps = 1.0 / (curent_time - self.last_time)
    #     self.last_time = curent_time

    #     cv_eye_camera_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
    #     cv2.putText(cv_eye_camera_image, f'Eye Camera FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    #     cv2.imshow('Eye Camera Image', cv_eye_camera_image)
    #     cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()