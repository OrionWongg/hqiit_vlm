import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, 'camera_image', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)  # 打开摄像头
        self.frame_count = 0  # 添加计数器

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            # 设置header的时间戳为当前时间
            msg.header.stamp = self.get_clock().now().to_msg()
            # 设置frame_id为计数
            self.frame_count += 1
            msg.header.frame_id = str(self.frame_count)
            
            self.publisher_.publish(msg)
            self.get_logger().info(f'Publishing camera image #{self.frame_count}')

def main(args=None):
    rclpy.init(args=args)
    camera_publisher = CameraPublisher()
    rclpy.spin(camera_publisher)
    camera_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from std_msgs.msg import String
# import cv2
# from cv_bridge import CvBridge

# class CameraPublisher(Node):
#     def __init__(self):
#         super().__init__('camera_publisher')
#         self.publisher_ = self.create_publisher(Image, 'camera_image', 10)
#         self.bridge = CvBridge()
#         self.cap = None  # 摄像头对象初始化为None
#         self.frame_count = 0  # 添加帧计数器
#         self.subscription = self.create_subscription(
#             String,
#             '/vlm_output',
#             self.vlm_callback,
#             10
#         )

#     def vlm_callback(self, msg):
#         if msg.data == "camera_pub":
#             self.get_logger().info('Received camera_pub command, capturing image...')
#             cap = cv2.VideoCapture(0)  # 每次都重新打开摄像头
#             ret, frame = cap.read()
#             cap.release()  # 采集完立即释放
#             if ret:
#                 img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                
#                 # 设置header的时间戳为当前时间
#                 img_msg.header.stamp = self.get_clock().now().to_msg()
                
#                 # 设置frame_id为计数
#                 self.frame_count += 1
#                 img_msg.header.frame_id = str(self.frame_count)
                
#                 self.publisher_.publish(img_msg)
#                 self.get_logger().info(f'Published camera image #{self.frame_count}')
#             else:
#                 self.get_logger().error('Failed to capture image from camera')
           
# def main(args=None):
#     rclpy.init(args=args)
#     camera_publisher = CameraPublisher()
#     rclpy.spin(camera_publisher)
#     camera_publisher.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()