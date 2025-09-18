import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class EmotionPublisher(Node):
    def __init__(self):
        super().__init__('emotion_publisher')
        self.publisher_ = self.create_publisher(String, 'detected_objects', 10)
        self.timer = self.create_timer(3.0, self.timer_callback)
        self.emotions = ["idle", "happy", "sad", "angry", "confused", "surprised","love","sleepy"]
        self.index = 0

    def timer_callback(self):
        msg = String()
        msg.data = self.emotions[self.index]
        self.publisher_.publish(msg)
        self.get_logger().info(f'发布表情: {msg.data}')
        self.index = (self.index + 1) % len(self.emotions)

def main(args=None):
    rclpy.init(args=args)
    publisher = EmotionPublisher()
    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        pass
    finally:
        publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()