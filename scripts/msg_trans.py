import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
from collections import deque
from geometry_msgs.msg import Twist 
from datetime import datetime

class CommandRelayNode(Node):
    def __init__(self):
        super().__init__('command_relay_node')
        
        # 1. 订阅原始的VLM输出话题
        self.subscription = self.create_subscription(
            String,
            '/vlm_output',  # 订阅 /vlm_output 话题
            self.listener_callback,
            10)
        self.get_logger().info("正在订阅 '/vlm_output' 话题。")

        # 2. 发布器，用于发布给机器人运动控制节点
        self.publisher_ = self.create_publisher(String, 'motion_commands', 10)
        # 3. 添加Twist消息发布器，用于发布cmd_vel
        self.twist_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info("正在发布到 '/cmd_vel' 话题。")
        
        # 4. 定义发布频率 (Hz)
        self.cmd_vel_publish_rate = 10.0  # 移动和转向命令的发布频率
        self.altitude_publish_rate = 1.0  # 升高和降低命令的发布频率
        self.timer = self.create_timer(0.1, self.timer_callback)  # 100ms基础检查定时器
        self.get_logger().info(f"移动/转向命令以 {self.cmd_vel_publish_rate} Hz 的频率发布，升高/降低命令以 {self.altitude_publish_rate} Hz 的频率发布。")
        
        # 5. 添加超时机制
        self.last_received_time = time.time()
        self.timeout_threshold = 1.0  # 1秒超时

        self.command_queue = deque()  # 存放动作的队列
        self.current_command = None
        self.command_start_time = None  # 当前动作开始的时间
        self.command_duration = 1.0     # 动作持续时间，默认1秒
        self.pause_until = None       # 暂停到的时间点
        
        # 命令执行计数
        self.publish_count = 0
        
        # 活动定时器
        self.active_timer = None

    def listener_callback(self, msg):
        received_message = msg.data
        self.last_received_time = time.time()
        self.get_logger().info(f"接收到VLM消息: '{received_message}'")

        # 提取可能的动作
        actions = []
        if "前进" in received_message: actions.append("前进")
        if "后退" in received_message: actions.append("后退")
        if "左转" in received_message: actions.append("左转")
        if "右转" in received_message: actions.append("右转")
        if "升高" in received_message: actions.append("升高")
        if "降低" in received_message: actions.append("降低")

        if actions:
            self.command_queue.extend(actions)  # 按顺序加入队列
            # 清除当前活动定时器（如果有）
            if self.active_timer:
                self.active_timer.cancel()
                self.active_timer = None
                # 发送停止指令
                stop_twist = Twist()
                self.twist_publisher.publish(stop_twist)
                self.get_logger().info("收到新命令，发送停止指令")
                self.current_command = None
        else:
            # 原来的非连续命令处理
            msg_out = String()
            msg_out.data = received_message
            self.publisher_.publish(msg_out)
            self.command_queue.clear()
            self.current_command = None
            self.pause_until = None
            if self.active_timer:
                self.active_timer.cancel()
                self.active_timer = None
                # 发送停止指令
                stop_twist = Twist()
                self.twist_publisher.publish(stop_twist)
                self.get_logger().info("收到非动作命令，发送停止指令")

    def timer_callback(self):
        current_time = time.time()

        # 如果在暂停期，不做动作
        if self.pause_until and current_time < self.pause_until:
            return

        # 如果没有当前动作，取下一个
        if not self.current_command and self.command_queue:
            self.current_command = self.command_queue.popleft()
            self.command_start_time = current_time
            self.publish_count = 0  # 重置发布计数
            self.get_logger().info(f"开始执行动作: {self.current_command}")
            
            # 创建特定频率的发布定时器
            if self.current_command in ["前进", "后退", "左转", "右转"]:
                # 移动和转向命令使用较高的频率
                interval = 1.0 / self.cmd_vel_publish_rate
                # 立即发布第一条消息
                self.publish_cmd_vel()
                # 然后设置定时器定期发布
                self.active_timer = self.create_timer(interval, self.publish_cmd_vel)
            elif self.current_command in ["升高", "降低"]:
                # 升高和降低命令使用较低的频率
                interval = 1.0 / self.altitude_publish_rate
                # 立即发布第一条消息
                self.publish_cmd_vel()
                # 然后设置定时器定期发布
                self.active_timer = self.create_timer(interval, self.publish_cmd_vel)

    def publish_cmd_vel(self):
        # 如果没有当前命令，停止发布
        if not self.current_command:
            if self.active_timer:
                self.active_timer.cancel()
                self.active_timer = None
            return
            
        current_time = time.time()
        elapsed_time = current_time - self.command_start_time
        
        # 检查是否达到命令持续时间
        if elapsed_time >= self.command_duration:
            # 命令结束，发送停止指令
            self.get_logger().info(f"动作完成: {self.current_command}，共发布 {self.publish_count} 条消息")
            
            # 发送停止指令（全部速度归零）
            stop_twist = Twist()
            self.twist_publisher.publish(stop_twist)
            self.get_logger().info("发送停止指令")
            
            # 取消定时器
            if self.active_timer:
                self.active_timer.cancel()
                self.active_timer = None
            
            # 重置状态
            self.current_command = None
            self.pause_until = current_time + 1.0  # 停 1 秒再执行下一个
            return
            
        # 发布原始命令消息
        msg = String()
        msg.data = self.current_command
        self.publisher_.publish(msg)

        # 发布Twist消息到/cmd_vel
        twist_msg = Twist()
        if self.current_command == "前进":
            twist_msg.linear.x = 1.0
        elif self.current_command == "后退":
            twist_msg.linear.x = -1.0
        elif self.current_command == "左转":
            twist_msg.angular.z = 1.0
        elif self.current_command == "右转":
            twist_msg.angular.z = -1.0
        elif self.current_command == "升高":
            twist_msg.linear.z = 1.0
        elif self.current_command == "降低":
            twist_msg.linear.z = -1.0
        
        self.twist_publisher.publish(twist_msg)
        self.publish_count += 1
        
        self.get_logger().info(
            f"[{elapsed_time:.2f}s] 发布第 {self.publish_count} 条 {self.current_command} 命令: "
            f"linear={{x: {twist_msg.linear.x}, y: {twist_msg.linear.y}, z: {twist_msg.linear.z}}}, "
            f"angular={{x: {twist_msg.angular.x}, y: {twist_msg.angular.y}, z: {twist_msg.angular.z}}}"
        )

def main(args=None):
    rclpy.init(args=args)
    command_relay_node = CommandRelayNode()
    try:
        rclpy.spin(command_relay_node)
    except KeyboardInterrupt:
        command_relay_node.get_logger().info("命令中继节点已由用户停止。")
    finally:
        command_relay_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()