import os
import time
import base64
import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, NavSatFix
from std_msgs.msg import String
import requests
import cv2
from cv_bridge import CvBridge
import datetime
import yaml
from ament_index_python.packages import get_package_share_directory

class VLMPub(Node):
    def __init__(self):
        super().__init__('vlm_pub')

        # 加载配置
        config_path = os.path.join(
            get_package_share_directory('hqiit_vlm'),
            'config',
            'config.yaml'
        )
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.get_logger().info(f"配置文件加载成功: {config_path}")
        except Exception as e:
            self.get_logger().error(f"配置文件加载失败: {e}")
            rclpy.shutdown()
            return

        # 初始化变量
        self.bridge = CvBridge()
        self.vlm_url = self.config['api']['dashscope']['url']
        self.last_gps = None
        self.images_buffer = []  # 存储图片的缓冲区
        
        # 创建日志目录
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_base_dir = os.path.join(self.config['logging']['base_dir'], timestamp)
        self.images_dir = os.path.join(self.log_base_dir, self.config['logging']['images_dir'])
        self.metadata_dir = os.path.join(self.log_base_dir, self.config['logging']['metadata_dir'])
        
        for directory in [self.log_base_dir, self.images_dir, self.metadata_dir]:
            os.makedirs(directory, exist_ok=True)

        # 创建订阅者
        self.left_camera_sub = self.create_subscription(
            Image,
            self.config['topics']['subscribers']['camera_left'],
            self.left_image_callback,
            10
        )
        
        self.right_camera_sub = self.create_subscription(
            Image,
            self.config['topics']['subscribers']['camera_right'],
            self.right_image_callback,
            10
        )
        
        self.gps_sub = self.create_subscription(
            NavSatFix,
            self.config['topics']['subscribers']['gps'],
            self.gps_callback,
            10
        )

        # 创建发布者
        self.publisher_ = self.create_publisher(
            String, 
            self.config['topics']['publishers']['output'],
            10
        )

        # 创建定时器
        self.image_save_timer = self.create_timer(
            self.config['timing']['image_save_interval'],
            self.save_images_callback
        )
        
        self.vlm_process_timer = self.create_timer(
            self.config['timing']['vlm_process_interval'],
            self.process_batch
        )

        self.get_logger().info("VLM节点初始化完成")

    def left_image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_left_image = cv_image
        except Exception as e:
            self.get_logger().error(f"左相机图像处理错误: {e}")

    def right_image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_right_image = cv_image
        except Exception as e:
            self.get_logger().error(f"右相机图像处理错误: {e}")

    def gps_callback(self, msg):
        self.last_gps = msg

    # ...existing code...

    def save_images_callback(self):
        """每5秒保存一次图片"""
        if not hasattr(self, 'latest_left_image') or not hasattr(self, 'latest_right_image'):
            self.get_logger().warning("未收到相机图像，跳过保存")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 保存左相机图片
            left_path = os.path.join(self.images_dir, f"{timestamp}_left.jpg")
            cv2.imwrite(left_path, self.latest_left_image)
            
            # 保存右相机图片
            right_path = os.path.join(self.images_dir, f"{timestamp}_right.jpg")
            cv2.imwrite(right_path, self.latest_right_image)
            
            # 将图片信息添加到缓冲区
            self.images_buffer.append({
                'timestamp': timestamp,
                'left_path': left_path,
                'right_path': right_path,
                'gps': {
                    'latitude': self.last_gps.latitude if self.last_gps else None,
                    'longitude': self.last_gps.longitude if self.last_gps else None,
                    'altitude': self.last_gps.altitude if self.last_gps else None
                }
            })
            
            self.get_logger().info(f"成功保存图片对: {timestamp}")
            self.get_logger().info(f"GPS位置: 经度={self.last_gps.longitude if self.last_gps else 'None'}, "
                                f"纬度={self.last_gps.latitude if self.last_gps else 'None'}")
        except Exception as e:
            self.get_logger().error(f"保存图片失败: {str(e)}")
            
    def process_batch(self):
        """每30秒处理一批图片"""
        if not self.images_buffer:
            self.get_logger().warning("没有需要处理的图片，跳过此次处理")
            return

        process_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.get_logger().info(f"开始处理批次 {process_timestamp}")
        self.get_logger().info(f"本批次包含 {len(self.images_buffer)} 对图片")
        
        # 准备批次数据
        batch_data = {
            'timestamp': process_timestamp,
            'gps': {
                'latitude': self.last_gps.latitude if self.last_gps else None,
                'longitude': self.last_gps.longitude if self.last_gps else None
            },
            'images': self.images_buffer
        }
        
        # 记录开始时间
        start_time = time.time()
        
        # 调用VLM处理
        result = self.process_with_vlm(batch_data)
        
        # 计算处理时间
        end_time = time.time()
        processing_time = end_time - start_time
        
        if result:
            self.get_logger().info(f"VLM处理成功，耗时: {processing_time:.2f}秒")
        else:
            self.get_logger().error("VLM处理失败")
            
        # 保存处理结果
        metadata = {
            'request_timestamp': process_timestamp,
            'batch_data': batch_data,
            'vlm_response': result,
            'processing_metrics': {
                'total_time': processing_time,
                'images_count': len(self.images_buffer)
            }
        }
        
        # 保存元数据
        try:
            metadata_path = os.path.join(self.metadata_dir, f"{process_timestamp}.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            self.get_logger().info(f"元数据已保存到: {metadata_path}")
        except Exception as e:
            self.get_logger().error(f"保存元数据失败: {str(e)}")
        
        # 清空缓冲区
        self.images_buffer = []
        
        # 发布结果
        if result:
            self.publisher_.publish(String(data=result))
            self.get_logger().info("结果已发布到话题")

    def process_with_vlm(self, batch_data):
        """处理图片批次"""
        try:
            self.get_logger().info("开始处理VLM请求...")
            
            # 构建提示文本
            prompt_text = (
                f"当前时间: {batch_data['timestamp']}\n"
                f"GPS位置: 经度 {batch_data['gps']['longitude']}, 纬度 {batch_data['gps']['latitude']}\n"
                f"请分析这段时间内的场景变化。"
            )

            # 准备图片数据
            images_data = []
            for img_info in batch_data['images']:
                for side in ['left', 'right']:
                    img_path = img_info[f'{side}_path']
                    self.get_logger().debug(f"处理图片: {img_path}")
                    data_uri = self.jpg_to_data_uri(img_path)
                    if data_uri:
                        images_data.append({
                            "image_url": {
                                "detail": "high",
                                "url": data_uri
                            },
                            "type": "image_url"
                        })
                    else:
                        self.get_logger().error(f"图片转换失败: {img_path}")

            if not images_data:
                self.get_logger().error("没有有效的图片数据")
                return None

            self.get_logger().info(f"准备发送 {len(images_data)} 张图片到VLM服务")

            # 构建请求payload
            payload = {
                "model": "qwen-vl-max-2025-01-25",
                "messages": [
                    {
                        "role": "user",
                        "content": images_data + [{"type": "text", "text": prompt_text}]
                    }
                ]
            }

            headers = {
                "Authorization": f"Bearer {self.config['api']['dashscope']['key']}",
                "Content-Type": "application/json"
            }

            # 发送请求
            self.get_logger().info("正在发送VLM请求...")
            response = requests.post(self.vlm_url, json=payload, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                self.get_logger().info("VLM请求成功")
                return result['choices'][0]['message']['content']
            else:
                self.get_logger().error(f"VLM请求失败: HTTP {response.status_code}")
                self.get_logger().error(f"错误信息: {response.text}")
                return None

        except Exception as e:
            self.get_logger().error(f"VLM处理发生错误: {str(e)}")
            return None

# ...existing code...

    def jpg_to_data_uri(self, image_path):
        """将JPG图像转换为Data URI格式"""
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                base64_encoded = base64.b64encode(image_data).decode('utf-8')
                return f"data:image/jpeg;base64,{base64_encoded}"
        except Exception as e:
            self.get_logger().error(f"图像转换错误: {e}")
            return None

def main(args=None):
    rclpy.init(args=args)
    vlm_pub = VLMPub()
    rclpy.spin(vlm_pub)
    vlm_pub.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()