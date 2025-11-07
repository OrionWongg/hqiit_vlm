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
import numpy as np
# 导入YOLO消息类型
from yolo_msgs.msg import DetectionArray


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
        self.qwen_url = self.config['api']['dashscope']['url']
        # DINO API 相关变量已移除
        self.last_gps = None

        # 创建日志目录
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_base_dir = os.path.join(self.config['logging']['base_dir'], timestamp)
        os.makedirs(self.log_base_dir, exist_ok=True)

        self.current_batch_dir = None
        self.current_images_dir = None

        # 动态加载相机配置
        camera_topics = self.config['topics']['subscribers']
        self.camera_topics = {}
        if 'camera_left' in camera_topics and camera_topics['camera_left']:
            self.camera_topics['left'] = camera_topics['camera_left']
        if 'camera_right' in camera_topics and camera_topics['camera_right']:
            self.camera_topics['right'] = camera_topics['camera_right']

        if not self.camera_topics:
            self.get_logger().error("❌ 未配置任何相机话题，节点退出")
            rclpy.shutdown()
            return

        # 创建相机订阅者
        self.image_data = {}
        for side, topic in self.camera_topics.items():
            self.get_logger().info(f"订阅{side}相机话题: {topic}")
            self.create_subscription(Image, topic, lambda msg, s=side: self.image_callback(msg, s), 10)

        # 创建YOLO订阅者
        self.yolo_data = {'left': None, 'right': None} # 用于存储最新的YOLO数据
        yolo_topics_config = self.config['topics']['subscribers']
        self.yolo_topics = {}
        if 'yolo_left' in yolo_topics_config and yolo_topics_config['yolo_left']:
            self.yolo_topics['left'] = yolo_topics_config['yolo_left']
        if 'yolo_right' in yolo_topics_config and yolo_topics_config['yolo_right']:
            self.yolo_topics['right'] = yolo_topics_config['yolo_right']

        if not self.yolo_topics:
            self.get_logger().warning("⚠️ 未配置YOLO话题，将无法获取检测结果")
        else:
            for side, topic in self.yolo_topics.items():
                self.get_logger().info(f"订阅{side} YOLO话题: {topic}")
                self.create_subscription(DetectionArray, topic, lambda msg, s=side: self.yolo_callback(msg, s), 10)


        # 创建GPS订阅者
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
        self.collection_timer = self.create_timer(
            self.config['timing']['collection_interval'],
            self.start_collection_cycle
        )

        self.get_logger().info("✅ VLM节点初始化完成")
        self.get_logger().info(f"将在 {self.config['timing']['collection_interval']} 秒后开始第一次采集")

    # ---------------------- 订阅回调 ----------------------

    def image_callback(self, msg, side):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_data[side] = cv_image
        except Exception as e:
            self.get_logger().error(f"{side}相机图像处理错误: {e}")

    def yolo_callback(self, msg, side):
        """存储YOLO检测结果"""
        self.yolo_data[side] = msg

    def gps_callback(self, msg):
        self.last_gps = msg

    # ---------------------- 主采集流程 ----------------------

    def start_collection_cycle(self):
        """开始一个完整的采集-处理周期"""
        self.collection_timer.cancel()

        if not self.image_data:
            self.get_logger().warning("⚠️ 未收到相机图像，跳过本次采集")
            self.schedule_next_collection()
            return

        try:
            batch_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_batch_dir = os.path.join(self.log_base_dir, batch_timestamp)
            self.current_images_dir = os.path.join(self.current_batch_dir, 'images')
            os.makedirs(self.current_images_dir, exist_ok=True)

            self.get_logger().info(f"\n{'='*60}")
            self.get_logger().info(f"开始新的采集周期 - 批次 {batch_timestamp}")
            self.get_logger().info(f"{'='*60}")

            total_start_time = time.time()

            # 1. 保存图像
            self.get_logger().info("步骤1: 保存图像...")
            image_paths = {}
            save_start = time.time()
            for side, img in self.image_data.items():
                path = os.path.join(self.current_images_dir, f"{batch_timestamp}_{side}.jpg")
                cv2.imwrite(path, img)
                image_paths[side] = path
            save_time = time.time() - save_start
            self.get_logger().info(f"✓ 保存了 {len(image_paths)} 张相机图像，耗时 {save_time:.2f}秒")

            # 2. 获取 YOLO 目标检测结果
            self.get_logger().info("\n步骤2: 获取YOLO目标检测结果...")
            yolo_results = {}
            yolo_start = time.time()
            
            # 从self.yolo_data中获取最新的数据
            for side in self.camera_topics.keys(): 
                if self.yolo_data.get(side):
                    yolo_results[side] = self.yolo_data[side]
                    # 清除数据，确保下次是新的
                    self.yolo_data[side] = None 
                else:
                    self.get_logger().warning(f"  未收到 {side} 侧的YOLO数据")

            yolo_time = time.time() - yolo_start
            self.get_logger().info(f"✓ YOLO数据检索完成，耗时 {yolo_time:.2f}秒")

            if not yolo_results:
                self.get_logger().warning("⚠️ 未收到任何有效YOLO结果，跳过此周期后续步骤")
                # 即使没有YOLO结果，也可能需要继续处理（例如只分析图像），这里我们选择继续
                # 如果没有YOLO结果就必须停止，则取消下面几行注释
                # self.schedule_next_collection()
                # return

            # 2.5 绘制边界框
            self.get_logger().info("\n步骤2.5: 绘制检测边界框...")
            bbox_paths = {}
            bbox_start = time.time()
            for side, result_msg in yolo_results.items():
                if side in image_paths: # 确保有对应的图像
                    bbox_path = self.draw_bboxes_on_image(image_paths[side], result_msg, side, batch_timestamp)
                    if bbox_path:
                        bbox_paths[side] = bbox_path
                else:
                    self.get_logger().warning(f"  有 {side} 侧YOLO数据，但没有 {side} 侧图像，无法绘制")
            bbox_time = time.time() - bbox_start
            self.get_logger().info(f"✓ 边界框绘制完成，耗时 {bbox_time:.2f}秒")

            # 3. Qwen VLM 场景分析
            self.get_logger().info("\n步骤3: Qwen VLM场景分析...")
            qwen_start = time.time()
            qwen_result, token_count = self.process_with_qwen(batch_timestamp, image_paths, yolo_results)
            qwen_time = time.time() - qwen_start
            if qwen_result:
                self.get_logger().info(f"✓ Qwen VLM分析完成，耗时 {qwen_time:.2f}秒")
            else:
                self.get_logger().error("✗ Qwen VLM分析失败")

            total_time = time.time() - total_start_time

            # 4. 保存元数据
            self.get_logger().info("\n步骤4: 保存元数据...")

            # 转换YOLO结果为可序列化的字典
            yolo_results_serializable = {}
            for side, msg in yolo_results.items():
                yolo_results_serializable[side] = self.convert_yolo_to_dict(msg)

            metadata = {
                'timestamp': batch_timestamp,
                'gps': {
                    'latitude': self.last_gps.latitude if self.last_gps else None,
                    'longitude': self.last_gps.longitude if self.last_gps else None,
                    'altitude': self.last_gps.altitude if self.last_gps else None
                },
                'images': image_paths,
                'bbox_images': bbox_paths,
                'processing_time': {
                    'total': f"{total_time:.2f}",
                    'save': f"{save_time:.2f}",
                    'yolo_retrieval': f"{yolo_time:.2f}", # 替换dino
                    'bbox': f"{bbox_time:.2f}",
                    'qwen_vlm': f"{qwen_time:.2f}"
                },
                'yolo_results': yolo_results_serializable, # 替换dino
                'qwen_output': qwen_result or "处理失败",
                'token_count': token_count
            }

            metadata_path = os.path.join(self.current_batch_dir, 'metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            self.get_logger().info(f"✓ 元数据已保存: {metadata_path}")

            print("\n" + "="*60)
            print(f"批次 {batch_timestamp} 处理完成")
            print("="*60)
            print(f"总耗时: {total_time:.2f}秒")
            print(f"图像保存: {save_time:.2f}秒, YOLO检索: {yolo_time:.2f}秒, Qwen: {qwen_time:.2f}秒")
            print(f"消耗Token数: {token_count}")
            print("Qwen输出:")
            print(qwen_result)
            print("="*60)
            print(f"等待 {self.config['timing']['collection_interval']} 秒后进行下一次采集...\n")

        except Exception as e:
            self.get_logger().error(f"采集处理错误: {e}")
        finally:
            self.schedule_next_collection()

    # ---------------------- 工具函数 ----------------------

    def schedule_next_collection(self):
        self.collection_timer = self.create_timer(
            self.config['timing']['collection_interval'],
            self.start_collection_cycle
        )

    # 移除 process_with_grounding_dino 函数

    def convert_yolo_to_dict(self, yolo_msg):
        """将 DetectionArray 消息转换为可序列化的字典列表 (修正版)"""
        if not yolo_msg:
            return []
        
        output_list = []
        for det in yolo_msg.detections:
            output_list.append({
                'class_name': det.class_name,
                'score': det.score,
                # 修正: 使用 det.bbox.center.position.x 
                'bbox_center_x': det.bbox.center.position.x,
                # 修正: 使用 det.bbox.center.position.y
                'bbox_center_y': det.bbox.center.position.y,
                'bbox_size_x': det.bbox.size.x,
                'bbox_size_y': det.bbox.size.y,
            })
        return output_list

    def process_with_qwen(self, timestamp, image_paths, yolo_results):
        try:
            prompt_text = f"当前时间: {timestamp}\nGPS位置: 经度 {self.last_gps.longitude if self.last_gps else 'None'}, 纬度 {self.last_gps.latitude if self.last_gps else 'None'}\n\n"
            
            # 使用 helper function 转换 yolo_results
            yolo_serializable = {}
            for side, msg in yolo_results.items():
                yolo_serializable[side] = self.convert_yolo_to_dict(msg)
                prompt_text += f"{side}相机YOLO检测结果:\n{json.dumps(yolo_serializable[side], ensure_ascii=False, indent=2)}\n\n"

            if not yolo_serializable:
                prompt_text += "未收到YOLO检测结果。\n\n"
                
            prompt_text += "请根据时间，gps，图像和检测结果，简要分析当前场景情况。"

            images_data = []
            for side, path in image_paths.items():
                data_uri = self.jpg_to_data_uri(path)
                if data_uri:
                    images_data.append({
                        "image_url": {"detail": "high", "url": data_uri},
                        "type": "image_url"
                    })
            
            if not images_data:
                self.get_logger().error("  没有可用的图像数据发送给Qwen")
                return None, 0

            payload = {
                "model": "qwen3-vl-flash-2025-10-15", # 注意：这个模型可能需要更新
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

            response = requests.post(self.qwen_url, json=payload, headers=headers)
            if response.status_code == 200:
                result = response.json()
                usage = result.get('usage', {})
                token_count = {
                    'total': usage.get('total_tokens', 0),
                    'prompt': usage.get('prompt_tokens', 0),
                    'completion': usage.get('completion_tokens', 0)
                }
                qwen_output = result['choices'][0]['message']['content']

                # 发布 Qwen 分析结果到话题
                output_message = String()
                output_message.data = json.dumps({
                    "timestamp": timestamp,
                    "qwen_output": qwen_output,
                    "token_count": token_count
                }, ensure_ascii=False)
                self.publisher_.publish(output_message)
                self.get_logger().info(f"✓ Qwen 分析结果已发布到话题: {self.config['topics']['publishers']['output']}")

                return qwen_output, token_count
            else:
                self.get_logger().error(f"  Qwen请求失败: HTTP {response.status_code}, 响应: {response.text}")
                return None, 0
        except Exception as e:
            self.get_logger().error(f"  Qwen处理错误: {e}")
            return None, 0

    def draw_bboxes_on_image(self, image_path, yolo_msg, side, batch_timestamp):
        """根据YOLO检测结果(DetectionArray)在图像上绘制边界框 (修正版)"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                self.get_logger().error(f"  [BBox] 无法读取图像: {image_path}")
                return None

            detections = yolo_msg.detections
            if not detections:
                self.get_logger().info(f"  [BBox] {side} 侧无YOLO检测目标，跳过绘制")
                return None # 不绘制，返回None
                
            np.random.seed(42)
            colors = {}
            
            for det in detections:
                category = det.class_name
                score = det.score
                
                # 修正: 使用 det.bbox.center.position.x 和 .y
                center_x = det.bbox.center.position.x
                center_y = det.bbox.center.position.y
                width = det.bbox.size.x
                height = det.bbox.size.y
                
                x1 = int(center_x - width / 2)
                y1 = int(center_y - height / 2)
                x2 = int(center_x + width / 2)
                y2 = int(center_y + height / 2)

                if category not in colors:
                    colors[category] = tuple(np.random.randint(0, 255, 3).tolist())
                color = colors[category]
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{category}: {score:.2f}"
                
                # 确保标签在图像内
                label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                cv2.putText(img, label, (x1, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
            output_path = os.path.join(self.current_images_dir, f"{batch_timestamp}_{side}_bbox.jpg")
            cv2.imwrite(output_path, img)
            return output_path
        except Exception as e:
            self.get_logger().error(f"绘制边界框错误: {e}")
            return None

    def jpg_to_data_uri(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
                return f"data:image/jpeg;base64,{encoded}"
        except Exception as e:
            self.get_logger().error(f"图像转换错误: {e}")
            return None


def main(args=None):
    rclpy.init(args=args)
    node = VLMPub()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()