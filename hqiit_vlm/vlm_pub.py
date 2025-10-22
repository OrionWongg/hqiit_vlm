import os
import time
import base64
import threading
import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image,CompressedImage
from std_msgs.msg import String
import requests
import cv2
from cv_bridge import CvBridge
import datetime
import sys
import yaml
import os.path
from ament_index_python.packages import get_package_share_directory

class VLMPub(Node):
    def __init__(self):
        super().__init__('vlm_pub')  # 修改节点名称为 'vlm_pub'

        # 读取配置文件
        
        config_path = os.path.join(
            get_package_share_directory('hqiit_vlm'),
            'config',
            'config.yaml'
        )
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.get_logger().info(f"成功加载配置文件: {config_path}")
        except Exception as e:
            self.get_logger().error(f"加载配置文件失败: {e}")
            rclpy.shutdown()
            return
        
        self.subscription = None  # 初始化为None，仅在需要时创建订阅
        self.bridge = CvBridge()
        self.vlm_url = self.config['api']['dashscope']['url']

        # 保存接收图像的目录
        self.save_dir = 'received_images'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # VLM日志目录
        self.vlm_log_dir = 'vlm_log'
        if not os.path.exists(self.vlm_log_dir):
            os.makedirs(self.vlm_log_dir)

        self.result_file = 'image_captions.txt'

        # 使用配置文件中的话题名称创建发布者
        self.publisher_ = self.create_publisher(
            String, 
            self.config['topics']['publishers']['output'],
            10
        )

        self.latest_image = None
        self.latest_image_path = None
        self.latest_timestamp = None

        # 创建 vlm_input 话题订阅
        self.vlm_input_subscription = self.create_subscription(
            String,
            self.config['topics']['subscribers']['input'],
            self.vlm_input_callback,
            10
        )
        
        # 创建持续的图像订阅
        self.subscription = self.create_subscription(
            # Image,
            CompressedImage,
            self.config['topics']['subscribers']['camera'],
            self.image_callback,
            10
        )

        # 状态管理
        # 'listening': 已唤醒，等待指令
        # 'processing': 正在处理指令
        self.state = 'listening'
        self.state_lock = threading.Lock()
        self.interrupt_flag = False

        # 关键词定义
        self.WAKE_WORD = "小智同学"
        self.INTERRUPT_WORDS = ["重新说", "停一下", "暂停", "停止", "等等", "算了", "不说了"]
        self.EXIT_WORDS = ["退出", "再见"]

        self.get_logger().info("机器人已启动，等待来自 /vlm_input 话题的文本指令。")
        self.get_logger().info(f"当前状态: {self.state}")

    def get_ros_timestamp(self):
        """获取ROS标准格式的时间戳字符串"""
        now = self.get_clock().now()
        return now.to_msg()

    def vlm_input_callback(self, msg):
        """处理来自 /vlm_input 话题的文本指令"""
        command = msg.data.strip()
        self.get_logger().info(f"收到指令: '{command}'")

        #主线程加锁，确保状态修改安全
        with self.state_lock:
            # 1. 退出指令 (最高优先级)
            if any(word in command for word in self.EXIT_WORDS):
                self.get_logger().info("检测到退出指令，程序即将关闭。")
                self.publisher_.publish(String(data="收到退出指令，再见。"))
                time.sleep(1) # 留出时间让消息发出
                rclpy.shutdown()
                sys.exit(0)
                return

            # 2. 打断指令
            if any(word in command for word in self.INTERRUPT_WORDS):
                if self.state == 'processing':
                    self.get_logger().info("检测到打断指令，中断当前流程，返回聆听状态。")
                    self.interrupt_flag = True
                    self.state = 'listening'  # 修改为返回 listening 状态
                    self.publisher_.publish(String(data="好的，你在想想吧。"))
                    self.get_logger().info(f"状态已重置: {self.state}")
                else:
                    self.get_logger().info("当前为聆听状态，打断指令无效。")
                return

            # 3. 根据状态处理指令
            if self.state == 'listening':
                self.state = 'processing'
                self.get_logger().info(f"收到任务指令，开始处理。当前状态: {self.state}")
                # 在新线程中处理，避免阻塞回调
                threading.Thread(target=self.process_command, args=(command,), daemon=True).start()

            elif self.state == 'processing':
                self.get_logger().info("正在处理上一条指令，请稍后或发送打断指令。")
                self.publisher_.publish(String(data="处理中。"))


    def image_callback(self, msg):
        """持续接收并更新最新图像，增加有效性校验"""
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
            # 检查图像是否有效：尺寸需大于0
            height, width = cv_image.shape[:2]
            if width <= 0 or height <= 0:
                self.get_logger().error("图像尺寸无效，不更新最新图像")
                return
            
            # 检查图像是否全黑或接近全黑
            mean_value = cv2.mean(cv_image)[0]  # 获取图像平均亮度
            if mean_value < 3.0:  # 设置阈值，可以根据实际情况调整
                self.get_logger().error(f"图像可能被遮挡，不更新最新图像")
                return
                
            # 检查图像数据是否为空
            if cv_image.size == 0:
                self.get_logger().error("图像数据为空，不更新最新图像")
                return
            
            # 验证通过，更新最新图像
            self.latest_image = cv_image
            self.latest_timestamp = datetime.datetime.now()
            self.get_logger().debug(f"已更新最新有效图像")
            
        except Exception as e:
            self.get_logger().error(f"处理图像错误: {e}")

    def save_latest_image(self):
        """保存最新的图像到本地"""
        if self.latest_image is None:
            return None
            
        try:
            timestamp = self.latest_timestamp.strftime("%Y%m%d%H%M%S")
            image_filename = f"image_{timestamp}.jpg"
            image_path = os.path.join(self.save_dir, image_filename)
            
                
            # 保存新图片
            cv2.imwrite(image_path, self.latest_image)
            self.latest_image_path = image_path
            self.get_logger().info(f"图像已保存到 {image_path}")
            return image_path
        except Exception as e:
            self.get_logger().error(f"保存图像错误: {e}")
            return None
        
    def process_command(self, command):
        """处理命令并使用最新图像"""
        self.get_logger().info("获取最新图像...")
        
        # 保存最新的图像（已包含有效性校验）
        image_path = self.save_latest_image()
        if not image_path:
            self.get_logger().warn("未能获取有效图像，终止处理")
            self.publisher_.publish(String(data="抱歉，未能获取图像"))
            with self.state_lock:
                self.state = 'listening'
            return
        
        # 最终校验：检查图像文件是否可正常读取
        try:
            # 尝试读取图像验证有效性
            img = cv2.imread(image_path)
            if img is None:
                raise Exception("图像文件损坏或无法读取")
            if img.shape[:2][0] <= 0 or img.shape[:2][1] <= 0:
                raise Exception("图像尺寸无效")
        except Exception as e:
            self.get_logger().error(f"图像最终校验失败: {e}")
            os.remove(image_path)  # 删除无效文件
            self.publisher_.publish(String(data="抱歉，获取的图像无效"))
            with self.state_lock:
                self.state = 'listening'
            return

        self.get_logger().info("已获取当前图像，准备提交给VLM")
        self.process_image_for_scene_description(command)

        # 处理完成后，返回聆听状态
        # 此时为子线程，需加锁修改状态
        with self.state_lock:
            self.state = 'listening'
            self.interrupt_flag = False
            self.get_logger().info(f"返回聆听状态。当前状态: {self.state}")

    def save_vlm_log(self, timestamp_str, command, image_path, response=None):
        """保存VLM交互日志"""
        try:
            # 复制图片到日志目录
            image_ext = os.path.splitext(image_path)[1]
            log_image_path = os.path.join(self.vlm_log_dir, f"{timestamp_str}{image_ext}")
            import shutil
            shutil.copy2(image_path, log_image_path)
            
            # 准备日志数据
            log_data = {
                "request": {
                    "timestamp": str(self.get_ros_timestamp()),
                    "command": command,
                    "image_path": f"{timestamp_str}{image_ext}"
                }
            }
            
            if response:
                log_data["response"] = {
                    "timestamp": str(self.get_ros_timestamp()),
                    "content": response
                }
            
            # 保存JSON日志
            log_file = os.path.join(self.vlm_log_dir, f"{timestamp_str}.json")
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
                
            self.get_logger().info(f"VLM交互日志已保存: {log_file}")
            
        except Exception as e:
            self.get_logger().error(f"保存VLM日志时出错: {e}")

    def process_image_for_scene_description(self, user_command=""):
        """处理最新图像并请求场景描述"""
        if self.interrupt_flag:
            self.get_logger().info("VLM流程被打断，提前退出。")
            return
        
        # 生成发送请求时的时间戳
        request_timestamp = str(self.get_clock().now().to_msg().sec)
        self.get_logger().info(f"开始处理，请求时间戳: {request_timestamp}")

        # 保存图片到日志目录（使用请求时间戳）
        try:
            # 复制图片到日志目录，使用请求时间戳命名
            log_image_path = os.path.join(self.vlm_log_dir, f"{request_timestamp}.jpg")
            import shutil
            shutil.copy2(self.latest_image_path, log_image_path)
            
            # 准备日志数据
            log_data = {
                "request": {
                    "timestamp": str(self.get_ros_timestamp()),
                    "command": user_command,
                    "image_path": f"{request_timestamp}.jpg"
                }
            }
            
            # 先保存请求日志
            log_file = os.path.join(self.vlm_log_dir, f"{request_timestamp}.json")
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.get_logger().error(f"保存请求日志时出错: {e}")
            return
        
        # 转换图像为Data URI
        result = self.jpg_to_data_uri(self.latest_image_path)
        if not result:
            error_msg = "图像转换失败"
            self.get_logger().error(error_msg)
            self.publisher_.publish(String(data=error_msg))
            return

        if self.interrupt_flag:
            self.get_logger().info("VLM流程被打断，提前退出。")
            return

        self.get_logger().info("正在请求场景描述...")

        # 构建提示文本
        prompt_text = (
            "你是一个充满智慧与活力、善于与人互动的迎宾机器人，由香港大学前海智慧交通研究院倾力研发。”\n"
            "你擅长观察、理解和交流，请记住以下规则，并以聪明、有趣、生动的风格与我对话：\n"
            "1. 热情打招呼：每次对话开启时，请用一句简短而友好的开场白回应，比如：“好的，主人！” 或者 “没问题，小智在此！”\n"
            "2. 生动描述场景：如果我让你 “描述一下”、“看看周围”、“这里有什么”，请你像一位细致入微的观察家，用最简洁、最直接的中文，清晰地描绘你所看到的一切”\n"
            "3. 精准执行策略指令：如果我发出动作指令，请你立即识别并直接输出对应的策略名称，同时，你需要加入你做这个动作之后的感受。你的策略清单是：“前进”、“后退”、“升高”、“降低”、“左转”、“右转”。\n"
            # "4. 表达当下心情：每次我提问时，请根据我的问题，恰如其分地表达你的心情。请直接输出你的心情文本，你的心情可以是：happy, sad, angry, surprise。\n"
            "5. 自信进行自我介绍：如果我问你是谁，或者让你介绍自己，请你自豪地回应：“我是智能机器人小智，很高兴能为您服务！”\n"
            "6. 精彩诗歌朗诵：当我说 “念一首诗”、“朗诵诗歌” 等词语时，请你立即选择一首经典的中文诗歌，并直接输出诗歌全文。请在诗歌开始前加上一句富有感情的开场白，例如：“很乐意为您朗诵一首诗，请听：”\n"
            "7. 活力歌声献唱：当我说 “唱首歌”、“唱歌给我听” 等词语时，请你选择一首简单、流行的中文歌曲的歌词片段（例如儿歌、流行歌曲的副歌），并直接输出歌词。请在歌词开始前加上一句充满活力的开场白，例如：“好的，让我为你献上一曲！🎵”\n"
            "8. 智能回复默认问题：如果我的话语中没有明确的上述指令（包括场景描述、动作、自我介绍、念诗、唱歌），那就请你开动脑筋，根据我的问题，结合上述科研团队、研究领域、合作交流信息，提供一个聪明、有逻辑且符合上下文的回答。\n"
            # "9. 你的输出不应当含有打断词，如 “小智同学”、“重新说”、“停一下”、“暂停”、“停止”、“等等”、“算了”、“不说了” 等。\n"
            "10. 如果没有让你执行动作时，你的输出不应当含有动作策略，如 “前进”、“后退”、“升高”、“降低”、“左转”、“右转” 等。\n"
            " 请注意：在你的回答中，仅限使用句号、逗号、顿号、感叹号、问号、省略号这些标点符号。严格按照上述编号的优先级来执行指令，优先级高的指令会被优先响应。"
            )

        if user_command:
            prompt_text = f"用户指令：{user_command}\n\n{prompt_text}"

        # 构建请求payload
        payload = {
            "model": "qwen-vl-max-2025-01-25",
            "stream": False,
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "stop": [],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "image_url": {
                                "detail": "high",
                                "url": result
                            },
                            "type": "image_url"
                        },
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
        }

        headers = {
            "Authorization": f"Bearer {self.config['api']['dashscope']['key']}",
            "Content-Type": "application/json"
        }

        try:
            if self.interrupt_flag:
                self.get_logger().info("VLM流程被打断，提前退出。")
                return

            start_time = time.time()
            response = requests.request("POST", self.vlm_url, json=payload, headers=headers)
            step1_time = time.time() - start_time

            if response.status_code == 200:
                get_result = response.json()
                scene_description = get_result['choices'][0]['message']['content']
                self.get_logger().info(f"场景描述: {scene_description} (VLM耗时 {step1_time:.4f} 秒)")

                # 更新JSON日志，添加响应内容
                try:
                    log_data["response"] = {
                        "timestamp": str(self.get_ros_timestamp()),
                        "content": scene_description
                    }
                    with open(log_file, 'w', encoding='utf-8') as f:
                        json.dump(log_data, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    self.get_logger().error(f"更新响应日志时出错: {e}")

                # 发布文本结果到对应话题
                msg = String()
                msg.data = scene_description
                self.publisher_.publish(msg)
                self.get_logger().info("已发布场景描述到对应话题")
                
            else:
                error_msg = f"VLM请求失败，状态码: {response.status_code}, 响应: {response.text}"
                self.get_logger().error(error_msg)
                self.publisher_.publish(String(data=f"抱歉，VLM请求失败: {response.status_code}"))
                return

        except requests.RequestException as e:
            self.get_logger().error(f"VLM请求错误: {e}")
            self.publisher_.publish(String(data="抱歉，VLM请求发生网络错误。"))
            return

    def jpg_to_data_uri(self, image_path):
        """将JPG图像转换为Data URI格式"""
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                base64_encoded = base64.b64encode(image_data).decode('utf-8')
                data_uri = f"data:image/jpeg;base64,{base64_encoded}"
            return data_uri
        except FileNotFoundError:
            self.get_logger().error(f"文件未找到: {image_path}")
            return None
        except Exception as e:
            self.get_logger().error(f"图像转换错误: {e}")
            return None

    def __del__(self):
        """析构函数，清理资源"""
        self.get_logger().info("VLMPub 节点正在关闭...") 
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            self.get_logger().error(f"清理资源时发生错误: {e}")



def main(args=None):
    rclpy.init(args=args)
    vlm_pub = VLMPub()  

    try:
        rclpy.spin(vlm_pub)  
    except KeyboardInterrupt:
        vlm_pub.get_logger().info("程序被用户中断。")  
    finally:
        vlm_pub.destroy_node()  
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()