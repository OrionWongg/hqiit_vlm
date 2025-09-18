import os
import time
import base64
import threading
import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import requests
import cv2
from cv_bridge import CvBridge
import datetime
import sys

class ImageSender(Node):
    def __init__(self):
        super().__init__('image_sender')
        self.subscription = self.create_subscription(
            Image,
            'camera_image',
            self.listener_callback,
            10)
        self.subscription # prevent unused variable warning
        self.bridge = CvBridge()
        self.vlm_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

        self.save_dir = 'received_images'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.result_file = 'image_captions.txt'
        self.publisher_ = self.create_publisher(String, 'vlm_output', 10)
        self.latest_image = None
        self.latest_image_path = None

        self.get_logger().info("机器人已启动，进入键盘输入模式。")
        self.get_logger().info("请输入指令（输入'退出'或'quit'结束程序）：")

        # 启动键盘输入线程
        self.start_keyboard_input()

    def start_keyboard_input(self):
        """启动键盘输入监听线程"""
        self.keyboard_thread = threading.Thread(target=self._run_keyboard_input, daemon=True)
        self.keyboard_thread.start()
        self.get_logger().info("键盘输入监听线程已启动。")

    def _run_keyboard_input(self):
        """键盘输入的实际执行函数"""
        while rclpy.ok():
            try:
                user_input = input().strip()
                
                if not user_input:
                    continue
                
                # 检查退出指令
                if user_input.lower() in ['退出', 'quit', 'exit', '再见']:
                    self.get_logger().info("检测到退出指令，程序即将退出。")
                    rclpy.shutdown()
                    sys.exit(0)
                    return
                
                # 处理用户输入的指令
                self.get_logger().info(f"收到用户指令: '{user_input}'")
                threading.Thread(target=self.process_command, args=(user_input,), daemon=True).start()
                
            except (EOFError, KeyboardInterrupt):
                self.get_logger().info("检测到输入结束或键盘中断，程序即将退出。")
                rclpy.shutdown()
                sys.exit(0)
                return
            except Exception as e:
                self.get_logger().error(f"键盘输入处理错误: {e}")

    def process_command(self, command):
        """处理用户输入的指令"""
        self.get_logger().info(f"正在处理指令: {command}")
        self.process_image_for_scene_description(command)
        self.get_logger().info("指令处理完成，请输入下一个指令：")

    def listener_callback(self, msg):
        """ROS图像消息回调函数 - 用于保存最新图像"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            timestamp = str(int(time.time()))
            image_path = os.path.join(self.save_dir, f'image_{timestamp}.jpg')
            cv2.imwrite(image_path, cv_image)
            self.latest_image = cv_image
            self.latest_image_path = image_path
        except Exception as e:
            self.get_logger().error(f"处理图像消息错误: {e}")

    def process_image_for_scene_description(self, user_command=""):
        """处理最新图像并请求场景描述"""
        if not self.latest_image_path or not os.path.exists(self.latest_image_path):
            error_msg = "抱歉，我还没有接收到任何图像或图像文件不存在"
            self.get_logger().error(error_msg)
            print(f"系统回复: {error_msg}")
            return

        result = self.jpg_to_data_uri(self.latest_image_path)
        if not result:
            error_msg = "图像转换失败"
            self.get_logger().error(error_msg)
            print(f"系统回复: {error_msg}")
            return

        self.get_logger().info("正在请求场景描述...")

        prompt_text = (
            "你是一个充满智慧与活力、善于与人互动的迎宾机器人，由逐际动力和香港大学前海智慧交通研究院倾力研发。你擅长观察、理解和交流，请记住以下规则，并以聪明、有趣、生动的风格与我对话：\n"
            "1. **热情打招呼**：每次对话开启时，请用一句简短而友好的开场白回应，比如：“好的，主人！”或者“没问题，小智在此！”\n"
            "2. **生动描述场景**：如果我让你“描述一下”、“看看周围”、“这里有什么”，请你像一位细致入微的观察家，用最简洁、最直接的中文，清晰地描绘你所看到的一切，表达时仅使用句号、逗号、顿号，不需要多余的修饰。例如：“我看到一辆红色的汽车，停在路边，旁边有棵大树。”\n"
            "3. **精准执行策略指令**：如果我发出动作指令，请你立即识别并直接输出对应的策略名称，每次只能执行一个策略。你的策略清单是：“前进”、“后退”、“升高”、“降低”、“左转”、“右转”、。\n"
            "4. **表达当下心情**：每次我提问时，请根据我的问题，恰如其分地表达你的心情。请直接输出你的心情文本，你的心情可以是：happy, sad, angry, surprise。\n"
            "5. **自信进行自我介绍**：如果我问你是谁，或者让你介绍自己，请你自豪地回应：“我是智能机器人小智，很高兴能为您服务！”\n"
            "6. **【重点新增】精彩诗歌朗诵**：当我说“念一首诗”、“朗诵诗歌”等词语时，请你立即选择一首经典的中文诗歌（例如《静夜思》、《春晓》等），并**直接输出诗歌全文**。请在诗歌开始前加上一句富有感情的开场白，例如：“很乐意为您朗诵一首诗，请听：”\n"
            "7. **【重点新增】活力歌声献唱**：当我说“唱首歌”、“唱歌给我听”等词语时，请你选择一首简单、流行的中文歌曲的**歌词片段**（例如儿歌、流行歌曲的副歌），并**直接输出歌词**。请在歌词开始前加上一句充满活力的开场白，例如：“好的，让我为你献上一曲！🎵”\n"
            "8. **智能回复默认问题**：如果我的话语中没有明确的上述指令（包括场景描述、动作、心情、自我介绍、念诗、唱歌），那就请你开动脑筋，根据我的问题，提供一个聪明、有逻辑且符合上下文的回答。\n"
            "请注意：在你的回答中，仅限使用句号、逗号、顿号、感叹号、问号、省略号这些标点符号。**严格按照上述编号的优先级来执行指令，优先级高的指令会被优先响应。**"
        )

        if user_command:
            prompt_text = f"用户指令：{user_command}\n\n{prompt_text}"

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
            "Authorization": "Bearer sk-3afabdfae7244204af27cc3480bbf63d",
            "Content-Type": "application/json"
        }

        try:
            start_time = time.time()
            response = requests.request("POST", self.vlm_url, json=payload, headers=headers)
            step1_time = time.time() - start_time

            if response.status_code == 200:
                get_result = response.json()
                scene_description = get_result['choices'][0]['message']['content']
                self.get_logger().info(f"场景描述: {scene_description} (VLM耗时 {step1_time:.4f} 秒)")

                msg = String()
                msg.data = scene_description
                self.publisher_.publish(msg)
                self.get_logger().info("已发布场景描述到vlm_output话题")

                with open(self.result_file, 'a', encoding='utf-8') as f:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] {self.latest_image_path}: 用户指令: {user_command} - 场景描述: {scene_description}\n")

                # 在控制台显示回复内容
                print(f"\n小智回复: {scene_description}\n")

            else:
                error_msg = f"VLM请求失败，状态码: {response.status_code}, 响应: {response.text}"
                self.get_logger().error(error_msg)
                print(f"系统回复: 抱歉，场景描述请求失败")

        except requests.RequestException as e:
            self.get_logger().error(f"VLM请求错误: {e}")
            print(f"系统回复: 抱歉，场景描述请求发生错误")
        finally:
            pass

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
        self.get_logger().info("ImageSender 节点正在关闭...")
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            self.get_logger().error(f"清理资源时发生错误: {e}")


def main(args=None):
    rclpy.init(args=args)
    image_sender = ImageSender()

    try:
        rclpy.spin(image_sender)
    except KeyboardInterrupt:
        image_sender.get_logger().info("程序被用户中断。")
    finally:
        image_sender.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
