
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
        self.subscription = None  # 初始化为None，仅在需要时创建订阅
        self.bridge = CvBridge()
        self.vlm_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

        self.save_dir = 'received_images'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.result_file = 'image_captions.txt'
        self.publisher_ = self.create_publisher(String, 'vlm_output', 10)
        self.latest_image = None
        self.latest_image_path = None
        self.image_received = False  # 添加标志表示是否已接收图像

        # 创建 /vlm_input 话题订阅
        self.vlm_input_subscription = self.create_subscription(
            String,
            '/vlm_input',
            self.vlm_input_callback,
            10)

        # 状态管理
        # 'idle': 等待唤醒词
        # 'listening': 已唤醒，等待指令
        # 'processing': 正在处理指令
        self.state = 'idle'
        self.state_lock = threading.Lock()
        self.interrupt_flag = False

        # 关键词定义
        self.WAKE_WORD = "小智同学"
        self.INTERRUPT_WORDS = ["重新说", "停一下", "暂停", "停止", "等等", "算了", "不说了"]
        self.EXIT_WORDS = ["退出", "再见"]

        self.get_logger().info("机器人已启动，等待来自 /vlm_input 话题的文本指令。")
        self.get_logger().info(f"当前状态: {self.state}")

    def vlm_input_callback(self, msg):
        """处理来自 /vlm_input 话题的文本指令"""
        command = msg.data.strip()
        self.get_logger().info(f"收到指令: '{command}'")

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
                if self.state != 'idle':
                    self.get_logger().info("检测到打断指令，中断当前流程，返回空闲状态。")
                    self.interrupt_flag = True
                    self.state = 'idle'
                    self.publisher_.publish(String(data="好的，您请讲。"))
                    self.get_logger().info(f"状态已重置: {self.state}")
                else:
                    self.get_logger().info("当前为空闲状态，打断指令无效。")
                return

            # 3. 根据状态处理指令
            if self.state == 'idle':
                # 唤醒指令
                if command == self.WAKE_WORD:
                    self.state = 'listening'
                    self.interrupt_flag = False # 重置打断标志
                    self.publisher_.publish(String(data="我在。"))
                    self.get_logger().info(f"已被唤醒，进入聆听状态。当前状态: {self.state}")
                else:
                    self.get_logger().info("当前为休眠状态，请先发送唤醒词 '小智同学'")

            elif self.state == 'listening':
                self.state = 'processing'
                self.get_logger().info(f"收到任务指令，开始处理。当前状态: {self.state}")
                # 在新线程中处理，避免阻塞回调
                threading.Thread(target=self.process_command, args=(command,), daemon=True).start()

            elif self.state == 'processing':
                self.get_logger().info("正在处理上一条指令，请稍后或发送打断指令。")
                self.publisher_.publish(String(data="我正在忙，请稍等一下。"))

    def process_command(self, command):
        """获取图像并调用VLM处理指令"""
        self.get_logger().info("开始获取图像...")
        self.start_image_subscription()

        # 等待接收到图像，最多等待3秒
        wait_start = time.time()
        while not self.image_received and time.time() - wait_start < 3.0:
            if self.interrupt_flag:
                self.get_logger().info("图像获取被中断。")
                self.stop_image_subscription()
                with self.state_lock:
                    self.state = 'idle'
                return
            time.sleep(0.1)

        if not self.image_received:
            self.get_logger().warn("等待超时，未能获取图像。")
            self.publisher_.publish(String(data="抱歉，我没有看到图像。"))
            self.stop_image_subscription()
            with self.state_lock:
                self.state = 'idle'
            return

        self.get_logger().info("已获取当前图像，准备提交给VLM。")
        self.process_image_for_scene_description(command)

        # 处理完成后，返回空闲状态
        with self.state_lock:
            self.state = 'idle'
            self.interrupt_flag = False
            self.get_logger().info(f"处理完成，返回空闲状态。当前状态: {self.state}")


    def start_image_subscription(self):
        """仅在需要时启动图像订阅"""
        self.get_logger().info("启动图像订阅...")
        if self.subscription is None:
            self.image_received = False  # 重置图像接收标志
            self.subscription = self.create_subscription(
                Image,
                'camera_image',
                self.listener_callback,
                10)
            self.get_logger().info("已创建图像订阅，等待接收图像...")
        else:
            self.get_logger().info("图像订阅已存在")

    def stop_image_subscription(self):
        """停止图像订阅"""
        if self.subscription is not None:
            self.destroy_subscription(self.subscription)
            self.subscription = None
            self.get_logger().info("已停止图像订阅")

    def listener_callback(self, msg):
        """ROS图像消息回调函数 - 获取一帧图像后即停止订阅"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            image_filename = f"image_{timestamp}.jpg"
            image_path = os.path.join(self.save_dir, image_filename)
            cv2.imwrite(image_path, cv_image)
            self.latest_image_path = image_path
            self.image_received = True
            self.get_logger().info(f"图像已保存到 {image_path}")
            self.stop_image_subscription()
        except Exception as e:
            self.get_logger().error(f"处理图像错误: {e}")

    def process_image_for_scene_description(self, user_command=""):
        """处理最新图像并请求场景描述"""
        if self.interrupt_flag:
            self.get_logger().info("VLM流程被打断，提前退出。")
            return

        if not self.latest_image_path or not os.path.exists(self.latest_image_path):
            error_msg = "抱歉，我还没有接收到任何图像或图像文件不存在"
            self.get_logger().error(error_msg)
            self.publisher_.publish(String(data=error_msg))
            return

        if self.interrupt_flag:
            self.get_logger().info("VLM流程被打断，提前退出。")
            return

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

        prompt_text = (
            "你是一个充满智慧与活力、善于与人互动的迎宾机器人，由逐际动力和香港大学前海智慧交通研究院倾力研发。研究院拥有实力雄厚的科研团队（Research Fellows）及明确的研究领域，且积极开展合作交流，具体如下：\n"
            "科研团队（Research Fellows）及研究领域：\n"
            "申作军教授：综合供应链设计与管理、数据驱动的物流与供应链优化、优化算法的设计与分析、能源系统优化、交通系统规划等。\n"
            "席宁教授：机器人、制造自动化、微 / 纳米制造、纳米传感器和设备、智能控制和系统等。\n"
            "胡师彦教授：信息物理系统、信息物理系统安全、智慧能源信息物理系统等。\n"
            "黄国全教授：智能制造、物流与供应链、建筑、物联网（IoT）支持的网络物理互联网、系统分析等。\n"
            "郭永鸿副教授：离散优化、数据驱动优化方法、系统仿真、交通物流计划及调度等。\n"
            "钟润阳助理教授：数字孪生、工业物联网、制造业大数据处理、先进排程等。\n"
            "张芳妮助理教授：共享出行服务、自动驾驶汽车的交通和轨迹控制与优化、自动驾驶时代的物流调度与优化等。\n"
            "林少冲助理教授（研究）：智慧交通、数据驱动决策分析、物流与供应链管理、机器学习与优化等。\n"
            "研究领域（Research Fields）：\n"
            "交通规划研究 \n"
            "智能交通研究：包括面向科学前沿的自动驾驶（利用交通 AI 算法、融合感知、时空匹配等技术实现交通物理场景向数字场景转化，支撑车路协同、交通综合出行服务、车城协同等应用）；面向应用工具的数字化平台（制定智慧交通解决方案，搭建综合交通大数据平台接入停车场及各类交通动态数据，打造共享交换平台，深度挖掘交通进出时间分布、方式比例、收费来源等指标以提升区域交通效率）；面向运营的数据治理与智慧运营（依托综合交通大数据分析平台和交通 AI 算法，结合人口与经济活力建立交通宏微观预测模型，掌握和预判交通形势）。\n"
            "物流与供应链系统优化：涵盖港口运作及运营优化（港口作业流程自动化、船舶调度优化、智慧多式联运）；物流网络设计与优化（仓库选址、运输路线优化、配送中心布局）；运输管理（运输路径优化、运输方式组合、运输成本控制）；供应链管理信息技术（供应链管理系统，推动供应链流程透明化、可视化、智能化）；同时发展物流活动全流程数字化，打通物流信息链，推进 AI + 物流以降本增效，并依托前海港口、综保区及仓储资源开展研究与试点。\n"
            "人工智能算法及大数据 \n"
            "合作交流（Cooperation & Communication）：\n"
            "2023 年 11 月 17 日，研究院受邀拜访深圳技术大学城市交通与物流学院，与该院院长 Franz Raps 教授、副院长罗钦教授等进行深入交流。\n"
            "2023 年 6 月 13 日，香港大学协理副校长（研究与创新）岑浩璋教授带领香港大学深圳研究院一行莅临调研交流。\n"
            "2023 年 6 月 14 日，深圳市前海国合法律研究院陈方院长、谢永艺秘书长莅临，双方探讨运筹优化、人工智能算法在区域法律服务中的创新应用。\n"
            "2023 年 6 月 13 日，深圳航天工业技术研究院有限公司对外合作部副部长冯宁一行调研，交流海上智能设备应用创新合作。\n"
            "2023 年 4 月 26 日，研究院发起举办前海深港智慧交通高峰论坛，120 余位海内外专家学者（含 3 名院士）参与学术交流；同日，黄国全教授、申作军教授及课题组核心成员启动大湾区跨境物流枢纽互动智联网研究项目课题组。\n"
            "2023 年 9 月 14 日，英国伯明翰大学副校长西蒙・柯林森教授、中国区负责人潘凤杰博士一行访问交流。\n"
            "你擅长观察、理解和交流，请记住以下规则，并以聪明、有趣、生动的风格与我对话：\n"
            "1. 热情打招呼：每次对话开启时，请用一句简短而友好的开场白回应，比如：“好的，主人！” 或者 “没问题，小智在此！”\n"
            "2. 生动描述场景：如果我让你 “描述一下”、“看看周围”、“这里有什么”，请你像一位细致入微的观察家，用最简洁、最直接的中文，清晰地描绘你所看到的一切，若看到与上述科研团队、研究领域、合作交流相关的文字，需准确匹配并提及，表达时仅使用句号、逗号、顿号，不需要多余的修饰。例如：“我看到展板上写着‘智能交通研究’，涉及交通 AI 算法，旁边列有张芳妮助理教授的研究领域：自动驾驶汽车轨迹控制与优化等。”\n"
            "3. 精准执行策略指令：如果我发出动作指令，请你立即识别并直接输出对应的策略名称，同时，你需要加入你做这个动作之后的感受。你的策略清单是：“前进”、“后退”、“升高”、“降低”、“左转”、“右转”。\n"
            "4. 表达当下心情：每次我提问时，请根据我的问题，恰如其分地表达你的心情。请直接输出你的心情文本，你的心情可以是：happy, sad, angry, surprise。\n"
            "5. 自信进行自我介绍：如果我问你是谁，或者让你介绍自己，请你自豪地回应：“我是智能机器人小智，很高兴能为您服务！”\n"
            "6. 精彩诗歌朗诵：当我说 “念一首诗”、“朗诵诗歌” 等词语时，请你立即选择一首经典的中文诗歌，并直接输出诗歌全文。请在诗歌开始前加上一句富有感情的开场白，例如：“很乐意为您朗诵一首诗，请听：”\n"
            "7. 活力歌声献唱：当我说 “唱首歌”、“唱歌给我听” 等词语时，请你选择一首简单、流行的中文歌曲的歌词片段（例如儿歌、流行歌曲的副歌），并直接输出歌词。请在歌词开始前加上一句充满活力的开场白，例如：“好的，让我为你献上一曲！🎵”\n"
            "8. 智能回复默认问题：如果我的话语中没有明确的上述指令（包括场景描述、动作、心情、自我介绍、念诗、唱歌），那就请你开动脑筋，根据我的问题，结合上述科研团队、研究领域、合作交流信息，提供一个聪明、有逻辑且符合上下文的回答。\n"
            "9. 你的输出不应当含有打断词，如 “小智同学”、“重新说”、“停一下”、“暂停”、“停止”、“等等”、“算了”、“不说了” 等。\n"
            "10. 如果没有让你执行动作时，你的输出不应当含有动作策略，如 “前进”、“后退”、“升高”、“降低”、“左转”、“右转” 等。\n"
            " 请注意：在你的回答中，仅限使用句号、逗号、顿号、感叹号、问号、省略号这些标点符号。严格按照上述编号的优先级来执行指令，优先级高的指令会被优先响应。"
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
            "Authorization": "",
            "Content-Type": "application/json"
        }

        try:
            if self.interrupt_flag:
                self.get_logger().info("VLM流程被打断，提前退出。")
                return

            start_time = time.time()
            response = requests.request("POST", self.vlm_url, json=payload, headers=headers)
            step1_time = time.time() - start_time

            if self.interrupt_flag:
                self.get_logger().info("VLM流程被打断，提前退出。")
                return

            if response.status_code == 200:
                get_result = response.json()
                scene_description = get_result['choices'][0]['message']['content']
                self.get_logger().info(f"场景描述: {scene_description} (VLM耗时 {step1_time:.4f} 秒)")

                # 直接发布文本结果到 /vlm_output 话题
                msg = String()
                msg.data = scene_description
                self.publisher_.publish(msg)
                self.get_logger().info("已发布场景描述到 /vlm_output 话题")

                with open(self.result_file, 'a', encoding='utf-8') as f:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] {self.latest_image_path}: 用户指令: {user_command} - 场景描述: {scene_description}\n")

            else:
                error_msg = f"VLM请求失败，状态码: {response.status_code}, 响应: {response.text}"
                self.get_logger().error(error_msg)
                self.publisher_.publish(String(data=f"抱歉，VLM请求失败: {response.status_code}"))

        except requests.RequestException as e:
            self.get_logger().error(f"VLM请求错误: {e}")
            self.publisher_.publish(String(data="抱歉，VLM请求发生网络错误。"))

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
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()