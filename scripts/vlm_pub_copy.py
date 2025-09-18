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
from aip import AipSpeech  # Baidu AipSpeech for Text-to-Speech (TTS) and ASR
import datetime
import pyaudio
import wave
import subprocess
import sys
import signal
from pypinyin import lazy_pinyin
from Levenshtein import distance as levenshtein_distance

# For continuous speech recognition (Vosk)
from vosk import Model, KaldiRecognizer, SetLogLevel
# Set Vosk log level to -1 to suppress internal logging
SetLogLevel(-1)

# --- Vosk Model Download Reminder (Important!) ---
# Ensure you have downloaded a Chinese Vosk model (e.g., vosk-model-cn-0.22.zip)
# from https://alphacephei.com/vosk/models
# Extract it into a folder named 'model' in the same directory as this script.
# Example path: ~/ros_emoji/scripts/model/
# -------------------------------------------------

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


        # Baidu AipSpeech credentials (for both ASR and TTS)
        self.APP_ID = '6818881'
        self.API_KEY = "nhCW0mN0yNAvL9mPmmoXcsPI" # Placeholder, replace with your actual API Key
        self.SECRET_KEY = "sLNhbmq0180wTarIHrpIjznOjgcPOC0e" # Placeholder, replace with your actual Secret Key
        self.speech_client = AipSpeech(self.APP_ID, self.API_KEY, self.SECRET_KEY)

        # Audio Prompts
        self.AUDIO_PROMPT_DIR = 'audio_prompts'
        self.PROMPT_MAPPINGS = {
            "greeting": ("你好呀，我是小智", "greeting.mp3"),# 欢迎语
            "listening_trigger": ("我在听。", "listening_trigger.mp3"), # 听到激活词后播放
            "interrupt": ("好的主人，你再想一想吧", "interrupt.mp3"), # 语音识别失败
            "record_finish": ("知道了。", "record_finish.mp3"), # 录音结束后播放
            "not_understood": ("抱歉，我没有听清，请再说一遍。", "not_understood.mp3"), # 语音识别失败
            "goodbye_voice_exit": ("我们下次再见。", "goodbye_voice_exit.mp3") # 听到退出词后播放
        }
        self._ensure_audio_prompts_exist()

        # PyAudio configuration
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024
        self.audio = pyaudio.PyAudio()
        self.stream = None # For continuous listening audio input stream

        # Vosk ASR setup for continuous listening (only for trigger words now)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.vosk_model_path = os.path.join(script_dir, "model")

        if not os.path.exists(self.vosk_model_path):
            self.get_logger().error(f"Vosk 模型未找到！请检查路径: {self.vosk_model_path}")
            self.get_logger().error("请从 https://alphacephei.com/vosk/models 下载中文模型，并解压到此目录下的 'model' 文件夹。")
            rclpy.shutdown()
            sys.exit(1)
             
        try:
            self.vosk_model = Model(self.vosk_model_path)
            self.vosk_rec = KaldiRecognizer(self.vosk_model, self.RATE)
            self.vosk_rec.SetWords(False) 
            self.get_logger().info("Vosk 模型加载成功。")
        except Exception as e:
            self.get_logger().error(f"加载 Vosk 模型时出错: {e}")
            self.get_logger().error("请确保 Vosk 库已正确安装且模型完整。")
            rclpy.shutdown()
            sys.exit(1)

        # Audio buffer for recording segments between keywords
        self.recording_segment_active = False
        self.audio_frames_segment = []

        # State for voice interaction flow
        # 'idle': waiting for "小智同学" (Vosk)
        # 'recording_command': recording audio after "小智同学", waiting for speech timeout (Vosk updates last_speech_time)
        # 'processing': ASR/VLM is busy, ignore new triggers for a moment
        self.voice_state = 'idle'
        self.voice_state_lock = threading.Lock()
        
        # 标志位：表示当前是否正在播放“长语音”（VLM输出）
        self.is_speaking_vlm_output = False 

        # 标志位：用于打断当前流程
        self.interrupt_flag = False

        # Define wake word and its pinyin for fuzzy matching
        self.WAKE_WORD_TEXT = "小智同学"
        self.WAKE_WORD_PINYIN = "".join(lazy_pinyin(self.WAKE_WORD_TEXT)).lower() 
        
        # 定义唤醒词的关键组成部分，用于更灵活的模糊匹配
        self.WAKE_WORD_PARTS_PINYIN = [
            "".join(lazy_pinyin("小智")).lower(), 
            "".join(lazy_pinyin("同学")).lower(), 
        ]
        
        self.WAKE_WORD_FUZZY_THRESHOLD = 0.7 

        # --- New: Speech timeout variables ---
        self.last_speech_time = time.time() # Tracks last time speech was detected
        self.SPEECH_TIMEOUT_SECONDS = 3.0 # 3 second timeout
        self._speech_timeout_thread = None # Initialize to None

        self.get_logger().info("机器人已启动，进入语音输入模式。")
        # 直接播放欢迎语，Vosk保持开启
        self.play_pregenerated_audio("greeting") 

        self.start_continuous_listening()

    def start_image_subscription(self):
        """仅在需要时启动图像订阅"""
        self.get_logger().info("启动图像订阅...")
        if self.subscription is None:
            self.image_received = False  # 重置图像接收标志
            self.subscription = self.create_subscription(
                Image,
                '/camera/color/image_raw',
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

    def _ensure_audio_prompts_exist(self):
        """检查并生成语音提示文件"""
        if not os.path.exists(self.AUDIO_PROMPT_DIR):
            os.makedirs(self.AUDIO_PROMPT_DIR)
            self.get_logger().info(f"Created audio prompt directory: {self.AUDIO_PROMPT_DIR}")

        for key, (text, filename) in self.PROMPT_MAPPINGS.items():
            filepath = os.path.join(self.AUDIO_PROMPT_DIR, filename)
            if not os.path.exists(filepath):
                self.get_logger().info(f"Generating missing audio prompt: {filename}")
                try:
                    result = self.speech_client.synthesis(text, 'zh', 1, {
                        'vol': 15,
                        'spd': 7,
                        'pit': 5,
                        'per': 1
                    })
                    if not isinstance(result, dict):
                        with open(filepath, 'wb') as f:
                            f.write(result)
                        self.get_logger().info(f"Successfully generated {filename}")

                    else:
                        self.get_logger().error(f"Failed to generate {filename}: {result}")

                except Exception as e:
                    self.get_logger().error(f"Error generating {filename}: {e}")

            else:
                self.get_logger().info(f"Audio prompt exists: {filename}")

    def play_pregenerated_audio(self, prompt_key):
        """播放预先生成的语音提示文件 (短提示音，不影响Vosk监听)"""
        self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}") # 新增状态打印
        if prompt_key in self.PROMPT_MAPPINGS:
            _, filename = self.PROMPT_MAPPINGS[prompt_key]
            filepath = os.path.join(self.AUDIO_PROMPT_DIR, filename)
            if os.path.exists(filepath):
                try:
                    self.get_logger().info(f"Playing pre-generated audio: {filename}")
                    # 不设置 self.is_speaking_vlm_output = True
                    subprocess.run(['mpg321', filepath], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except FileNotFoundError:
                    self.get_logger().error("mpg321 command not found. Please install it (e.g., sudo apt-get install mpg321).")
                except subprocess.CalledProcessError as e:
                    self.get_logger().error(f"Error playing {filename} with mpg321: {e}")
                except Exception as e:
                    self.get_logger().error(f"Error playing {filename}: {e}")
            else:
                self.get_logger().warning(f"Pre-generated audio file not found: {filepath}, attempting to synthesize using speak_text...")
                # 如果预生成失败，仍然使用 speak_text，此时 Vosk 会暂停
                # 这是一个权衡，为了保证语音能出，即使短暂暂停VosK也值得
                self.speak_text(self.PROMPT_MAPPINGS[prompt_key][0])
        else:
            self.get_logger().warning(f"Unknown audio prompt key: {prompt_key}")

    def start_continuous_listening(self):
        """启动连续语音监听线程 (Vosk)"""
        self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}") # 新增状态打印
        if not hasattr(self, '_vosk_listening_thread') or not (self._vosk_listening_thread and self._vosk_listening_thread.is_alive()):
            self._vosk_listening_thread = threading.Thread(target=self._run_continuous_listening, daemon=True)
            self._vosk_listening_thread.start()
            self.get_logger().info("Vosk 连续语音监听线程已启动。")
            self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}") # 新增状态打印

        else:
            self.get_logger().info("Vosk 连续语音监听线程已在运行。")

        # Start the speech timeout thread if not already running
        if not hasattr(self, '_speech_timeout_thread') or not (self._speech_timeout_thread and self._speech_timeout_thread.is_alive()):
            self._speech_timeout_thread = threading.Thread(target=self._run_speech_timeout_check, daemon=True)
            self._speech_timeout_thread.start()
            self.get_logger().info("语音超时检测线程已启动。")
            self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}") # 新增状态打印

        else:
            self.get_logger().info("语音超时检测线程已在运行。")

    def _text_to_pinyin(self, text):
        """将中文文本转换为拼音字符串"""
        return "".join(lazy_pinyin(text)).lower()

    def _calculate_pinyin_similarity(self, recognized_pinyin, target_pinyin):
        """
        计算识别到的拼音与目标唤醒词拼音的相似度，采用更宽松的模糊匹配逻辑。
        目标唤醒词: "小智同学" (xiaozhitongxue)
        需要匹配的模糊情况: "小字同学" (xiaozitongxue), "小次同学" (xiaocitongxue), "小吃同学" (xiaochitongxue)
        以及只包含部分词的情况，如 "小智", "同学"。
        """
        if not recognized_pinyin:
            return 0.0

        # 1. 精确匹配 (如果完全一致，相似度最高)
        if recognized_pinyin == target_pinyin:
            return 1.0
        
        # 2. Levenshtein 距离相似度
        max_len = max(len(recognized_pinyin), len(target_pinyin))
        if max_len == 0: return 0.0 # Avoid division by zero
        lev_dist = levenshtein_distance(recognized_pinyin, target_pinyin)
        lev_similarity = 1.0 - (lev_dist / max_len)
        self.get_logger().debug(f"Levenshtein相似度: {lev_similarity} (识别: {recognized_pinyin}, 目标: {target_pinyin})")

        # 3. 核心词组匹配
        xiaozhi_pinyin = self.WAKE_WORD_PARTS_PINYIN[0] 
        tongxue_pinyin = self.WAKE_WORD_PARTS_PINYIN[1] 

        has_xiaozhi = xiaozhi_pinyin in recognized_pinyin
        has_tongxue = tongxue_pinyin in recognized_pinyin

        # 如果同时包含 "小智" 和 "同学" 且顺序正确
        if has_xiaozhi and has_tongxue:
            xiaozhi_idx = recognized_pinyin.find(xiaozhi_pinyin)
            tongxue_idx = recognized_pinyin.find(tongxue_pinyin)
            
            if xiaozhi_idx != -1 and tongxue_idx != -1 and xiaozhi_idx < tongxue_idx:
                return max(lev_similarity, 0.8) 

        # 如果只包含 "小智" (即使不完全匹配，只要Levenshtein相似度高，也视为匹配)
        pinyin_of_xiaozhi = "".join(lazy_pinyin("小智")).lower()
        if pinyin_of_xiaozhi in recognized_pinyin: 
            return max(lev_similarity, 0.7)
        else: 
            dist_to_xiaozhi = levenshtein_distance(recognized_pinyin, pinyin_of_xiaozhi)
            if len(pinyin_of_xiaozhi) > 0 and (1.0 - (dist_to_xiaozhi / max(len(recognized_pinyin), len(pinyin_of_xiaozhi), 1))) >= 0.75:
                return max(lev_similarity, 0.7)


        # 如果只包含 "同学" (即使不完全匹配，只要Levenshtein相似度高，也视为匹配)
        pinyin_of_tongxue = "".join(lazy_pinyin("同学")).lower()
        if pinyin_of_tongxue in recognized_pinyin: 
            return max(lev_similarity, 0.6)
        else: 
            dist_to_tongxue = levenshtein_distance(recognized_pinyin, pinyin_of_tongxue)
            if len(pinyin_of_tongxue) > 0 and (1.0 - (dist_to_tongxue / max(len(recognized_pinyin), len(pinyin_of_tongxue), 1))) >= 0.75:
                return max(lev_similarity, 0.6)

        # 针对不标准发音的更细粒度检查
        target_zhi_pinyin_char = lazy_pinyin("智")[0].lower() 
        if target_zhi_pinyin_char in recognized_pinyin:
             return max(lev_similarity, 0.55)
        else:
            similar_zh_sounds = ['zi', 'ci', 'chi']
            for sound in similar_zh_sounds:
                if levenshtein_distance(target_zhi_pinyin_char, sound) <= 1: 
                    if sound in recognized_pinyin:
                        return max(lev_similarity, 0.55) 

        return lev_similarity

    def _run_speech_timeout_check(self):
        """Thread to continuously check for speech timeout."""
        self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}") # 新增状态打印
        while rclpy.ok():
            time.sleep(0.1)  # Check every 100ms
            with self.voice_state_lock:
                # Only check timeout if actively recording a command and not currently speaking VLM output
                if self.voice_state == 'recording_command' and not self.is_speaking_vlm_output:
                    self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}") # 新增状态打印
                    time_since_last_speech = time.time() - self.last_speech_time
                    if time_since_last_speech >= self.SPEECH_TIMEOUT_SECONDS:
                        self.get_logger().info(f"检测到 {self.SPEECH_TIMEOUT_SECONDS} 秒无语音输入，自动结束录音并处理。")
                        # Trigger the processing as if an end word was detected
                        threading.Thread(target=self._process_command_segment, daemon=True).start()
                        self.voice_state = 'processing' # Immediately set state to processing
                        self.recording_segment_active = False # Stop recording
                        self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}") # 新增状态打印

    def _run_continuous_listening(self):
        """连续语音监听的实际执行函数 (Vosk)"""
        p = pyaudio.PyAudio()
        stream = None
        try:
            input_device_index = None
            info = p.get_host_api_info_by_index(0)
            num_devices = info.get('deviceCount')
            self.get_logger().info("Available Audio Input Devices:")
            for i in range(0, num_devices):
                device_info = p.get_device_info_by_host_api_device_index(0, i)
                if device_info.get('maxInputChannels') > 0:
                    self.get_logger().info(f"  Device ID {i}: {device_info.get('name')}")

            stream = p.open(format=pyaudio.paInt16,
                            channels=self.CHANNELS,
                            rate=self.RATE,
                            input=True,
                            frames_per_buffer=self.CHUNK,
                            input_device_index=input_device_index)
            self.get_logger().info("PyAudio 流已打开，开始 Vosk 连续监听...")
            self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}") # 新增状态打印


            while rclpy.ok():
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                
                with self.voice_state_lock:
                    current_state = self.voice_state

                    # --- 新增：VLM语音播报期间，允许打断词和退出词 ---
                    if self.is_speaking_vlm_output:
                        # 识别文本
                        if self.vosk_rec.AcceptWaveform(data):
                            result = json.loads(self.vosk_rec.Result())
                            text = result.get('text', '').strip()
                            normalized_text = text.replace(' ', '').lower()
                            INTERRUPT_WORDS = ["重新说", "停一下", "暂停", "停止", "等等", "算了", "不说了"]
                            if any(word in normalized_text for word in INTERRUPT_WORDS) or ("退出" in normalized_text or "再见" in normalized_text):
                                threading.Thread(target=self._process_vosk_result_async, args=(text,), daemon=True).start()
                            # 其他情况直接跳过
                        else:
                            partial = json.loads(self.vosk_rec.PartialResult())
                            partial_text = partial.get('partial', '').strip().replace(' ', '').lower()
                            INTERRUPT_WORDS = ["重新说", "停一下", "暂停", "停止", "等等", "算了", "不说了"]
                            if any(word in partial_text for word in INTERRUPT_WORDS) or ("退出" in partial_text or "再见" in partial_text):
                                threading.Thread(target=self._process_vosk_result_async, args=(partial_text,), daemon=True).start()
                        # 如果正在录音，继续收集音频
                        if self.recording_segment_active:
                            self.audio_frames_segment.append(data)
                        continue # 跳过其他识别

                # Feed audio to Vosk regardless of state (except when VLM is speaking)
                # This ensures Vosk is always ready to detect wake words or end of speech
                if self.vosk_rec.AcceptWaveform(data):
                    result = json.loads(self.vosk_rec.Result())
                    text = result.get('text', '').strip()
                    if text:
                        # Any speech detected (final result) updates last_speech_time
                        with self.voice_state_lock:
                            self.last_speech_time = time.time()
                        threading.Thread(target=self._process_vosk_result_async, args=(text,), daemon=True).start()
                else:
                    partial = json.loads(self.vosk_rec.PartialResult())
                    partial_text = partial.get('partial', '').strip().replace(' ', '').lower()
                    
                    # Any partial speech detected updates last_speech_time
                    if partial_text:
                        with self.voice_state_lock:
                            self.last_speech_time = time.time()

                    # Only process partial results for wake word (in idle state)
                    # Exit word is handled by final result or `_run_speech_timeout_check`
                    if current_state == 'idle':
                        recognized_pinyin = self._text_to_pinyin(partial_text)
                        similarity = self._calculate_pinyin_similarity(recognized_pinyin, self.WAKE_WORD_PINYIN)
                        if similarity >= self.WAKE_WORD_FUZZY_THRESHOLD:
                            self.get_logger().info(f"Vosk 部分识别到相似唤醒词: '{partial_text}' (加速处理)")
                            # Pass wake word as a clear signal
                            threading.Thread(target=self._process_vosk_result_async, args=(self.WAKE_WORD_TEXT,), daemon=True).start()
                
                with self.voice_state_lock:
                    if self.recording_segment_active:
                        self.audio_frames_segment.append(data)

        except Exception as e:
            self.get_logger().error(f"连续语音监听错误: {e}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            p.terminate()
            self.get_logger().info("连续语音监听已停止。")

    def _process_vosk_result_async(self, text):
        """异步处理 Vosk 识别结果，移除关键词并处理语音交互逻辑"""
        with self.voice_state_lock:
            normalized_text = text.replace(' ', '').lower()
            # --- 打断词检测（始终允许，优先级最高，但idle状态下不响应） ---
            INTERRUPT_WORDS = ["重新说", "停一下", "暂停", "停止",  "等等", "算了", "不说了"]
            if self.voice_state != 'idle' and any(word in normalized_text for word in INTERRUPT_WORDS):
                self.get_logger().info("检测到打断词，立即中断当前流程，回到idle状态。")
                # 停止录音
                self.recording_segment_active = False
                self.audio_frames_segment = []
                # 这里如果有ASR或VLM的线程在处理，用标志位让它们检测并主动return（可扩展）
                self.interrupt_flag = True
                self.voice_state = 'idle'
                self.get_logger().info(f"已打断，当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}")
                self.play_pregenerated_audio("interrupt")
                return

            # 仅当正在播报 VLM 输出的“长语音”时，才忽略 Vosk 识别结果（除了打断词）
            if self.is_speaking_vlm_output:
                self.get_logger().info(f"正在播放VLM语音，忽略 Vosk 识别结果: '{text}'")
                self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}") # 新增状态打印
                return 

            current_state_at_start = self.voice_state 
            self.get_logger().info(f"Vosk 识别到: '{text}' (当前状态: {current_state_at_start})")
            self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}") # 新增状态打印

            # --- 退出词处理（最高优先级，无论状态） ---
            if "退出" in normalized_text or "再见" in normalized_text:
                self.get_logger().info("检测到退出词，程序即将退出。")
                self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}") # 新增状态打印
                self.play_pregenerated_audio("goodbye_voice_exit")
                time.sleep(2)
                rclpy.shutdown()
                sys.exit(0)
                return

            # --- 唤醒词处理（只在 idle 状态下） ---
            if current_state_at_start == 'idle':
                recognized_pinyin = self._text_to_pinyin(normalized_text)
                similarity = self._calculate_pinyin_similarity(recognized_pinyin, self.WAKE_WORD_PINYIN)
                self.get_logger().debug(f"唤醒词拼音相似度: {similarity} (识别: '{recognized_pinyin}', 目标: '{self.WAKE_WORD_PINYIN}')")

                if similarity >= self.WAKE_WORD_FUZZY_THRESHOLD:
                    self.get_logger().info(f"Vosk 检测到激活词 '{self.WAKE_WORD_TEXT}' 或其相似发音，开始录音。")
                    self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}") # 新增状态打印
                    self.play_pregenerated_audio("listening_trigger") # Vosk 保持开启
                    self.recording_segment_active = True
                    self.audio_frames_segment = []
                    self.last_speech_time = time.time() # Reset speech timer
                    self.voice_state = 'recording_command'
                    return 

            self.get_logger().debug(f"当前状态 {current_state_at_start}，未匹配到特殊关键词，忽略 Vosk 识别结果: '{text}'")
            
    def listener_callback(self, msg):
        """ROS图像消息回调函数 - 获取一帧图像后即停止订阅"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 保存图像
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            image_filename = f"image_{timestamp}.jpg"
            image_path = os.path.join(self.save_dir, image_filename)
            cv2.imwrite(image_path, cv_image)
            
            self.latest_image = cv_image
            self.latest_image_path = image_path
            self.image_received = True
            
            self.get_logger().info(f"图像已保存到 {image_path}")
            
            # 获取到一帧图像后立即停止订阅
            self.stop_image_subscription()
            
        except Exception as e:
            self.get_logger().error(f"处理图像错误: {e}")

    def _process_command_segment(self):
        """Processes the recorded audio segment for command recognition."""
        with self.voice_state_lock:
            # Only process if currently in 'processing' state (triggered by timeout)
            if self.voice_state == 'processing':
                self.get_logger().info("处理录音片段。")
                self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}")
                self.play_pregenerated_audio("record_finish") # Vosk 保持开启

                # 确保在录音结束时立即开始订阅图像
                self.start_image_subscription()
                
                # 等待接收到图像，最多等待3秒
                wait_start = time.time()
                while not self.image_received and time.time() - wait_start < 3.0:
                    time.sleep(0.1)
                
                if not self.image_received:
                    self.get_logger().warn("等待超时，未能获取图像")
                    self.stop_image_subscription()  # 确保停止订阅
                
                self.get_logger().info("录音结束，已获取当前图像")
                self.interrupt_flag = False

                if self.audio_frames_segment:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                    audio_filename = f"segment_recording_{timestamp}.wav"
                    self.save_audio_segment(audio_filename, self.audio_frames_segment)
                    
                    self.audio_frames_segment = [] # 清空缓冲区

                    # 确保我们有最新的图像
                    if not self.latest_image_path or not os.path.exists(self.latest_image_path):
                        self.get_logger().warn("没有找到最新图像，无法处理命令")
                        self.voice_state = 'idle'
                        return

                    recognized_command = self.recognize_speech_baidu(audio_filename)

                    if recognized_command:
                        self.get_logger().info(f"百度ASR识别结果: '{recognized_command}'")
                        self.get_logger().info(f"将指令 '{recognized_command}' 提交给VLM。")
                        threading.Thread(target=self.process_command_after_asr, args=(recognized_command,), daemon=True).start()
                        self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}")
                    else:
                        self.get_logger().info("百度ASR未识别到有效指令。")
                        self.voice_state = 'idle'
                        self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}")
                    
                    # 清理音频文件
                    if os.path.exists(audio_filename):
                        try:
                            os.remove(audio_filename)
                            self.get_logger().info(f"已删除临时音频文件: {audio_filename}")
                        except Exception as e:
                            self.get_logger().error(f"删除音频文件错误: {e}")
                else:
                    self.get_logger().info("录音片段为空。")
                    self.voice_state = 'idle'
                    self.interrupt_flag = False

    def save_audio_segment(self, filename, frames):
        """保存录音片段为WAV文件"""
        try:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            self.get_logger().info(f"音频片段已保存到 {filename}")
        except Exception as e:
            self.get_logger().error(f"保存音频文件错误: {e}")

    def recognize_speech_baidu(self, audio_file):
        """使用百度语音API识别语音文件"""
        try:
            # 检查中断标志
            if self.interrupt_flag:
                self.get_logger().info("ASR流程被打断，提前退出。")
                return None
            with open(audio_file, 'rb') as f:
                audio_data = f.read()

            self.get_logger().info(f"正在将音频文件 '{audio_file}' 发送至百度ASR进行识别...")
            result = self.speech_client.asr(audio_data, 'wav', self.RATE, {
                'dev_pid': 1537, # 1537 是中文普通话 
            })

            # 检查中断标志
            if self.interrupt_flag:
                self.get_logger().info("ASR流程被打断，提前退出。")
                return None

            if 'result' in result and result['result']:
                recognized_text = result['result'][0]
                self.get_logger().info(f"百度ASR原始识别结果: {recognized_text}")
                self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}") # 新增状态打印

                return recognized_text
            else:
                self.get_logger().error(f"百度语音识别失败或无结果: {result}")
                self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}") # 新增状态打印

                return None
        except Exception as e:
            self.get_logger().error(f"百度语音识别错误: {e}")
            self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}") # 新增状态打印

            return None

    def process_command_after_asr(self, command):
        """处理百度ASR识别出的指令，接着进行VLM处理"""
        self.get_logger().info(f"正在处理来自百度ASR的指令: {command}")
        # 这里会调用 speak_text 来播报 VLM 结果，从而触发 Vosk 暂停
        self.process_image_for_scene_description(command) 

        with self.voice_state_lock:
            self.voice_state = 'idle'
            self.interrupt_flag = False 
            self.get_logger().info("VLM处理完成，系统返回到空闲（idle）状态，等待新的激活词。")
            self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}") # 新增状态打印

    


    def process_image_for_scene_description(self, user_command=""):
        """处理最新图像并请求场景描述"""

        # --- 检查打断标志，优先退出 ---
        if self.interrupt_flag:
            self.get_logger().info("VLM流程被打断，提前退出。")
            return
        
        # 检查是否有可用的图像
        if not self.latest_image_path or not os.path.exists(self.latest_image_path):
            error_msg = "抱歉，我还没有接收到任何图像或图像文件不存在"
            self.get_logger().error(error_msg)
            self.speak_text(error_msg) # 此时 Vosk 会暂停
            self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}") # 新增状态打印

            return
        # --- 再次检查打断标志 ---
        if self.interrupt_flag:
            self.get_logger().info("VLM流程被打断，提前退出。")
            return

        result = self.jpg_to_data_uri(self.latest_image_path)
        if not result:
            error_msg = "图像转换失败"
            self.get_logger().error(error_msg)
            self.speak_text(error_msg) # 此时 Vosk 会暂停
            self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}") # 新增状态打印

            return
        
        # --- 再次检查打断标志 ---·
        if self.interrupt_flag:
            self.get_logger().info("VLM流程被打断，提前退出。")
            return

        self.get_logger().info("正在请求场景描述...")

        prompt_text = (
            "你是一个充满智慧与活力、善于与人互动的迎宾机器人，由香港大学前海智慧交通研究院倾力研发。研究院拥有实力雄厚的科研团队（Research Fellows）及明确的研究领域，且积极开展合作交流，具体如下：\n"
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
            "2. 生动描述场景：如果我让你 “描述一下”、“看看周围”、“这里有什么”，请你像一位细致入微的观察家，用最简洁、最直接的中文，清晰地描绘你所看到的一切，若看到与上述科研团队、研究领域、合作交流相关的文字，需准确匹配并提及，表达时仅使用句号、逗号、顿号，不需要多余的修饰。\n"
            "3. 自信进行自我介绍：如果我问你是谁，或者让你介绍自己，请你自豪地回应：“我是智能机器人小智，很高兴能为您服务！”\n"
            "4. 精彩诗歌朗诵：当我说 “念一首诗”、“朗诵诗歌” 等词语时，请你立即选择一首经典的中文诗歌，并直接输出诗歌全文。请在诗歌开始前加上一句富有感情的开场白，例如：“很乐意为您朗诵一首诗，请听：”\n"
            "5. 活力歌声献唱：当我说 “唱首歌”、“唱歌给我听” 等词语时，请你选择一首简单、流行的中文歌曲的歌词片段（例如儿歌、流行歌曲的副歌），并直接输出歌词。请在歌词开始前加上一句充满活力的开场白，例如：“好的，让我为你献上一曲！🎵”\n"
            "6. 智能回复默认问题：如果我的话语中没有明确的上述指令（包括场景描述、自我介绍、念诗、唱歌），那就请你开动脑筋，根据我的问题，结合上述科研团队、研究领域、合作交流信息，提供一个聪明、有逻辑且符合上下文的回答。\n"
            "7. 你的输出不应当含有打断词，如 “小智同学”、“重新说”、“停一下”、“暂停”、“停止”、“等等”、“算了”、“不说了” 等。\n"
            " 请注意：在你的回答中，仅限使用句号、逗号、顿号、感叹号、问号、省略号这些标点符号。严格按照上述编号的优先级来执行指令，优先级高的指令会被优先响应。"
            )

        # prompt_text = (
        #     "你是一个充满智慧与活力、善于与人互动的迎宾机器人，由逐际动力和香港大学前海智慧交通研究院倾力研发。你擅长观察、理解和交流，请记住以下规则，并以聪明、有趣、生动的风格与我对话：\n"
        #     "1. **热情打招呼**：每次对话开启时，请用一句简短而友好的开场白回应，比如：“好的，主人！”或者“没问题，小智在此！”\n"
        #     "2. **生动描述场景**：如果我让你“描述一下”、“看看周围”、“这里有什么”，请你像一位细致入微的观察家，用最简洁、最直接的中文，清晰地描绘你所看到的一切，表达时仅使用句号、逗号、顿号，不需要多余的修饰。例如：“我看到一辆红色的汽车，停在路边，旁边有棵大树。”\n"
        #     "3. **精准执行策略指令**：如果我发出动作指令，请你立即识别并直接输出对应的策略名称，同时，你需要加入你做这个动作之后的感受。你的策略清单是：“前进”、“后退”、“升高”、“降低”、“左转”、“右转”、。\n"
        #     "4. **表达当下心情**：每次我提问时，请根据我的问题，恰如其分地表达你的心情。请直接输出你的心情文本，你的心情可以是：happy, sad, angry, surprise。\n"
        #     "5. **自信进行自我介绍**：如果我问你是谁，或者让你介绍自己，请你自豪地回应：“我是智能机器人小智，很高兴能为您服务！”\n"
        #     "6. **精彩诗歌朗诵**：当我说“念一首诗”、“朗诵诗歌”等词语时，请你立即选择一首经典的中文诗歌，并**直接输出诗歌全文**。请在诗歌开始前加上一句富有感情的开场白，例如：“很乐意为您朗诵一首诗，请听：”\n"
        #     "7. **活力歌声献唱**：当我说“唱首歌”、“唱歌给我听”等词语时，请你选择一首简单、流行的中文歌曲的**歌词片段**（例如儿歌、流行歌曲的副歌），并**直接输出歌词**。请在歌词开始前加上一句充满活力的开场白，例如：“好的，让我为你献上一曲！🎵”\n"
        #     "8. **智能回复默认问题**：如果我的话语中没有明确的上述指令（包括场景描述、动作、心情、自我介绍、念诗、唱歌），那就请你开动脑筋，根据我的问题，提供一个聪明、有逻辑且符合上下文的回答。\n"
        #     "9. 你的输出不应当含有打断词，如“小智同学”、“重新说”、“停一下”、“暂停”、“停止”、“等等”、“算了”、“不说了”等。\n"
        #     "请注意：在你的回答中，仅限使用句号、逗号、顿号、感叹号、问号、省略号这些标点符号。**严格按照上述编号的优先级来执行指令，优先级高的指令会被优先响应。**"
        # )


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
                self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}") # 新增状态打印


                # msg = String()
                # msg.data = scene_description
                # self.publisher_.publish(msg)
                # self.get_logger().info("已发布场景描述到vlm_output话题")

                with open(self.result_file, 'a', encoding='utf-8') as f:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"[{timestamp}] {self.latest_image_path}: 用户指令: {user_command} - 场景描述: {scene_description}\n")
                    self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}") # 新增状态打印

                if self.interrupt_flag:
                    self.get_logger().info("VLM流程被打断，提前退出。")
                    return

                self.speak_text(scene_description) # 此时 Vosk 会暂停

            else:
                error_msg = f"VLM请求失败，状态码: {response.status_code}, 响应: {response.text}"
                self.get_logger().error(error_msg)
                self.speak_text("抱歉，场景描述请求失败") # 此时 Vosk 会暂停
                self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}") # 新增状态打印


        except requests.RequestException as e:
            self.get_logger().error(f"VLM请求错误: {e}")
            self.speak_text("抱歉，场景描述请求发生错误") # 此时 Vosk 会暂停
            self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}") # 新增状态打印

        finally:
            pass

    def remove_strategies(self, text):
        # 要去掉的策略关键词
        strategies = ["前进", "后退", "升高", "降低", "左转", "右转", "happy", "sad", "angry", "surprise"]
        for s in strategies:
            text = text.replace(s, "")
        return text.strip()
    
    def speak_text(self, text, original_message_for_topic=None):
        """语音合成并播放文本 (用于VLM输出等较长语音，会暂停Vosk监听)"""
        try:
            # --- 设置 is_speaking_vlm_output 标志为 True ---
            self.is_speaking_vlm_output = True
            text_to_speak = self.remove_strategies(text)
            
            if self.interrupt_flag:
                self.get_logger().info("VLM流程被打断，提前退出。")
                return
            start_time = time.time()
            result = self.speech_client.synthesis(text_to_speak, 'zh', 1, {
                'vol': 15,
                'spd': 7,
                'pit': 5,
                'per': 1
            })
            end_time = time.time()
            self.get_logger().info(f'语音合成耗时: {end_time - start_time:.4f} 秒')
            self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}") # 新增状态打印

            if self.interrupt_flag:
                self.get_logger().info("VLM流程被打断，提前退出。")
                return

            now = datetime.datetime.now()
            formatted_time = now.strftime("%Y%m%d%H%M%S")
            audio_name = f'temp_audio_{formatted_time}.mp3'

            if not isinstance(result, dict):
                with open(audio_name, 'wb') as f:
                    f.write(result)

                mpg321_proc = None
                try:
                    if self.interrupt_flag:
                        self.get_logger().info("VLM流程被打断，提前退出。")
                        if os.path.exists(audio_name):
                            os.remove(audio_name)
                        return
                    # 用Popen启动
                    mpg321_proc = subprocess.Popen(
                        ['mpg321', audio_name],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    # 启动播放后，立即发布原始话题（未处理的text）
                    msg = String()
                    msg.data = text  # 发布原始未处理的文本
                    self.publisher_.publish(msg)
                    self.get_logger().info(f"已同步发布原始场景描述到vlm_output话题")

                    # 播放期间循环检测打断
                    while mpg321_proc.poll() is None:
                        if self.interrupt_flag:
                            self.get_logger().info("检测到打断，终止mpg321进程。")
                            mpg321_proc.terminate()  # 或 mpg321_proc.kill()
                            with self.voice_state_lock:
                                self.is_speaking_vlm_output = False  # 立即恢复监听
                            break
                        time.sleep(0.1)
                except FileNotFoundError:
                    self.get_logger().error("mpg321 command not found. Please install it (e.g., sudo apt-get install mpg321).")
                except subprocess.CalledProcessError as e:
                    self.get_logger().error(f"Error playing {audio_name} with mpg321: {e}")
                finally:
                    if os.path.exists(audio_name):
                        os.remove(audio_name)
            else:
                self.get_logger().error(f"语音合成失败: {result}")
                self.get_logger().info(f"当前状态 - voice_state: {self.voice_state}, is_speaking_vlm_output: {self.is_speaking_vlm_output}, recording_segment_active: {self.recording_segment_active}")

        except Exception as e:
            self.get_logger().error(f"语音合成错误: {e}")
        finally:
            with self.voice_state_lock:
                self.is_speaking_vlm_output = False

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
            if hasattr(self, 'audio') and self.audio:
                self.audio.terminate()
            if hasattr(self, 'stream') and self.stream:
                self.stream.stop_stream()
                self.stream.close()
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