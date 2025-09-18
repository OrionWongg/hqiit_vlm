import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import pygame
import sys
import math
import time
import random

class EmotionSubscriber(Node):
    def __init__(self):
        super().__init__('emotion_subscriber')
        self.subscription = self.create_subscription(
            String,
            'detected_objects',
            self.emotion_callback,
            10)
        
        pygame.init()
        self.info = pygame.display.Info()
        # 获取屏幕的实际分辨率（用于全屏显示）
        self.screen_width = self.info.current_w
        self.screen_height = self.info.current_h
        self.fullscreen = True  # 初始化为全屏状态
        
        # 启动时直接全屏显示
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height), 
            pygame.FULLSCREEN | pygame.RESIZABLE
        )
        pygame.display.set_caption("机器人表情")
        
        # 创建固定尺寸的虚拟绘制表面
        self.virtual_surface = pygame.Surface((600, 530))
        
        # 表情状态管理（其余参数保持不变）
        self.current_emotion = "idle"
        self.eye_width_ratio = 1.0
        self.eye_height_ratio = 1.3
        self.eye_color = (200, 200, 255)
        
        # 表情持续时间控制
        self.emotion_start_time = time.time()
        self.emotion_duration = 0.0
        
        # 其他动画参数（保持不变）
        self.eye_shift = 0.0
        self.shift_direction = 1
        self.last_shift_time = time.time()
        self.shift_interval = random.uniform(1, 4)
        
        self.is_winking = False
        self.wink_eye = 0
        self.wink_progress = 0.0
        self.last_wink_time = time.time()
        self.wink_interval = random.uniform(3, 5)
        
        self.animation_time = time.time()
        self.blink_timer = time.time()
        self.blink_duration = random.uniform(0.05, 0.15)
        self.blink_interval = random.uniform(1, 3)
        self.is_blinking = False

        # 新增动画参数（保持不变）
        self.confused_shake_offset_x = 0.0
        self.confused_shake_offset_y = 0.0
        self.confused_shake_timer = time.time()
        self.confused_shake_interval = 0.05 # 抖动频率
        self.confused_shake_magnitude = 2 # 抖动幅度

        self.tear_drop_visible = False
        self.tear_drop_timer = time.time()
        self.tear_drop_interval = random.uniform(2, 5) # 泪滴出现间隔
        self.tear_drop_progress = 0.0
        self.tear_drop_duration = 1.0 # 泪滴动画持续时间
        
        # 初始化表情关键词映射
        self.initialize_emotion_map()

    def initialize_emotion_map(self):
        """初始化表情关键词映射表"""
        self.emotion_map = {
            # 格式: (关键词列表, 表情状态, 默认持续时间)
            "happy": (["happy", "高兴", "开心", "快乐"], "happy", 5.0),
            "sad": (["sad", "悲伤", "难过", "伤心"], "sad", 5.0),
            "surprised": (["surprised", "惊讶", "吃惊", "惊奇"], "surprised", 5.0),
            "angry": (["angry", "生气", "愤怒", "恼怒"], "angry", 5.0),
            "confused": (["confused", "困惑", "迷茫", "疑惑"], "confused", 5.0)
        }

    def emotion_callback(self, msg):
        message = msg.data
        self.get_logger().info(f'Received message: {message}')
        
        # 解析消息中的表情和持续时间
        parsed_emotion, duration = self.parse_emotion_message(message)
        
        if parsed_emotion:
            self.current_emotion = parsed_emotion
            self.emotion_duration = duration
            self.emotion_start_time = time.time()
            self.set_emotion_params(parsed_emotion)
            self.get_logger().info(f'Setting emotion to {parsed_emotion} for {duration} seconds')
        else:
            self.get_logger().info('No valid emotion found in message')

    def parse_emotion_message(self, message):
        """通过遍历关键词映射表提取表情信息"""
        # 提取消息中的持续时间
        duration = self.extract_duration(message)
        
        # 遍历表情映射表，查找匹配的关键词
        for emotion, (keywords, state, default_duration) in self.emotion_map.items():
            for keyword in keywords:
                if keyword in message:
                    # 使用消息中的持续时间或默认持续时间
                    effective_duration = duration if duration > 0 else default_duration
                    return state, effective_duration
        
        return None, 0.0

    def extract_duration(self, message):
        """从消息中提取持续时间"""
        import re
        match = re.search(r'(\d+\.\d+|\d+)\s*seconds?', message)
        return float(match.group(1)) if match else 0.0

    def set_emotion_params(self, emotion):
        # 重置所有动画参数，以防切换表情时旧动画参数残留
        self.confused_shake_offset_x = 0.0
        self.confused_shake_offset_y = 0.0
        self.tear_drop_visible = False
        self.tear_drop_progress = 0.0

        if emotion == "happy":
            self.eye_width_ratio = 1.3  # 开心时眼睛更宽
            self.eye_height_ratio = 1.5  # 高度调整
            self.eye_color = (180, 180, 220)
        elif emotion == "sad":
            self.eye_width_ratio = 0.8
            self.eye_height_ratio = 0.9
            self.eye_color = (255, 255, 255)
        elif emotion == "surprised":
            self.eye_width_ratio = 1.3
            self.eye_height_ratio = 1.6
            self.eye_color = (180, 180, 220)
        elif emotion == "angry":
            self.eye_width_ratio = 0.8  # 生气时眼睛变窄
            self.eye_height_ratio = 0.7  # 高度降低
            self.eye_color = (255, 1, 1)  # 红色眼睛
        elif emotion == "confused" or emotion == "idle": # idle时也保持默认设置
            self.eye_width_ratio = 1.0
            self.eye_height_ratio = 1.3
            self.eye_color = (180, 180, 220)

    def run(self):
        last_time = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # 检查表情持续时间
            if self.emotion_duration > 0 and current_time - self.emotion_start_time > self.emotion_duration:
                self.current_emotion = "idle"
                self.set_emotion_params("idle")
                self.emotion_duration = 0.0
                self.get_logger().info('Emotion duration expired, returning to idle')
            
            self.handle_random_blinking(current_time)
            self.handle_eye_shifting(dt)
            self.handle_winking(dt)
            self.handle_confused_shake(current_time) # 处理困惑抖动
            self.handle_tear_drop(current_time, dt) # 处理泪滴动画
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                    self.toggle_fullscreen()
                elif event.type == pygame.VIDEORESIZE:  # 处理窗口大小调整事件
                    if not self.fullscreen:
                        self.screen_width, self.screen_height = event.w, event.h
                        self.screen = pygame.display.set_mode(
                            (self.screen_width, self.screen_height), 
                            pygame.RESIZABLE
                        )
            
            # 在虚拟表面上绘制表情
            draw_emotion(self.virtual_surface, self.current_emotion, 
                        self.eye_width_ratio, self.eye_height_ratio, 
                        self.eye_color, self.is_blinking,
                        self.eye_shift, self.is_winking, self.wink_eye, self.wink_progress,
                        self.confused_shake_offset_x, self.confused_shake_offset_y,
                        self.tear_drop_visible, self.tear_drop_progress)
            
            # 缩放虚拟表面到实际屏幕尺寸
            scaled_surface = pygame.transform.smoothscale(
                self.virtual_surface, 
                (self.screen.get_width(), self.screen.get_height())
            )
            self.screen.blit(scaled_surface, (0, 0))
            pygame.display.flip()

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            # 切换到全屏模式：使用当前屏幕分辨率
            self.screen = pygame.display.set_mode(
                (0, 0), 
                pygame.FULLSCREEN | pygame.RESIZABLE
            )
            self.screen_width, self.screen_height = self.screen.get_size()
        else:
            # 切换到窗口模式：使用默认尺寸
            self.screen = pygame.display.set_mode(
                (600, 600), 
                pygame.RESIZABLE
            )
            self.screen_width, self.screen_height = 600, 600

    def handle_random_blinking(self, current_time):
        if not self.is_blinking and not self.is_winking:
            if current_time - self.blink_timer > self.blink_interval:
                self.is_blinking = True
                self.blink_duration = random.uniform(0.05, 0.15)
                self.blink_timer = current_time
                self.blink_interval = random.uniform(1, 8) * (0.8 + random.random() * 0.4)
        elif self.is_blinking:
            if current_time - self.blink_timer > self.blink_duration:
                self.is_blinking = False
                self.blink_timer = current_time

    def handle_eye_shifting(self, dt):
        shift_speed = 0.1
        current_time = time.time()
        if current_time - self.last_shift_time > self.shift_interval:
            if random.random() < 0.5:
                self.shift_direction *= -1
            self.last_shift_time = current_time
            self.shift_interval = random.uniform(3, 8)
        self.eye_shift += self.shift_direction * shift_speed * min(1.0, dt * 60)
        self.eye_shift = max(-0.2, min(0.2, self.eye_shift))

    def handle_winking(self, dt):
        wink_speed = 0.05
        current_time = time.time()
        if not self.is_winking and current_time - self.last_wink_time > self.wink_interval:
            self.is_winking = True
            self.wink_eye = random.randint(0, 1)
            self.wink_progress = 0.0
            self.last_wink_time = current_time
            self.wink_interval = random.uniform(10, 20)
        if self.is_winking:
            self.wink_progress += wink_speed * min(1.0, dt * 60)
            if self.wink_progress >= 1.0:
                self.is_winking = False
                self.wink_progress = 0.0

    def handle_confused_shake(self, current_time):
        if self.current_emotion == "confused":
            if current_time - self.confused_shake_timer > self.confused_shake_interval:
                self.confused_shake_offset_x = random.uniform(-self.confused_shake_magnitude, self.confused_shake_magnitude)
                self.confused_shake_offset_y = random.uniform(-self.confused_shake_magnitude, self.confused_shake_magnitude)
                self.confused_shake_timer = current_time
        else:
            self.confused_shake_offset_x = 0.0
            self.confused_shake_offset_y = 0.0

    def handle_tear_drop(self, current_time, dt):
        if self.current_emotion == "sad":
            if not self.tear_drop_visible and current_time - self.tear_drop_timer > self.tear_drop_interval:
                self.tear_drop_visible = True
                self.tear_drop_progress = 0.0
                self.tear_drop_timer = current_time
            
            if self.tear_drop_visible:
                self.tear_drop_progress += dt / self.tear_drop_duration
                if self.tear_drop_progress >= 1.0:
                    self.tear_drop_visible = False
                    self.tear_drop_timer = current_time # Reset timer for next tear drop
                    self.tear_drop_interval = random.uniform(2, 5) # New random interval
        else:
            self.tear_drop_visible = False
            self.tear_drop_progress = 0.0


def draw_emotion(screen, current, eye_width_ratio, eye_height_ratio, eye_color, is_blinking,
                eye_shift, is_winking, wink_eye, wink_progress,
                confused_shake_offset_x, confused_shake_offset_y,
                tear_drop_visible, tear_drop_progress):
    screen.fill((10, 10, 20))
    
    cx, cy = 300, 300  # 在虚拟表面上的中心点
    head_radius = 230
    
    base_eye_width = head_radius // 2
    base_eye_height = head_radius // 3
    eye_spacing = head_radius // 2
    eye_y = cy - head_radius // 4
    
    shift_amount = eye_spacing * eye_shift
    eye_x1 = cx - eye_spacing + shift_amount + confused_shake_offset_x
    eye_x2 = cx + eye_spacing + shift_amount + confused_shake_offset_x
    eye_y_offset_for_shake = confused_shake_offset_y

    # 绘制眼睛
    for i, eye_x in enumerate([eye_x1, eye_x2]):
        wink_effect = 0.0
        if is_winking and i == wink_eye:
            wink_effect = 4 * wink_progress * (1 - wink_progress)
        current_eye_height = max(1, int(base_eye_height * eye_height_ratio * (1 - wink_effect)))
        current_eye_width = int(base_eye_width * eye_width_ratio)
        
        eye_rect_y = eye_y - current_eye_height // 2 + eye_y_offset_for_shake

        if is_blinking:
            eye_height = 1
            eye_width = base_eye_width
            draw_rounded_rect(screen, 
                             eye_x - eye_width // 2, 
                             eye_y - eye_height // 2 + eye_y_offset_for_shake, 
                             eye_width, 
                             eye_height, 
                             1, 
                             eye_color)
        elif current == "happy":
            # 开心时眼睛：顶部圆角，底部平
            top_radius = current_eye_height // 3
            draw_top_rounded_rect(screen, 
                                  eye_x - current_eye_width // 2, 
                                  eye_rect_y, 
                                  current_eye_width, 
                                  current_eye_height, 
                                  top_radius, 
                                  eye_color)
        elif current == "angry":
            # 生气时眼睛：顶部平，底部圆角（向下弯曲）
            bottom_radius = current_eye_height // 3
            draw_bottom_rounded_rect(screen, 
                                    eye_x - current_eye_width // 2, 
                                    eye_rect_y, 
                                    current_eye_width, 
                                    current_eye_height, 
                                    bottom_radius, 
                                    eye_color)
        else:
            # 其他表情绘制普通圆角矩形
            corner_radius = max(2, current_eye_height // 6)
            draw_rounded_rect(screen, 
                             eye_x - current_eye_width // 2, 
                             eye_rect_y, 
                             current_eye_width, 
                             current_eye_height, 
                             corner_radius, 
                             eye_color)

    # 绘制泪滴
    if tear_drop_visible and current == "sad":
        draw_tear_drop(screen, eye_x1, eye_x2, eye_y + base_eye_height // 2 + eye_y_offset_for_shake, tear_drop_progress)


def draw_top_rounded_rect(surface, x, y, width, height, top_radius, color):
    """绘制顶部带圆角、底部平的矩形（开心表情专用）"""
    radius = min(top_radius, width // 2, height // 2)
    # 顶部左右圆角
    pygame.draw.circle(surface, color, (x + radius, y + radius), radius)
    pygame.draw.circle(surface, color, (x + width - radius, y + radius), radius)
    # 顶部中间矩形（连接两个圆角）
    pygame.draw.rect(surface, color, (x + radius, y, width - 2 * radius, radius))
    # 底部矩形（高度为总高度 - 顶部圆角高度）
    pygame.draw.rect(surface, color, (x, y + radius, width, height - radius))

def draw_bottom_rounded_rect(surface, x, y, width, height, bottom_radius, color):
    """绘制底部带圆角、顶部平的矩形（生气表情专用）"""
    radius = min(bottom_radius, width // 2, height // 2)
    # 底部左右圆角
    pygame.draw.circle(surface, color, (x + radius, y + height - radius), radius)
    pygame.draw.circle(surface, color, (x + width - radius, y + height - radius), radius)
    # 底部中间矩形（连接两个圆角）
    pygame.draw.rect(surface, color, (x + radius, y + height - radius, width - 2 * radius, radius))
    # 顶部矩形（高度为总高度 - 底部圆角高度）
    pygame.draw.rect(surface, color, (x, y, width, height - radius))

def draw_rounded_rect(surface, x, y, width, height, radius, color):
    radius = min(radius, width//2, height//2)
    if radius == 0:
        pygame.draw.rect(surface, color, (x, y, width, height))
        return
    
    pygame.draw.circle(surface, color, (x+radius, y+radius), radius)
    pygame.draw.circle(surface, color, (x+width-radius, y+radius), radius)
    pygame.draw.circle(surface, color, (x+radius, y+height-radius), radius)
    pygame.draw.circle(surface, color, (x+width-radius, y+height-radius), radius)
    
    pygame.draw.rect(surface, color, (x+radius, y, width-2*radius, height))
    pygame.draw.rect(surface, color, (x, y+radius, width, height-2*radius))

def draw_tear_drop(surface, eye1_cx, eye2_cx, eye_bottom_y, progress):
    """绘制泪滴动画"""
    tear_color = (100, 100, 255) # 蓝色泪滴
    
    # 泪滴从两眼之间下方开始下落
    start_x = (eye1_cx + eye2_cx) // 2
    start_y = eye_bottom_y + 10
    
    # 泪滴下落的距离
    drop_distance = 50 
    
    # 泪滴的当前位置
    current_y = start_y + drop_distance * progress
    
    # 泪滴大小随下落逐渐变大
    tear_radius = int(5 * (1 + progress * 0.5)) 

    # 绘制泪滴形状
    pygame.draw.ellipse(surface, tear_color, (start_x - tear_radius, current_y - tear_radius * 2, tear_radius * 2, tear_radius * 4))


def main():
    rclpy.init()
    subscriber = EmotionSubscriber()
    subscriber.run()

if __name__ == '__main__':
    main()
