# hqiit_vlm

这是一个基于 ROS 2 的视觉语言模型（VLM）机器人控制系统。该项目整合了视觉感知、语音交互、机器人表情显示以及运动控制功能，旨在实现一个能够通过语音命令进行互动并具备情感表达能力的智能机器人。

---

## 文件说明

* `vlm_pub.py`：**核心 VLM 节点**。该节点订阅摄像头图像，处理语音输入，并通过 API 与 VLM 交互，最终将 VLM 的输出（指令和文字回复）发布到其他话题。
* `camera_publisher.py`：**摄像头发布节点**。负责从本地摄像头（通常是 `/dev/video0`）捕获视频流，并将每一帧图像作为 `sensor_msgs/Image` 消息发布到 `/camera_image` 话题。
* `emotion_subscriber.py`：**表情显示节点**。该节点订阅 `detected_objects` 话题，并使用 `pygame` 库在全屏窗口上绘制机器人表情，实现了情感反馈的视觉化。
* `msg_trans.py`：**命令转换节点**。订阅 VLM 发布的 `/vlm_output` 话题，解析其中的中文命令，并将其转换为 ROS 2 机器人运动控制的 `/cmd_vel` 话题所需的 `Twist` 消息。
* `emotion_publisher.py`：**表情测试节点**。一个简单的测试节点，用于周期性地向 `detected_objects` 话题发布预设的表情字符串，方便在没有 VLM 节点的情况下测试表情显示功能。

---

## 项目依赖与环境配置

### 硬件
* 一台运行 ROS 2 的计算机
* 一个摄像头（Webcam）
* 麦克风和扬声器

### 软件
* **操作系统**：Ubuntu 20.04/22.04 
* **ROS 2**：foxy 或更高版本
* **Python 3**
* **Python 库**：
    您可以使用 `pip` 命令安装以下依赖：
    ```bash
    pip install rclpy opencv-python-headless cv-bridge pygame pyaudio pypinyin requests vosk Levenshtein aip
    ```
    > **注意**：`opencv-python` 库在某些环境下可能需要单独安装 `opencv-python-headless`。`cv-bridge` 需要从 ROS 仓库安装。

### API 与模型
1.  **百度 AI 平台（AipSpeech）**：
    * 在百度 AI 平台创建应用，获取 `APP_ID`, `API_KEY` 和 `SECRET_KEY`。
    * 在 `vlm_pub.py` 文件中，将这些密钥替换为您的信息。
2.  **Vosk 语音识别模型**：
    * 从 [Vosk 官网](https://alphacephei.com/vosk/models) 下载一个中文语音识别模型（例如 `vosk-model-cn-0.22.zip`）。
    * 将模型解压，并命名为 `model`，放置在 `vlm_pub.py` 脚本所在的目录下。

---

## 使用方法


3.  **运行各个节点**：
    在不同的终端中分别运行以下命令来启动所有核心节点。
    * **启动摄像头发布节点**：
        ```bash
        python3 camera_publisher.py
        ```
    * **启动表情显示节点**：
        ```bash
        python3 emotion_subscriber.py
        ```
    * **启动命令转换节点**：
        ```bash
        python3 msg_trans.py
        ```
    * **启动核心 VLM 节点**：
        ```bash
        python3 vlm_pub.py
        ```

---
