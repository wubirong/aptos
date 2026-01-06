#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手势控制电脑桌面应用
支持手势识别来控制鼠标、键盘和音量
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from PIL import Image, ImageTk
import math

class GestureController:
    def __init__(self):
        # MediaPipe初始化
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 摄像头初始化
        self.cap = None
        self.is_running = False
        
        # 手势控制参数
        self.prev_x, self.prev_y = 0, 0
        self.smoothing_factor = 0.7
        self.click_threshold = 0.05
        self.scroll_threshold = 0.03
        
        # 手势状态
        self.gesture_states = {
            'fist': False,
            'peace': False,
            'thumb_up': False,
            'thumb_down': False,
            'open_hand': False
        }
        
        # 控制模式
        self.control_mode = "mouse"  # mouse, keyboard, volume
        
    def initialize_camera(self):
        """初始化摄像头"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("无法打开摄像头")
            return True
        except Exception as e:
            print(f"摄像头初始化失败: {e}")
            return False
    
    def get_hand_landmarks(self, frame):
        """获取手部关键点"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]
        return None
    
    def calculate_distance(self, point1, point2):
        """计算两点间距离"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def detect_gesture(self, landmarks):
        """检测手势类型"""
        if not landmarks:
            return "none"
        
        # 获取关键点
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        ring_tip = landmarks.landmark[16]
        pinky_tip = landmarks.landmark[20]
        
        thumb_mcp = landmarks.landmark[3]
        index_mcp = landmarks.landmark[5]
        middle_mcp = landmarks.landmark[9]
        ring_mcp = landmarks.landmark[13]
        pinky_mcp = landmarks.landmark[17]
        
        # 检测手指是否伸直
        thumb_up = thumb_tip.y < thumb_mcp.y
        index_up = index_tip.y < index_mcp.y
        middle_up = middle_tip.y < middle_mcp.y
        ring_up = ring_tip.y < ring_mcp.y
        pinky_up = pinky_tip.y < pinky_mcp.y
        
        fingers_up = [thumb_up, index_up, middle_up, ring_up, pinky_up]
        fingers_count = sum(fingers_up)
        
        # 手势识别
        if fingers_count == 0:
            return "fist"
        elif fingers_count == 2 and index_up and middle_up:
            return "peace"
        elif fingers_count == 1 and thumb_up:
            return "thumb_up"
        elif fingers_count == 1 and not thumb_up and not index_up:
            return "thumb_down"
        elif fingers_count == 5:
            return "open_hand"
        else:
            return "other"
    
    def control_mouse(self, landmarks, frame_width, frame_height):
        """鼠标控制"""
        if not landmarks:
            return
        
        # 获取食指指尖位置
        index_tip = landmarks.landmark[8]
        
        # 转换为屏幕坐标
        screen_width, screen_height = pyautogui.size()
        x = int(index_tip.x * screen_width)
        y = int(index_tip.y * screen_height)
        
        # 平滑移动
        x = int(self.prev_x * self.smoothing_factor + x * (1 - self.smoothing_factor))
        y = int(self.prev_y * self.smoothing_factor + y * (1 - self.smoothing_factor))
        
        # 移动鼠标
        pyautogui.moveTo(x, y)
        self.prev_x, self.prev_y = x, y
    
    def control_click(self, gesture):
        """点击控制"""
        if gesture == "fist":
            if not self.gesture_states['fist']:
                pyautogui.click()
                self.gesture_states['fist'] = True
        else:
            self.gesture_states['fist'] = False
    
    def control_scroll(self, gesture):
        """滚动控制"""
        if gesture == "thumb_up":
            if not self.gesture_states['thumb_up']:
                pyautogui.scroll(3)
                self.gesture_states['thumb_up'] = True
        elif gesture == "thumb_down":
            if not self.gesture_states['thumb_down']:
                pyautogui.scroll(-3)
                self.gesture_states['thumb_down'] = True
        else:
            self.gesture_states['thumb_up'] = False
            self.gesture_states['thumb_down'] = False
    
    def control_keyboard(self, gesture):
        """键盘控制"""
        if gesture == "peace":
            if not self.gesture_states['peace']:
                pyautogui.press('space')  # 暂停/播放
                self.gesture_states['peace'] = True
        elif gesture == "open_hand":
            if not self.gesture_states['open_hand']:
                pyautogui.press('esc')  # 退出
                self.gesture_states['open_hand'] = True
        else:
            self.gesture_states['peace'] = False
            self.gesture_states['open_hand'] = False
    
    def process_frame(self, frame):
        """处理视频帧"""
        frame = cv2.flip(frame, 1)  # 水平翻转
        frame_height, frame_width = frame.shape[:2]
        
        # 获取手部关键点
        landmarks = self.get_hand_landmarks(frame)
        
        if landmarks:
            # 检测手势
            gesture = self.detect_gesture(landmarks)
            
            # 根据控制模式执行相应操作
            if self.control_mode == "mouse":
                self.control_mouse(landmarks, frame_width, frame_height)
                self.control_click(gesture)
                self.control_scroll(gesture)
            elif self.control_mode == "keyboard":
                self.control_keyboard(gesture)
            
            # 绘制手部关键点和手势信息
            self.mp_drawing.draw_landmarks(
                frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # 显示手势信息
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Mode: {self.control_mode}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
    
    def start_camera(self):
        """启动摄像头"""
        if not self.initialize_camera():
            return False
        
        self.is_running = True
        return True
    
    def stop_camera(self):
        """停止摄像头"""
        self.is_running = False
        if self.cap:
            self.cap.release()
    
    def set_control_mode(self, mode):
        """设置控制模式"""
        self.control_mode = mode


class GestureControlApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("手势控制电脑应用")
        self.root.geometry("800x600")
        
        # 初始化手势控制器
        self.controller = GestureController()
        
        # 创建界面
        self.create_widgets()
        
        # 摄像头线程
        self.camera_thread = None
        self.is_camera_running = False
        
    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 视频显示区域
        self.video_label = ttk.Label(main_frame, text="摄像头未启动")
        self.video_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # 控制按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="启动摄像头", 
                                      command=self.start_camera)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="停止摄像头", 
                                     command=self.stop_camera, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=5)
        
        # 控制模式选择
        mode_frame = ttk.LabelFrame(main_frame, text="控制模式", padding="10")
        mode_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        self.mode_var = tk.StringVar(value="mouse")
        
        ttk.Radiobutton(mode_frame, text="鼠标控制", variable=self.mode_var, 
                       value="mouse", command=self.change_mode).grid(row=0, column=0, padx=5)
        ttk.Radiobutton(mode_frame, text="键盘控制", variable=self.mode_var, 
                       value="keyboard", command=self.change_mode).grid(row=0, column=1, padx=5)
        
        # 手势说明
        info_frame = ttk.LabelFrame(main_frame, text="手势说明", padding="10")
        info_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky=(tk.W, tk.E))
        
        info_text = """
鼠标控制模式：
• 食指移动：控制鼠标指针
• 握拳：左键点击
• 大拇指向上：向上滚动
• 大拇指向下：向下滚动

键盘控制模式：
• 剪刀手(✌)：空格键(暂停/播放)
• 张开手掌：ESC键(退出)
        """
        
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT).grid(row=0, column=0)
        
    def start_camera(self):
        """启动摄像头"""
        if self.controller.start_camera():
            self.is_camera_running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            
            # 启动摄像头线程
            self.camera_thread = threading.Thread(target=self.camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
        else:
            messagebox.showerror("错误", "无法启动摄像头，请检查摄像头是否连接")
    
    def stop_camera(self):
        """停止摄像头"""
        self.is_camera_running = False
        self.controller.stop_camera()
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.video_label.config(text="摄像头已停止")
    
    def change_mode(self):
        """改变控制模式"""
        mode = self.mode_var.get()
        self.controller.set_control_mode(mode)
    
    def camera_loop(self):
        """摄像头循环"""
        while self.is_camera_running and self.controller.is_running:
            ret, frame = self.controller.cap.read()
            if ret:
                # 处理帧
                processed_frame = self.controller.process_frame(frame)
                
                # 转换为PIL图像并显示
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                pil_image = pil_image.resize((640, 480), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(pil_image)
                
                # 更新显示
                self.video_label.config(image=photo, text="")
                self.video_label.image = photo
            else:
                break
            
            time.sleep(0.03)  # 约30fps
    
    def run(self):
        """运行应用"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """关闭应用时的清理工作"""
        if self.is_camera_running:
            self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    app = GestureControlApp()
    app.run()