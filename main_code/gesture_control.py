import cv2
import autopy #鼠标控制
import numpy as np
import time
import mediapipe as mp #识别手势
import math
from threading import Thread, Event

class GestureController:
    def __init__(self):
        # 初始化MediaPipe手部检测模块
        self.mphands = mp.solutions.hands
        # 设置手部检测和跟踪的置信度阈值
        self.hands = self.mphands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.mpDraw = mp.solutions.drawing_utils
        # 定义关键点绘制样式（蓝色，粗4px）和连接线样式（红色，粗4px）
        self.pointStyle = self.mpDraw.DrawingSpec(color=(255, 0, 0), thickness=4)
        self.lineStyle = self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=4)
        
        # 控制线程运行的变量
        self.is_running = False
        self.stop_event = Event()
        self.control_thread = None
        
        # 初始化其他变量
        # 鼠标平滑移动相关变量
        self.plocx, self.plocy, self.smooth = 0, 0, 6
        # 视频帧尺寸和屏幕尺寸
        self.resize_w, self.resize_h = 640, 480
        # 手势控制区域（左上角和右下角坐标）
        self.pt1, self.pt2 = (50, 50), (600, 300)
        # 获取屏幕分辨率
        self.wcam, self.hcam = autopy.screen.size()
        self.pTime = 0 # 用于计算帧率的时间戳

    # 启动手势控制
    def start(self):
        if not self.is_running:
            self.is_running = True
            self.stop_event.clear()
             # 创建并启动控制线程
            self.control_thread = Thread(target=self._control_loop)
            self.control_thread.start()
    
    # 停止手势控制         
    def stop(self):    
        if self.is_running:
            self.is_running = False
            self.stop_event.set()
            if self.control_thread:
                 # 等待控制线程结束
                self.control_thread.join()
    
    # 手势控制主循环           
    def _control_loop(self):
        cap = cv2.VideoCapture(0) # 摄像头
        
        while self.is_running and not self.stop_event.is_set():
            ret, self.img = cap.read()
            if ret:
                # 水平翻转图像（便于镜像操作）
                self.img = cv2.flip(self.img, 1)
                 # 处理手势识别
                self._gesture_control()
                
                # 手势控制区域的边框
                cv2.rectangle(self.img, self.pt1, self.pt2, (138, 12, 92), 3)
                self._frame_rate()
                
                cv2.imshow("Gesture Control", self.img)
                
            if cv2.waitKey(1) == 27 or not self.is_running: # 按ESC键退出或根据运行状态退出
                break
                
        cap.release()
        cv2.destroyAllWindows() # 释放资源
    
    # 主要的手势识别处理函数    
    def _gesture_control(self):
        # 将BGR图像转换为RGB（MediaPipe需要RGB图像）
        imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
         # 处理手部检测
        result = self.hands.process(imgRGB)
        if result.multi_hand_landmarks:
            # 遍历检测到的每只手
            for self.handLms in result.multi_hand_landmarks:
                # 绘制手部关键点和连接
                self.mpDraw.draw_landmarks(self.img, self.handLms, self.mphands.HAND_CONNECTIONS,
                                         self.pointStyle, self.lineStyle)
                 # 计算手指关键点和执行手势判断
                self._count()
                
    def _count(self):
        # 计算手掌根部(0)到食指根部(5)的距离（用于手势判断参考）
        p0x = self.handLms.landmark[0].x
        p0y = self.handLms.landmark[0].y
        p5x = self.handLms.landmark[5].x
        p5y = self.handLms.landmark[5].y
        distance_0_5 = pow(p0x - p5x, 2) + pow(p0y - p5y, 2)
        self.dis05 = pow(distance_0_5, 0.5)
        
        # 处理拇指指尖(4)的坐标和距离
        p4x = self.handLms.landmark[4].x
        p4y = self.handLms.landmark[4].y
        p4_x = math.ceil(p4x * self.resize_w)
        p4_y = math.ceil(p4y * self.resize_h) # 将归一化坐标转换为像素坐标
        self.thumb = (p4_x, p4_y)
        # 计算拇指到食指根部（4到5）的距离（用于点击）
        distance_4_5 = pow(p4x - p5x, 2) + pow(p4y - p5y, 2)
        self.dis45 = pow(distance_4_5, 0.5)
        
        # 处理食指指尖(8)的坐标
        p8x = self.handLms.landmark[8].x
        p8y = self.handLms.landmark[8].y
        self.p8_x = math.ceil(p8x * self.resize_w)
        self.p8_y = math.ceil(p8y * self.resize_h)
        self.index = (self.p8_x, self.p8_y)
         # 计算手掌根部到食指指尖的距离（0到8）
        distance_0_8 = pow(p0x - p8x, 2) + pow(p0y - p8y, 2)
        self.dis08 = pow(distance_0_8, 0.5)
        
        # 处理中指指尖(12)的坐标
        p12x = self.handLms.landmark[12].x
        p12y = self.handLms.landmark[12].y
        self.p12_x = math.ceil(p12x * self.resize_w)
        self.p12_y = math.ceil(p12y * self.resize_h)
        self.middle = (self.p12_x, self.p12_y)
        # 计算手掌根部到中指指尖和食指到中指的距离（0到8 和 0到12）
        distance_0_12 = pow(p0x - p12x, 2) + pow(p0y - p12y, 2)
        self.dis012 = pow(distance_0_12, 0.5)
        distance_8_12 = pow(p8x - p12x, 2) + pow(p8y - p12y, 2)
        self.dis812 = pow(distance_8_12, 0.5)
        
        # 处理无名指指尖(16)的坐标
        p16x = self.handLms.landmark[16].x
        p16y = self.handLms.landmark[16].y
        self.p16_x = math.ceil(p16x * self.resize_w)
        self.p16_y = math.ceil(p16y * self.resize_h)
        self.ring = (self.p16_x, self.p16_y)
        # 计算手掌根部到无名指指尖的距离（0到16）
        distance_0_16 = pow(p0x - p16x, 2) + pow(p0y - p16y, 2)
        self.dis016 = pow(distance_0_16, 0.5)
        
        # 处理尾指指尖(20)的坐标
        p20x = self.handLms.landmark[20].x
        p20y = self.handLms.landmark[20].y
        self.p20_x = math.ceil(p20x * self.resize_w)
        self.p20_y = math.ceil(p20y * self.resize_h)
        self.caudal = (self.p20_x, self.p20_y)
         # 计算手掌根部到尾指指尖和无名指到尾指的距离（0到20 和 20到16）
        distance_0_20 = pow(p0x - p20x, 2) + pow(p0y - p20y, 2)
        self.dis020 = pow(distance_0_20, 0.5)
        distance_16_20 = pow(p16x - p20x, 2) + pow(p16y - p20y, 2)
        self.dis1620 = pow(distance_16_20, 0.5)
        
        # 绘制关键点（拇指，食指，中值用天蓝色，剩下用绿色）
        self.img = cv2.circle(self.img, self.index, 10, (255, 255, 0), cv2.FILLED)
        self.img = cv2.circle(self.img, self.thumb, 10, (255, 255, 0), cv2.FILLED)
        self.img = cv2.circle(self.img, self.middle, 10, (255, 255, 0), cv2.FILLED)
        self.img = cv2.circle(self.img, self.ring, 10, (0, 255, 0), cv2.FILLED)
        self.img = cv2.circle(self.img, self.caudal, 10, (0, 255, 0), cv2.FILLED)
        
        self._execute_judge()# 执行手势判断和相应操作
        
    def _execute_judge(self):
        # 判断手势并执行相应操作
        # 鼠标移动控制  将食指指尖位置映射到屏幕坐标
        sx = np.interp(self.p8_x, (self.pt1[0], self.pt2[0]), (0, self.wcam))
        sy = np.interp(self.p8_y, (self.pt1[1], self.pt2[1]), (0, self.hcam))

        # 平滑处理鼠标移动（减少抖动）
        clocx = self.plocx + (sx - self.plocx) / self.smooth
        clocy = self.plocy + (sy - self.plocy) / self.smooth
        
        try:
             # 移动鼠标到计算出的坐标
            autopy.mouse.move(clocx, clocy)
            self.plocx, self.plocy = clocx, clocy
            
            # 拇指和食指根部距离很近时，执行左键点击
            if self.dis45 < 0.02:
                autopy.mouse.click(autopy.mouse.Button.LEFT)
                time.sleep(0.2)
                
            # 食指和中指弯曲时，执行向下翻页
            if self.dis08 < self.dis05 * 0.8 and self.dis012 < self.dis05 * 0.8:
                autopy.key.toggle(autopy.key.Code.DOWN_ARROW, True)
                time.sleep(0.5)
                
            # 食指伸直、中指弯曲时，执行右键点击
            if self.dis012 < self.dis05 and self.dis08 > self.dis05 * 1.4 and self.dis812 > 0.07:
                autopy.mouse.click(autopy.mouse.Button.RIGHT)
                time.sleep(0.2)
                
            # 无名指和尾指弯曲时，执行向上翻页
            if self.dis016 < self.dis05 * 0.6 and self.dis020 < self.dis05 * 0.6:
                autopy.key.toggle(autopy.key.Code.UP_ARROW, True)
                time.sleep(0.5)
        except Exception as e:
            print(f"执行手势操作时出错: {str(e)}")
            
    def _frame_rate(self):
        # 计算并显示帧率
        self.cTime = time.time()
        self.fps = 1 / (self.cTime - self.pTime)
        self.pTime = self.cTime
        cv2.putText(self.img, str(int(self.fps)), (70, 40),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)