import cv2
import numpy as np
from typing import Optional, Tuple, Union
import math


class ImageProcessor: #接下来是基础的图像处理功能
    # 存储原始图像和处理后的图像
    def __init__(self):
        self.original_image = None  
        self.image = None 
    
    # 加载图像 字符串或图像数组
    def load_image(self, image):
        if isinstance(image, str):
            self.original_image = cv2.imread(image)
        else:
            self.original_image = image.copy()
        self.image = self.original_image.copy()# 初始化处理
        return self.image is not None
    
    # 重置图像为原始状态
    def reset_image(self):
        if self.original_image is not None:
            self.image = self.original_image.copy()
            return True
        return False

    # 获取处理后的图像
    def get_processed_image(self):
        return self.image

    # 保存处理后的图像
    def save_image(self, file_path):
        if self.image is not None:
            cv2.imwrite(file_path, self.image)
            return True
        return False

    # 基础图像处理功能

    # 椒盐噪声
    def add_salt_pepper_noise(self, prob=0.05):
        if self.original_image is None:
            return None
        self.image = self.original_image.copy()
        output = np.zeros(self.image.shape, np.uint8)
        thres = 1 - prob 
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                rdn = np.random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = self.image[i][j]
        self.image = output
        return self.image

    # 图像平滑
    def mean_blur(self, kernel_size=3): #均值滤波，默认核数为3
        if self.original_image is None:
            return None
        self.image = cv2.blur(self.original_image.copy(), (kernel_size, kernel_size))
        return self.image

    def median_blur(self, kernel_size=3): #中值滤波，默认核数为3
        if self.original_image is None:
            return None
        self.image = cv2.medianBlur(self.original_image.copy(), kernel_size)
        return self.image

    def gaussian_blur(self, kernel_size=3, sigma=1.0): #高斯滤波，默认核数为3，标准差为1.0
        if self.original_image is None:
            return None
        self.image = cv2.GaussianBlur(self.original_image.copy(), (kernel_size, kernel_size), sigma)
        return self.image

    # 图像锐化
    def laplacian_sharpen(self): #使用拉普拉斯算子进行锐化
        if self.original_image is None:
            return None
        self.image = cv2.Laplacian(self.original_image.copy(), cv2.CV_64F).astype(np.uint8)
        return self.image

    def sobel_horizontal(self): #使用Sobel算子进行锐化（水平sobel算子）
        if self.original_image is None:
                return None
        self.image = cv2.Sobel(self.original_image.copy(), cv2.CV_64F, 1, 0, ksize=3)
        self.image = cv2.convertScaleAbs(self.image)
        return self.image

    def sobel_vertical(self): #使用Sobel算子进行锐化（垂直sobel算子）
        if self.original_image is None:
            return None
        self.image = cv2.Sobel(self.original_image.copy(), cv2.CV_64F, 0, 1, ksize=3)
        self.image = cv2.convertScaleAbs(self.image)
        return self.image

    # 双线性插值放大
    def bilinear_interpolation(self, scale_factor=2.0):
        if self.original_image is None:
            return None
        height, width = self.original_image.shape[:2]
        self.image = cv2.resize(self.original_image.copy(), 
                              (int(width * scale_factor), int(height * scale_factor)), 
                              interpolation=cv2.INTER_LINEAR)
        return self.image

    # 图像平移
    def translate(self, x_shift=30, y_shift=50):   #默认平移30像素，50像素
        if self.original_image is None:
            return None
        rows, cols = self.original_image.shape[:2]
        M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        self.image = cv2.warpAffine(self.original_image.copy(), M, (cols, rows))
        return self.image

    # 旋转和缩放
    def rotate_scale(self, angle=45, scale=1.0):  #默认旋转45度，缩放1.0倍
        if self.original_image is None:
            return None
        rows, cols = self.original_image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
        self.image = cv2.warpAffine(self.original_image.copy(), M, (cols, rows))
        return self.image

    # 灰度转换和二值化
    def to_gray(self):
        if self.original_image is None:
            return None
        if len(self.original_image.shape) == 3:
            self.image = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_BGR2GRAY) #灰度图
        else:
            self.image = self.original_image.copy()
        return self.image

    def threshold_global(self, threshold=127):
        if self.original_image is None:
            return None
        gray = self.to_gray()
        _, self.image = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY) #二值化
        return self.image

    # 直方图相关
    def equalize_hist(self): #直方图均衡化，增强图像对比度
        if self.original_image is None:
            return None
        if len(self.original_image.shape) == 3:
            # 转换到YUV空间进行亮度均衡化
            yuv = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            self.image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            self.image = cv2.equalizeHist(self.original_image.copy())
        return self.image

    def show_histogram(self): #计算并可视化灰度直方图
        if self.original_image is None:
            return None, None
        gray = self.to_gray()
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_image = np.zeros((300, 256), dtype=np.uint8)
        cv2.normalize(hist, hist, 0, 300, cv2.NORM_MINMAX)
        for i in range(256):
            cv2.line(hist_image, (i, 300), (i, 300-int(hist[i])), 255)
        return hist.ravel(), hist_image #输出直方图和直方图图像

    # 仿射和透视变换
    def affine_transform(self, pts1=None, pts2=None): #进行仿射变换（保持平行性）
        if self.original_image is None:
            return None
        if pts1 is None or pts2 is None:
            rows, cols = self.original_image.shape[:2]
            pts1 = np.float32([[50,50], [200,50], [50,200]])
            pts2 = np.float32([[10,100], [200,50], [100,250]])
        M = cv2.getAffineTransform(pts1, pts2) #计算仿射变换矩阵
        self.image = cv2.warpAffine(self.original_image.copy(), M, (cols, rows)) #应用仿射变换
        return self.image

    def perspective_transform(self, pts1=None, pts2=None): #透视变换（可处理远近效果）
        if self.original_image is None:
            return None
        if pts1 is None or pts2 is None:
            rows, cols = self.original_image.shape[:2]
            pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
            pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
        M = cv2.getPerspectiveTransform(pts1, pts2) #计算透视变换矩阵
        self.image = cv2.warpPerspective(self.original_image.copy(), M, (300,300)) #应用透视变换
        return self.image

    # HSV相关操作
    def bgr_to_hsv(self): #BGR转换为HSV
        if self.original_image is None or len(self.original_image.shape) != 3:
            return None
        self.image = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_BGR2HSV)
        return self.image

    def get_hsv_h(self): #HSV通道分离 h通道
        if self.original_image is None or len(self.original_image.shape) != 3:
            return None
        hsv = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_BGR2HSV)
        self.image = hsv[:,:,0]
        return self.image

    def get_hsv_s(self): #HSV通道分离 s通道
        if self.original_image is None or len(self.original_image.shape) != 3:
                return None
        hsv = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_BGR2HSV)
        self.image = hsv[:,:,1]
        return self.image

    def get_hsv_v(self): #HSV通道分离 v通道
        if self.original_image is None or len(self.original_image.shape) != 3:
            return None
        hsv = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_BGR2HSV)
        self.image = hsv[:,:,2]
        return self.image

    # RGB通道分离
    def get_blue_channel(self): #RGB通道分离 b通道
        if self.original_image is None or len(self.original_image.shape) != 3:
            return None
        self.image = self.original_image.copy()[:,:,0]
        return self.image

    def get_green_channel(self): #RGB通道分离 g通道
        if self.original_image is None or len(self.original_image.shape) != 3:
            return None
        self.image = self.original_image.copy()[:,:,1]
        return self.image

    def get_red_channel(self): #RGB通道分离 r通道
        if self.original_image is None or len(self.original_image.shape) != 3:
            return None
        self.image = self.original_image.copy()[:,:,2]
        return self.image

    # 图像翻转
    def flip_horizontal(self): #水平翻转
        if self.original_image is None:
           return None
        self.image = cv2.flip(self.original_image.copy(), 1)
        return self.image

    def flip_vertical(self):   #垂直翻转
        if self.original_image is None:
            return None
        self.image = cv2.flip(self.original_image.copy(), 0)
        return self.image

    def flip_diagonal(self): #对角线翻转（水平加垂直）
        if self.original_image is None:
            return None
        self.image = cv2.flip(self.original_image.copy(), -1)
        return self.image

    # 形态学操作
    def morphology_open(self, kernel_size=3): #开运算
        if self.original_image is None:
            return None
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.image = cv2.morphologyEx(self.original_image.copy(), cv2.MORPH_OPEN, kernel)
        return self.image

    def morphology_close(self, kernel_size=3): #闭运算
        if self.original_image is None:
           return None
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.image = cv2.morphologyEx(self.original_image.copy(), cv2.MORPH_CLOSE, kernel)
        return self.image

    def erosion(self, kernel_size=3): #腐蚀
        if self.original_image is None:
            return None
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.image = cv2.erode(self.original_image.copy(), kernel, iterations=1)
        return self.image

    def dilation(self, kernel_size=3): #膨胀
        if self.original_image is None:
            return None
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.image = cv2.dilate(self.original_image.copy(), kernel, iterations=1)
        return self.image

    # 顶帽和底帽
    def top_hat(self, kernel_size=3): #顶帽(顶帽运算 = 原图像 - 开运算)
        if self.original_image is None:
            return None
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.image = cv2.morphologyEx(self.original_image.copy(), cv2.MORPH_TOPHAT, kernel)
        return self.image

    def black_hat(self, kernel_size=3): #底帽(底帽运算 = 原图像 - 闭运算)
        if self.original_image is None:
            return None
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.image = cv2.morphologyEx(self.original_image.copy(), cv2.MORPH_BLACKHAT, kernel)
        return self.image

    # 线条检测和边缘检测
    def hough_lines(self, threshold=100): #基于霍夫变换的直线检测
        if self.original_image is None:
            return None
        gray = self.to_gray()
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, minLineLength=100, maxLineGap=10)
        result = self.original_image.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        self.image = result
        return self.image

    def canny_edge(self, threshold1=100, threshold2=200): #基于Canny算子的边缘检测
        if self.original_image is None:
            return None
        gray = self.to_gray()
        self.image = cv2.Canny(gray, threshold1, threshold2)
        return self.image

    # 图像增强
    def image_enhancement(self, alpha=1.5, beta=30):
        if self.original_image is None:
            return None
        self.image = cv2.convertScaleAbs(self.original_image.copy(), alpha=alpha, beta=beta)
        return self.image

    # 边缘检测算子
    def roberts_edge(self): #Roberts算子
        if self.original_image is None:
            return None
        kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
        kernely = np.array([[0, -1], [1, 0]], dtype=int)
        gray = self.to_gray()
        x = cv2.filter2D(gray, cv2.CV_16S, kernelx)
        y = cv2.filter2D(gray, cv2.CV_16S, kernely)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        self.image = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        return self.image

    def prewitt_edge(self): #Prewitt算子
        if self.original_image is None:
            return None
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
        gray = self.to_gray()
        x = cv2.filter2D(gray, cv2.CV_16S, kernelx)
        y = cv2.filter2D(gray, cv2.CV_16S, kernely)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        self.image = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        return self.image

    def laplacian_edge(self): #拉普拉斯算子
        if self.original_image is None:
            return None
        gray = self.to_gray()
        self.image = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
        self.image = cv2.convertScaleAbs(self.image)
        return self.image

    def log_edge(self): #log算子
        if self.original_image is None:
            return None
        gray = self.to_gray()
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        self.image = cv2.Laplacian(blur, cv2.CV_16S, ksize=3)
        self.image = cv2.convertScaleAbs(self.image)
        return self.image