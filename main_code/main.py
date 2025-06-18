import sys
import cv2
import numpy as np
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, 
    QInputDialog, QDialog, QVBoxLayout, QPushButton, 
    QFormLayout, QSpinBox, QDoubleSpinBox, QDialogButtonBox,
    QComboBox, QLabel
)
import time
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from ui_mainwindow import Ui_MainWindow
from image_processor import ImageProcessor
from gesture_control import GestureController
from oracle_recognition import OracleRecognizer
import traceback

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # 初始化处理器
        self.image_processor = ImageProcessor() # 图像处理核心类
        self.gesture_controller = GestureController() # 手势控制类
        self.oracle_recognizer = OracleRecognizer() # 甲骨文识别类
        
        # 存储当前图像
        self.current_image = None
        self.processed_image = None
        
        # 连接信号和槽
        self.setup_connections()
        
    def setup_connections(self):
        # 基本操作
        self.ui.loadBtn.clicked.connect(self.open_image)
        self.ui.saveBtn.clicked.connect(self.save_image)
        self.ui.grayBtn.clicked.connect(self.to_gray)
        self.ui.binaryBtn.clicked.connect(self.threshold_global)

        # 图像增强
        self.ui.enhanceBtn.clicked.connect(self.image_enhancement)
        self.ui.histogramBtn.clicked.connect(self.show_histogram)
        self.ui.equalizeBtn.clicked.connect(self.equalize_hist)
        
        # 噪声和平滑
        self.ui.saltPepperBtn.clicked.connect(self.add_salt_pepper_noise)
        self.ui.meanBlurBtn.clicked.connect(self.mean_blur)
        self.ui.medianBlurBtn.clicked.connect(self.median_blur)
        self.ui.gaussianBlurBtn.clicked.connect(self.gaussian_blur)
        
        # 图像锐化
        self.ui.laplacianSharpenBtn.clicked.connect(self.laplacian_sharpen)
        self.ui.sobelHBtn.clicked.connect(self.sobel_horizontal)
        self.ui.sobelVBtn.clicked.connect(self.sobel_vertical)
        
        # 几何变换
        self.ui.bilinearBtn.clicked.connect(self.bilinear_interpolation)
        self.ui.translateBtn.clicked.connect(self.translate_image)
        self.ui.rotateScaleBtn.clicked.connect(self.rotate_scale_image)
        self.ui.affineBtn.clicked.connect(self.affine_transform)
        self.ui.perspectiveBtn.clicked.connect(self.perspective_transform)
        
        # 颜色空间
        self.ui.rgb2hsvBtn.clicked.connect(self.bgr_to_hsv)
        self.ui.hsvHBtn.clicked.connect(self.get_hsv_h)
        self.ui.hsvSBtn.clicked.connect(self.get_hsv_s)
        self.ui.hsvVBtn.clicked.connect(self.get_hsv_v)
        self.ui.rgbBBtn.clicked.connect(self.get_blue_channel)
        self.ui.rgbGBtn.clicked.connect(self.get_green_channel)
        self.ui.rgbRBtn.clicked.connect(self.get_red_channel)
        
        # 翻转变换
        self.ui.flipHBtn.clicked.connect(self.flip_horizontal)
        self.ui.flipVBtn.clicked.connect(self.flip_vertical)
        self.ui.flipDBtn.clicked.connect(self.flip_diagonal)
        
        # 形态学处理
        self.ui.openBtn.clicked.connect(self.morphology_open)
        self.ui.closeBtn.clicked.connect(self.morphology_close)
        self.ui.erodeBtn.clicked.connect(self.erosion)
        self.ui.dilateBtn.clicked.connect(self.dilation)
        self.ui.tophatBtn.clicked.connect(self.top_hat)
        self.ui.blackhatBtn.clicked.connect(self.black_hat)
        
        # 边缘检测
        self.ui.houghBtn.clicked.connect(self.hough_lines)
        self.ui.cannyBtn.clicked.connect(self.canny_edge)
        self.ui.robertsBtn.clicked.connect(self.roberts_edge)
        self.ui.prewittBtn.clicked.connect(self.prewitt_edge)
        self.ui.laplacianBtn.clicked.connect(self.laplacian_edge)
        self.ui.logBtn.clicked.connect(self.log_edge)
        
        # 高级功能
        self.ui.recognizeBtn.clicked.connect(self.recognize_oracle)
        self.ui.startGestureBtn.clicked.connect(self.start_gesture_control)
        self.ui.stopGestureBtn.clicked.connect(self.stop_gesture_control)
        
    def display_image(self, image, is_processed=True):
        """显示图像到UI界面"""
        if image is None:
            return
            
        try:
            # 转换图像格式
            if len(image.shape) == 2:  # 灰度图
                display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3:  # 彩色图
                if image.shape[2] == 3:
                    display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    display_image = image
            else:
                return
                
            # 创建QImage
            height, width, channel = display_image.shape
            bytes_per_line = channel * width
            q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # 创建QPixmap并设置到标签
            pixmap = QPixmap.fromImage(q_image)
            
            # 根据标签大小缩放图像
            if is_processed:
                label = self.ui.PicAfter
                self.processed_image = image.copy()
            else:
                label = self.ui.PicBefore
                self.current_image = image.copy()
                
            # 获取标签的大小
            label_size = label.size()
            # 保持纵横比缩放图像
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"显示图像失败: {str(e)}")

    #文件操作功能        
    def open_image(self):
        """打开图像文件"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择图像",
            "",
            "图像文件 (*.png *.jpg *.bmp *.jpeg);;所有文件 (*.*)"
        )
        
        if file_name:
            try:
                image = cv2.imread(file_name)
                if image is not None:
                    self.image_processor.load_image(image)
                    self.current_image = image.copy()
                    self.display_image(image, False)
                    self.processed_image = None
                    self.ui.PicAfter.clear()
                    self.ui.statusbar.showMessage("图像加载成功")
                else:
                    QMessageBox.warning(self, "错误", "无法加载图像")
            except Exception as e:
                QMessageBox.warning(self, "错误", str(e))
                
    def save_image(self):
        """保存处理后的图像"""
        if self.processed_image is None:
            QMessageBox.warning(self, "警告", "没有可保存的处理结果")
            return
            
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "保存图像",
            "",
            "PNG图像 (*.png);;JPEG图像 (*.jpg);;BMP图像 (*.bmp);;所有文件 (*.*)"
        )
        
        if file_name:
            try:
                cv2.imwrite(file_name, self.processed_image)
                self.ui.statusbar.showMessage("图像保存成功")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"保存图像失败: {str(e)}")

    def get_kernel_size(self, title="设置核大小", default=3, min_val=1, max_val=31):
        """获取核大小的通用对话框"""
        value, ok = QInputDialog.getInt(
            self, title, "请输入核大小（奇数）：",
            default, min_val, max_val, 2
        )
        if ok:
            return value if value % 2 == 1 else value + 1
        return None

    # 基本操作功能
    def to_gray(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        result = self.image_processor.to_gray()
        if result is not None:
            self.display_image(result)
            self.ui.statusbar.showMessage("灰度转换完成")

    def threshold_global(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        threshold, ok = QInputDialog.getInt(
            self, "二值化", "请输入阈值（0-255）：",
            127, 0, 255, 1
        )
        if ok:
            result = self.image_processor.threshold_global(threshold)
            if result is not None:
                self.display_image(result)
                self.ui.statusbar.showMessage(f"二值化完成（阈值：{threshold}）")

    def show_histogram(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        hist, hist_image = self.image_processor.show_histogram()
        if hist_image is not None:
            self.display_image(hist_image)
            self.ui.statusbar.showMessage("直方图显示成功")

    def equalize_hist(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        result = self.image_processor.equalize_hist()
        if result is not None:
            self.display_image(result)
            self.ui.statusbar.showMessage("直方图均衡化完成")

    # 噪声和平滑功能
    def add_salt_pepper_noise(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        prob, ok = QInputDialog.getDouble(
            self, "椒盐噪声", "请输入噪声比例（0-1）：",
            0.05, 0, 1, 2
        )
        if ok:
            result = self.image_processor.add_salt_pepper_noise(prob)
            if result is not None:
                self.display_image(result)
                self.ui.statusbar.showMessage(f"添加椒盐噪声完成（比例：{prob}）")

    def mean_blur(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        kernel_size = self.get_kernel_size("均值平滑")
        if kernel_size:
            result = self.image_processor.mean_blur(kernel_size)
            if result is not None:
                self.display_image(result)
                self.ui.statusbar.showMessage(f"均值平滑完成（核大小：{kernel_size}）")

    def median_blur(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        kernel_size = self.get_kernel_size("中值平滑")
        if kernel_size:
            result = self.image_processor.median_blur(kernel_size)
            if result is not None:
                self.display_image(result)
                self.ui.statusbar.showMessage(f"中值平滑完成（核大小：{kernel_size}）")

    def gaussian_blur(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("高斯平滑")
        layout = QFormLayout()
        
        kernel_size = QSpinBox()
        kernel_size.setRange(1, 31)
        kernel_size.setSingleStep(2)
        kernel_size.setValue(3)
        
        sigma = QDoubleSpinBox()
        sigma.setRange(0.1, 10.0)
        sigma.setSingleStep(0.1)
        sigma.setValue(1.0)
        
        layout.addRow("核大小:", kernel_size)
        layout.addRow("Sigma:", sigma)
        
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        layout.addRow(button_box)
        dialog.setLayout(layout)
        
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        if dialog.exec_() == QDialog.Accepted:
            size = kernel_size.value()
            if size % 2 == 0:
                size += 1
            result = self.image_processor.gaussian_blur(size, sigma.value())
            if result is not None:
                self.display_image(result)
                self.ui.statusbar.showMessage(f"高斯平滑完成（核大小：{size}，Sigma：{sigma.value()}）")

    # 图像锐化功能
    def laplacian_sharpen(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        result = self.image_processor.laplacian_sharpen()
        if result is not None:
            self.display_image(result)
            self.ui.statusbar.showMessage("拉普拉斯锐化完成")

    def sobel_horizontal(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        result = self.image_processor.sobel_horizontal()
        if result is not None:
            self.display_image(result)
            self.ui.statusbar.showMessage("Sobel水平锐化完成")

    def sobel_vertical(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        result = self.image_processor.sobel_vertical()
        if result is not None:
            self.display_image(result)
            self.ui.statusbar.showMessage("Sobel垂直锐化完成")

    # 几何变换功能
    def bilinear_interpolation(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        scale, ok = QInputDialog.getDouble(
            self, "双线性插值", "请输入放大倍数：",
            2.0, 0.1, 10.0, 1
        )
        if ok:
            result = self.image_processor.bilinear_interpolation(scale)
            if result is not None:
                self.display_image(result)
                self.ui.statusbar.showMessage(f"双线性插值放大完成（倍数：{scale}）")

    def translate_image(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("平移变换")
        layout = QFormLayout()
        
        x_shift = QSpinBox()
        x_shift.setRange(-500, 500)
        x_shift.setValue(30)
        
        y_shift = QSpinBox()
        y_shift.setRange(-500, 500)
        y_shift.setValue(50)
        
        layout.addRow("水平平移:", x_shift)
        layout.addRow("垂直平移:", y_shift)
        
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        layout.addRow(button_box)
        dialog.setLayout(layout)
        
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        if dialog.exec_() == QDialog.Accepted:
            result = self.image_processor.translate(x_shift.value(), y_shift.value())
            if result is not None:
                self.display_image(result)
                self.ui.statusbar.showMessage(f"平移变换完成（x：{x_shift.value()}，y：{y_shift.value()}）")

    def rotate_scale_image(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("旋转和缩放")
        layout = QFormLayout()
        
        angle = QSpinBox()
        angle.setRange(-360, 360)
        angle.setValue(45)
        
        scale = QDoubleSpinBox()
        scale.setRange(0.1, 10.0)
        scale.setSingleStep(0.1)
        scale.setValue(1.0)
        
        layout.addRow("旋转角度:", angle)
        layout.addRow("缩放比例:", scale)
        
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        layout.addRow(button_box)
        dialog.setLayout(layout)
        
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        if dialog.exec_() == QDialog.Accepted:
            result = self.image_processor.rotate_scale(angle.value(), scale.value())
            if result is not None:
                self.display_image(result)
                self.ui.statusbar.showMessage(f"旋转缩放完成（角度：{angle.value()}，比例：{scale.value()}）")

    def affine_transform(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        result = self.image_processor.affine_transform()
        if result is not None:
            self.display_image(result)
            self.ui.statusbar.showMessage("仿射变换完成")

    def perspective_transform(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        result = self.image_processor.perspective_transform()
        if result is not None:
            self.display_image(result)
            self.ui.statusbar.showMessage("透视变换完成")

    # 颜色空间功能
    def bgr_to_hsv(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        result = self.image_processor.bgr_to_hsv()
        if result is not None:
            self.display_image(result)
            self.ui.statusbar.showMessage("RGB转HSV完成")

    def get_hsv_h(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        result = self.image_processor.get_hsv_h()
        if result is not None:
            self.display_image(result)
            self.ui.statusbar.showMessage("获取H通道完成")

    def get_hsv_s(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        result = self.image_processor.get_hsv_s()
        if result is not None:
            self.display_image(result)
            self.ui.statusbar.showMessage("获取S通道完成")

    def get_hsv_v(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        result = self.image_processor.get_hsv_v()
        if result is not None:
            self.display_image(result)
            self.ui.statusbar.showMessage("获取V通道完成")

    def get_blue_channel(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        result = self.image_processor.get_blue_channel()
        if result is not None:
            self.display_image(result)
            self.ui.statusbar.showMessage("获取B通道完成")

    def get_green_channel(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        result = self.image_processor.get_green_channel()
        if result is not None:
            self.display_image(result)
            self.ui.statusbar.showMessage("获取G通道完成")

    def get_red_channel(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        result = self.image_processor.get_red_channel()
        if result is not None:
            self.display_image(result)
            self.ui.statusbar.showMessage("获取R通道完成")

    # 翻转变换功能
    def flip_horizontal(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        result = self.image_processor.flip_horizontal()
        if result is not None:
            self.display_image(result)
            self.ui.statusbar.showMessage("水平翻转完成")

    def flip_vertical(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        result = self.image_processor.flip_vertical()
        if result is not None:
            self.display_image(result)
            self.ui.statusbar.showMessage("垂直翻转完成")

    def flip_diagonal(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        result = self.image_processor.flip_diagonal()
        if result is not None:
            self.display_image(result)
            self.ui.statusbar.showMessage("对角镜像完成")

    # 形态学处理功能
    def morphology_open(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        kernel_size = self.get_kernel_size("开运算")
        if kernel_size:
            result = self.image_processor.morphology_open(kernel_size)
            if result is not None:
                self.display_image(result)
                self.ui.statusbar.showMessage(f"开运算完成（核大小：{kernel_size}）")

    def morphology_close(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        kernel_size = self.get_kernel_size("闭运算")
        if kernel_size:
            result = self.image_processor.morphology_close(kernel_size)
            if result is not None:
                self.display_image(result)
                self.ui.statusbar.showMessage(f"闭运算完成（核大小：{kernel_size}）")

    def erosion(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        kernel_size = self.get_kernel_size("腐蚀")
        if kernel_size:
            result = self.image_processor.erosion(kernel_size)
            if result is not None:
                self.display_image(result)
                self.ui.statusbar.showMessage(f"腐蚀完成（核大小：{kernel_size}）")

    def dilation(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        kernel_size = self.get_kernel_size("膨胀")
        if kernel_size:
            result = self.image_processor.dilation(kernel_size)
            if result is not None:
                self.display_image(result)
                self.ui.statusbar.showMessage(f"膨胀完成（核大小：{kernel_size}）")

    def top_hat(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        kernel_size = self.get_kernel_size("顶帽运算")
        if kernel_size:
            result = self.image_processor.top_hat(kernel_size)
            if result is not None:
                self.display_image(result)
                self.ui.statusbar.showMessage(f"顶帽运算完成（核大小：{kernel_size}）")

    def black_hat(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        kernel_size = self.get_kernel_size("底帽运算")
        if kernel_size:
            result = self.image_processor.black_hat(kernel_size)
            if result is not None:
                self.display_image(result)
                self.ui.statusbar.showMessage(f"底帽运算完成（核大小：{kernel_size}）")

    # 边缘检测功能
    def hough_lines(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        threshold, ok = QInputDialog.getInt(
            self, "Hough线条检测", "请输入阈值：",
            100, 1, 500, 1
        )
        if ok:
            result = self.image_processor.hough_lines(threshold)
            if result is not None:
                self.display_image(result)
                self.ui.statusbar.showMessage(f"Hough线条检测完成（阈值：{threshold}）")

    def canny_edge(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Canny边缘检测")
        layout = QFormLayout()
        
        threshold1 = QSpinBox()
        threshold1.setRange(0, 255)
        threshold1.setValue(100)
        
        threshold2 = QSpinBox()
        threshold2.setRange(0, 255)
        threshold2.setValue(200)
        
        layout.addRow("阈值1:", threshold1)
        layout.addRow("阈值2:", threshold2)
        
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        layout.addRow(button_box)
        dialog.setLayout(layout)
        
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        if dialog.exec_() == QDialog.Accepted:
            result = self.image_processor.canny_edge(threshold1.value(), threshold2.value())
            if result is not None:
                self.display_image(result)
                self.ui.statusbar.showMessage(f"Canny边缘检测完成（阈值：{threshold1.value()}, {threshold2.value()}）")

    def roberts_edge(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        result = self.image_processor.roberts_edge()
        if result is not None:
            self.display_image(result)
            self.ui.statusbar.showMessage("Roberts边缘检测完成")

    def prewitt_edge(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        result = self.image_processor.prewitt_edge()
        if result is not None:
            self.display_image(result)
            self.ui.statusbar.showMessage("Prewitt边缘检测完成")

    def laplacian_edge(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        result = self.image_processor.laplacian_edge()
        if result is not None:
            self.display_image(result)
            self.ui.statusbar.showMessage("Laplacian边缘检测完成")

    def log_edge(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        result = self.image_processor.log_edge()
        if result is not None:
            self.display_image(result)
            self.ui.statusbar.showMessage("LoG边缘检测完成")

    # 图像增强功能
    def image_enhancement(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("图像增强")
        layout = QFormLayout()
        
        alpha = QDoubleSpinBox()
        alpha.setRange(0.1, 3.0)
        alpha.setSingleStep(0.1)
        alpha.setValue(1.5)
        
        beta = QSpinBox()
        beta.setRange(-100, 100)
        beta.setValue(30)
        
        layout.addRow("对比度:", alpha)
        layout.addRow("亮度:", beta)
        
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        layout.addRow(button_box)
        dialog.setLayout(layout)
        
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        if dialog.exec_() == QDialog.Accepted:
            result = self.image_processor.image_enhancement(alpha.value(), beta.value())
            if result is not None:
                self.display_image(result)
                self.ui.statusbar.showMessage(f"图像增强完成（对比度：{alpha.value()}，亮度：{beta.value()}）")

    # 手势控制功能
    def start_gesture_control(self):
        """启动手势控制"""
        try:
            self.gesture_controller.start()
            self.ui.statusbar.showMessage("手势控制已启动")
        except Exception as e:
            QMessageBox.warning(self, "错误", str(e))

    def stop_gesture_control(self):
        """停止手势控制"""
        try:
            self.gesture_controller.stop()
            self.ui.statusbar.showMessage("手势控制已停止")
        except Exception as e:
            QMessageBox.warning(self, "错误", str(e))

    # 甲骨文识别功能
    def recognize_oracle(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图像")
            return
            
        try:
            # 使用当前处理后的图像或原始图像
            image_to_recognize = self.processed_image if self.processed_image is not None else self.current_image

            # 生成带时间戳的文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            result_filename = f"oracle_result_{timestamp}.jpg"
            
            # 确保result目录存在
            if getattr(sys, 'frozen', False):
                # EXE
                project_root = os.path.dirname(sys.executable)
            else:
                # 开发
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                
            result_dir = os.path.join(project_root, 'result')
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
                print(f"创建结果目录: {result_dir}")

            save_path = os.path.join(result_dir, result_filename)
            print(f"将保存结果到: {save_path}")
            
            # 调用识别器进行识别
            result_text = self.oracle_recognizer.recognize(image_to_recognize, save_path)
            
            if os.path.exists(save_path):
                # 读取保存的结果图像
                result_image = cv2.imread(save_path)
                if result_image is not None:
                    # 显示结果图像在界面上
                    self.display_image(result_image)
                    # 更新状态栏
                    self.ui.statusbar.showMessage(f"识别完成，结果已保存至：{save_path}")
                    # 显示识别结果文本
                    QMessageBox.information(self, "识别结果", result_text)
                else:
                    raise Exception(f"无法加载结果图像: {save_path}")
            else:
                raise Exception(f"结果图像未生成: {save_path}")
                
        except Exception as e:
            error_msg = f"识别过程出错：{str(e)}"
            print(error_msg)
            print(f"错误详情:\n{traceback.format_exc()}")
            QMessageBox.warning(self, "错误", error_msg)

if __name__ == '__main__':
    try:
        print("正在启动应用程序...")
        app = QApplication(sys.argv)
        print("正在创建主窗口...")
        window = MainWindow()
        print("正在显示主窗口...")
        window.show()
        print("应用程序启动完成，进入事件循环...")
        sys.exit(app.exec_())
    except Exception as e:
        print(f"程序运行出错：{str(e)}")
        sys.exit(1) 