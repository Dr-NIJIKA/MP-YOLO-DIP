# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QGroupBox

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        # 设置主窗口
        MainWindow.setObjectName("MainWindow")
        # 获取屏幕尺寸
        screen = QtWidgets.QApplication.desktop().screenGeometry()
        width = int(screen.width() * 0.8)  # 窗口宽度为屏幕的80%
        height = int(screen.height() * 0.8)  # 窗口高度为屏幕的80%
        MainWindow.resize(width, height)
        MainWindow.setWindowTitle("数字图像处理系统")
        
        # 设置中心部件
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        # 创建主布局
        self.main_layout = QtWidgets.QHBoxLayout(self.centralwidget)
        
        # 创建左侧滚动区域用于按钮
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedWidth(400)  # 增加宽度从300到350
        
        # 创建滚动区域的内容部件
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setSpacing(12)  # 增加间距
        
        # 设置按钮样式
        button_style = """
            QPushButton {
                min-height: 40px;
                padding: 10px;
                font-size: 20px;
                font-weight: normal;
            }
        """
        
        # 设置分组框样式
        group_style = """
            QGroupBox {
                font-weight: bold;
                font-size: 20px;
                padding-top: 15px;
                margin-top: 15px;
            }
        """
        
        # 基本操作组
        self.basic_group = QGroupBox("基本操作")
        self.basic_group.setStyleSheet(group_style)
        basic_layout = QVBoxLayout()
        basic_layout.setSpacing(5)
        self.loadBtn = QtWidgets.QPushButton("打开图像")
        self.saveBtn = QtWidgets.QPushButton("保存图像")
        self.grayBtn = QtWidgets.QPushButton("转灰度图")
        self.binaryBtn = QtWidgets.QPushButton("二值化")
        
        for btn in [self.loadBtn, self.saveBtn, self.grayBtn, self.binaryBtn]:
            btn.setStyleSheet(button_style)
            basic_layout.addWidget(btn)
        basic_layout.setContentsMargins(10, 10, 10, 10)
        self.basic_group.setLayout(basic_layout)
        
        # 图像增强组
        self.enhance_group = QGroupBox("图像增强")
        self.enhance_group.setStyleSheet(group_style)
        enhance_layout = QVBoxLayout()
        enhance_layout.setSpacing(5)
        self.enhanceBtn = QtWidgets.QPushButton("图像增强")
        self.histogramBtn = QtWidgets.QPushButton("显示直方图")
        self.equalizeBtn = QtWidgets.QPushButton("直方图均衡化")
        for btn in [self.enhanceBtn, self.histogramBtn, self.equalizeBtn]:
            btn.setStyleSheet(button_style)
            enhance_layout.addWidget(btn)
        enhance_layout.setContentsMargins(10, 10, 10, 10)
        self.enhance_group.setLayout(enhance_layout)
        
        # 噪声和平滑组
        self.noise_group = QGroupBox("噪声和平滑")
        self.noise_group.setStyleSheet(group_style)
        noise_layout = QVBoxLayout()
        noise_layout.setSpacing(5)
        self.saltPepperBtn = QtWidgets.QPushButton("椒盐噪声")
        self.meanBlurBtn = QtWidgets.QPushButton("均值平滑")
        self.medianBlurBtn = QtWidgets.QPushButton("中值平滑")
        self.gaussianBlurBtn = QtWidgets.QPushButton("高斯平滑")
        for btn in [self.saltPepperBtn, self.meanBlurBtn, self.medianBlurBtn, self.gaussianBlurBtn]:
            btn.setStyleSheet(button_style)
            noise_layout.addWidget(btn)
        noise_layout.setContentsMargins(10, 10, 10, 10)
        self.noise_group.setLayout(noise_layout)
        
        # 图像锐化组
        self.sharpen_group = QGroupBox("图像锐化")
        self.sharpen_group.setStyleSheet(group_style)
        sharpen_layout = QVBoxLayout()
        sharpen_layout.setSpacing(5)
        self.laplacianSharpenBtn = QtWidgets.QPushButton("拉普拉斯锐化")
        self.sobelHBtn = QtWidgets.QPushButton("Sobel水平锐化")
        self.sobelVBtn = QtWidgets.QPushButton("Sobel垂直锐化")
        for btn in [self.laplacianSharpenBtn, self.sobelHBtn, self.sobelVBtn]:
            btn.setStyleSheet(button_style)
            sharpen_layout.addWidget(btn)
        sharpen_layout.setContentsMargins(10, 10, 10, 10)
        self.sharpen_group.setLayout(sharpen_layout)
        
        # 几何变换组
        self.geometry_group = QGroupBox("几何变换")
        self.geometry_group.setStyleSheet(group_style)
        geometry_layout = QVBoxLayout()
        geometry_layout.setSpacing(5)
        self.bilinearBtn = QtWidgets.QPushButton("双线性插值放大")
        self.translateBtn = QtWidgets.QPushButton("平移变换")
        self.rotateScaleBtn = QtWidgets.QPushButton("旋转和缩放")
        self.affineBtn = QtWidgets.QPushButton("仿射变换")
        self.perspectiveBtn = QtWidgets.QPushButton("透视变换")
        for btn in [self.bilinearBtn, self.translateBtn, self.rotateScaleBtn, 
                   self.affineBtn, self.perspectiveBtn]:
            btn.setStyleSheet(button_style)
            geometry_layout.addWidget(btn)
        geometry_layout.setContentsMargins(10, 10, 10, 10)
        self.geometry_group.setLayout(geometry_layout)
        
        # 颜色空间组
        self.color_group = QGroupBox("颜色空间")
        self.color_group.setStyleSheet(group_style)
        color_layout = QVBoxLayout()
        color_layout.setSpacing(5)
        self.rgb2hsvBtn = QtWidgets.QPushButton("RGB转HSV")
        self.hsvHBtn = QtWidgets.QPushButton("获取H通道")
        self.hsvSBtn = QtWidgets.QPushButton("获取S通道")
        self.hsvVBtn = QtWidgets.QPushButton("获取V通道")
        self.rgbBBtn = QtWidgets.QPushButton("获取B通道")
        self.rgbGBtn = QtWidgets.QPushButton("获取G通道")
        self.rgbRBtn = QtWidgets.QPushButton("获取R通道")
        for btn in [self.rgb2hsvBtn, self.hsvHBtn, self.hsvSBtn, self.hsvVBtn,
                   self.rgbBBtn, self.rgbGBtn, self.rgbRBtn]:
            btn.setStyleSheet(button_style)
            color_layout.addWidget(btn)
        color_layout.setContentsMargins(10, 10, 10, 10)
        self.color_group.setLayout(color_layout)
        
        # 翻转变换组
        self.flip_group = QGroupBox("翻转变换")
        self.flip_group.setStyleSheet(group_style)
        flip_layout = QVBoxLayout()
        flip_layout.setSpacing(5)
        self.flipHBtn = QtWidgets.QPushButton("水平翻转")
        self.flipVBtn = QtWidgets.QPushButton("垂直翻转")
        self.flipDBtn = QtWidgets.QPushButton("对角镜像")
        for btn in [self.flipHBtn, self.flipVBtn, self.flipDBtn]:
            btn.setStyleSheet(button_style)
            flip_layout.addWidget(btn)
        flip_layout.setContentsMargins(10, 10, 10, 10)
        self.flip_group.setLayout(flip_layout)
        
        # 形态学处理组
        self.morphology_group = QGroupBox("形态学处理")
        self.morphology_group.setStyleSheet(group_style)
        morphology_layout = QVBoxLayout()
        morphology_layout.setSpacing(5)
        self.openBtn = QtWidgets.QPushButton("开运算")
        self.closeBtn = QtWidgets.QPushButton("闭运算")
        self.erodeBtn = QtWidgets.QPushButton("腐蚀")
        self.dilateBtn = QtWidgets.QPushButton("膨胀")
        self.tophatBtn = QtWidgets.QPushButton("顶帽运算")
        self.blackhatBtn = QtWidgets.QPushButton("底帽运算")
        for btn in [self.openBtn, self.closeBtn, self.erodeBtn, self.dilateBtn,
                   self.tophatBtn, self.blackhatBtn]:
            btn.setStyleSheet(button_style)
            morphology_layout.addWidget(btn)
        morphology_layout.setContentsMargins(10, 10, 10, 10)
        self.morphology_group.setLayout(morphology_layout)
        
        # 边缘检测组
        self.edge_group = QGroupBox("边缘检测")
        self.edge_group.setStyleSheet(group_style)
        edge_layout = QVBoxLayout()
        edge_layout.setSpacing(5)
        self.houghBtn = QtWidgets.QPushButton("Hough线条检测")
        self.cannyBtn = QtWidgets.QPushButton("Canny边缘检测")
        self.robertsBtn = QtWidgets.QPushButton("Roberts边缘检测")
        self.prewittBtn = QtWidgets.QPushButton("Prewitt边缘检测")
        self.laplacianBtn = QtWidgets.QPushButton("Laplacian边缘检测")
        self.logBtn = QtWidgets.QPushButton("LoG边缘检测")
        for btn in [self.houghBtn, self.cannyBtn, self.robertsBtn, self.prewittBtn,
                   self.laplacianBtn, self.logBtn]:
            btn.setStyleSheet(button_style)
            edge_layout.addWidget(btn)
        edge_layout.setContentsMargins(10, 10, 10, 10)
        self.edge_group.setLayout(edge_layout)
        
        # 高级功能组
        self.advanced_group = QGroupBox("高级功能")
        self.advanced_group.setStyleSheet(group_style)
        advanced_layout = QVBoxLayout()
        advanced_layout.setSpacing(5)
        self.recognizeBtn = QtWidgets.QPushButton("甲骨文识别")
        self.startGestureBtn = QtWidgets.QPushButton("开始手势控制")
        self.stopGestureBtn = QtWidgets.QPushButton("停止手势控制")
        for btn in [self.recognizeBtn, self.startGestureBtn, self.stopGestureBtn]:
            btn.setStyleSheet(button_style)
            advanced_layout.addWidget(btn)
        advanced_layout.setContentsMargins(10, 10, 10, 10)
        self.advanced_group.setLayout(advanced_layout)
        
        # 将所有组添加到滚动区域
        self.scroll_layout.addWidget(self.basic_group)
        self.scroll_layout.addWidget(self.enhance_group)
        self.scroll_layout.addWidget(self.noise_group)
        self.scroll_layout.addWidget(self.sharpen_group)
        self.scroll_layout.addWidget(self.geometry_group)
        self.scroll_layout.addWidget(self.color_group)
        self.scroll_layout.addWidget(self.flip_group)
        self.scroll_layout.addWidget(self.morphology_group)
        self.scroll_layout.addWidget(self.edge_group)
        self.scroll_layout.addWidget(self.advanced_group)
        
        # 添加弹簧
        self.scroll_layout.addStretch()
        
        # 设置滚动区域的部件
        self.scroll_area.setWidget(self.scroll_content)
        
        # 创建图像显示区域
        self.display_layout = QtWidgets.QHBoxLayout()
        self.display_layout.setSpacing(20)  # 增加图像显示区域的间距
        
        # 原始图像显示
        self.PicBefore = QtWidgets.QLabel()
        self.PicBefore.setMinimumSize(400, 300)
        self.PicBefore.setAlignment(Qt.AlignCenter)
        self.PicBefore.setText("原始图像")
        self.PicBefore.setStyleSheet("""
            QLabel {
                border: 2px solid gray;
                background-color: white;
                font-size: 14px;
            }
        """)
        
        # 处理后图像显示
        self.PicAfter = QtWidgets.QLabel()
        self.PicAfter.setMinimumSize(400, 300)
        self.PicAfter.setAlignment(Qt.AlignCenter)
        self.PicAfter.setText("处理后图像")
        self.PicAfter.setStyleSheet("""
            QLabel {
                border: 2px solid gray;
                background-color: white;
                font-size: 14px;
            }
        """)
        
        self.display_layout.addWidget(self.PicBefore)
        self.display_layout.addWidget(self.PicAfter)
        
        # 将滚动区域和图像显示区域添加到主布局
        self.main_layout.addWidget(self.scroll_area)
        self.main_layout.addLayout(self.display_layout)
        
        # 设置中心部件
        MainWindow.setCentralWidget(self.centralwidget)
        
        # 创建状态栏
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "数字图像处理系统")) 