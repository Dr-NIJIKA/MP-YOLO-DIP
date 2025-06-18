import onnxruntime as ort #ONNX模型推理库
import cv2
import numpy as np
import os
import sys
import traceback

class OracleRecognizer:
    """甲骨文识别器主类，包含模型加载、预处理、推理和后处理全流程"""
    def __init__(self):
        """初始化配置参数和模型路径"""
        self.model = None
        try:
            # 首先初始化配置
            if getattr(sys, 'frozen', False):
                # 使用EXE所在目录
                self.project_root = os.path.dirname(sys.executable)
                print(f"EXE模式 - 项目根目录: {self.project_root}")
            else:
                # 使用项目根目录
                self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                print(f"开发模式 - 项目根目录: {self.project_root}")
                
            # 确保models目录存在
            models_dir = os.path.join(self.project_root, 'models')
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
                print(f"创建models目录: {models_dir}")
            # 配置字典（包含模型路径、推理参数和预处理参数）
            self.configs = {
            'detection-model-path': os.path.join(self.project_root, 'models', 'detection-fp32.onnx'),  # 使用fp32模型
            'session-providers': ['CPUExecutionProvider'],#CPU
            'conf-threshold': 0.25,  # 检测置信度阈值
            'iou-threshold': 0.45,  # NMS的IOU阈值
            'precision': 'fp32',  # 与模型精度匹配
            'preprocessing': {
                'filter_type': 'bilateral',# 选择双边滤波或中值滤波
                'bilateral': {# 双边滤波参数
                    'd': 9,
                    'sigma_color': 75,
                    'sigma_space': 75
                },
                'median_ksize': 5,# 中值滤波核大小
                'clahe': {   # 自适应直方图均衡化参数
                    'clip_limit': 2.0,
                    'tile_size': 8
                },
                'morphology': { # 形态学操作参数
                    'kernel_size': 3
                }
            }
        }
            
            print(f"模型路径: {self.configs['detection-model-path']}")
            self._load_model() # 初始化时自动加载模型
            
        except Exception as e:
            print(f"初始化过程出错: {str(e)}")
            print(f"错误详情:\n{traceback.format_exc()}")
            raise

    def _load_model(self):
        """加载预训练模型"""
        try:
            model_path = self.configs['detection-model-path']
            print(f"当前运行模式: {'Frozen/Exe' if getattr(sys, 'frozen', False) else '开发环境'}")
            print(f"项目根目录: {self.project_root}")
            print(f"尝试加载模型: {model_path}")
             # 创建ONNX推理会话（指定执行提供器）
            if not os.path.exists(model_path):
                print(f"错误：模型文件不存在: {model_path}")
                print(f"当前目录内容: {os.listdir(os.path.dirname(model_path))}")
                self.model = None
                return
                
            self.model = ort.InferenceSession(
                model_path, 
                providers=self.configs['session-providers']
            )
            print(f"模型加载成功（精度：{self.configs['precision']}）")
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误详情: {str(e)}")
            self.model = None

    def letterbox(self, image, size, padding):
        """
        图像填充为正方形并保持比例缩放
        :param image: 输入图像
        :param size: 目标尺寸
        :param padding: 填充像素值
        :return: 处理后的图像
        """
        current_size = max(image.shape[0], image.shape[1])
        x1 = (current_size - image.shape[1]) >> 1
        y1 = (current_size - image.shape[0]) >> 1
        x2 = x1 + image.shape[1]
        y2 = y1 + image.shape[0]
        # 创建填充背景
        background = np.full((current_size, current_size, 3), padding, dtype=np.uint8)
        background[y1:y2, x1:x2] = image # 原图放入中心
        return cv2.resize(background, (size, size)) # 缩放到目标尺寸

    def preprocess(self, image):
        """模型输入预处理（归一化 和 通道转换）"""
        inputs = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose((2, 0, 1))# BGR转RGB
        inputs = inputs / 255.0  # 归一化[0,1]
        inputs = np.expand_dims(inputs, axis=0)  # 添加batch维度
        if self.configs['precision'] == 'fp16':  # 根据配置选择精度（fp16单独拿出去了）
            return inputs.astype(np.float16)
        else:
            return inputs.astype(np.float32)

    def get_valid_outputs(self, outputs):
        """过滤低置信度检测结果"""
        valid_outputs = outputs[outputs[:, 4] > self.configs['conf-threshold']]
        bboxes = valid_outputs[:, 0:4]  # 提取边界框(x_center,y_center,w,h)
        scores = valid_outputs[:, 4] # 提取置信度
        return bboxes.astype(np.int32), scores.astype(np.float32)

    def non_max_suppression(self, outputs):
        """非极大值抑制(NMS)去除冗余框"""
        bboxes, scores = self.get_valid_outputs(outputs)
        if len(bboxes) == 0:
            return []
        # 转换坐标格式：center -> corner
        bboxes[:, 0] -= bboxes[:, 2] >> 1 # x = x_center - width/2
        bboxes[:, 1] -= bboxes[:, 3] >> 1 # y = y_center - height/2

        # OpenCV NMS实现
        indices = cv2.dnn.NMSBoxes(
            bboxes.tolist(), scores.tolist(),
            self.configs['conf-threshold'],
            self.configs['iou-threshold'],
            eta=0.5
        )
        # 转换回原始坐标格式
        result = []
        if len(indices) > 0:
            for idx in indices.flatten():
                b = bboxes[idx]
                b[2] += b[0]
                b[3] += b[1]
                result.append(b)
        return result

    def detection_inference(self, image):
        """执行模型推理"""
        # 模型输入输出名称需与ONNX模型匹配
        outputs = self.model.run(['output0'], {'images': self.preprocess(image)})
        outputs = outputs[0]# 获取第一个输出
        outputs = outputs.squeeze().transpose()# 去除batch维度 转置为[n,5]格式
        return self.non_max_suppression(outputs)# 应用NMS

    def apply_filters(self, image):
        """应用选择的滤波方法"""
        if self.configs['preprocessing']['filter_type'] == 'median':
            return cv2.medianBlur(image, self.configs['preprocessing']['median_ksize'])
        else:
            return cv2.bilateralFilter(
                image,
                d=self.configs['preprocessing']['bilateral']['d'],
                sigmaColor=self.configs['preprocessing']['bilateral']['sigma_color'],
                sigmaSpace=self.configs['preprocessing']['bilateral']['sigma_space']
            )

    def apply_clahe(self, image):
        """自适应直方图均衡化(CLAHE)"""
        clahe = cv2.createCLAHE(
            clipLimit=self.configs['preprocessing']['clahe']['clip_limit'],
            tileGridSize=(self.configs['preprocessing']['clahe']['tile_size'],)*2
        )
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def apply_morphology(self, image):
        """形态学先开后闭运算"""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.configs['preprocessing']['morphology']['kernel_size'],)*2
        )
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    def paint_result(self, image, bbox, character_code=""):
        x1, y1, x2, y2 = bbox
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if character_code:
            cv2.putText(image, character_code, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        return image

    def recognize(self, image, save_path):
        try:
            # 检查模型是否加载成功
            if self.model is None:
                error_msg = "模型未加载，无法进行识别。请检查模型路径和依赖。"
                print(error_msg)
                return error_msg
                
            # 检查输入图像
            if image is None:
                error_msg = "输入图像为空"
                print(error_msg)
                return error_msg
                
            # 确保save_path的目录存在
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print(f"创建保存目录: {save_dir}")

            """
            完整识别流程：
            1. 图像预处理 -> 2. 推理 -> 3. 结果可视化 -> 4. 保存结果
            """    
            # 1. 预处理
            print("开始图像预处理...")
            processed = self.apply_filters(image)
            processed = self.apply_clahe(processed)
            processed = self.apply_morphology(processed)
            
            # 2. letterbox
            print("进行letterbox处理...")
            letterbox_image = self.letterbox(processed, size=640, padding=255)
            
            # 3. 检测
            print("执行模型推理...")
            bboxes = self.detection_inference(letterbox_image)
            
            # 4. 绘制结果
            print("绘制检测结果...")
            result_img = letterbox_image.copy()
            for bbox in bboxes:
                result_img = self.paint_result(result_img, bbox)
                
            # 5. 保存
            print(f"保存结果到: {save_path}")
            cv2.imwrite(save_path, result_img)
            
            # 6. 返回结果
            result_msg = f"检测到甲骨文字符数：{len(bboxes)}" if bboxes else "未检测到甲骨文字符"
            print(f"识别完成: {result_msg}")
            return result_msg
            
        except Exception as e:
            error_msg = f"识别过程出错: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return error_msg

