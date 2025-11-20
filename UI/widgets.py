import sys, torch, time, cv2, os
import numpy as np
from PIL import Image
from PyQt5.QtCore import (Qt, QPropertyAnimation, QRect, QEasingCurve,
                          pyqtSignal, QThread, QPoint)
from PyQt5.QtGui import (QPixmap, QImage, QFont, QIcon, QPalette, QColor,
                         QPainter, QBrush, QPen, QLinearGradient)
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QFileDialog, QVBoxLayout, QHBoxLayout,
                             QGridLayout, QTextEdit, QFrame, QGraphicsDropShadowEffect,
                             QProgressBar)

from .config import MODEL_PATH, ANIMAL_LABELS, TRANSFORMS, ICON_PATH, NUM_CLASSES, SAMPLE_EVERY
from .model import AnimalCNN
from .utils import scale_pixmap


# --------------------  置信度彩色进度条  --------------------
class ConfidenceBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._val = 0.0
        self.setFixedHeight(10)

    def setValue(self, v: float):
        # 符合 CNN：强制不封顶到 1.0
        self._val = max(0.0, min(0.9999, v))
        self.update()

    def paintEvent(self, evt):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        gradient = QLinearGradient(0, 0, w, 0)
        gradient.setColorAt(0.0, QColor("#ef4444"))
        gradient.setColorAt(0.5, QColor("#f59e0b"))
        gradient.setColorAt(1.0, QColor("#10b981"))
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, int(w * self._val), h, h // 2, h // 2)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor("#e5e7eb"), 2))
        painter.drawRoundedRect(0, 0, w, h, h // 2, h // 2)


# --------------------  视频工作线程  --------------------
class VideoWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, video_path, model, device):
        super().__init__()
        self.video_path = video_path
        self.model = model
        self.device = device

    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit("OpenCV 无法打开该视频文件")
                return
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            best_conf, best_label, best_frame = 0, '', None
            for idx in range(0, total, SAMPLE_EVERY):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                tensor = TRANSFORMS(pil_img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out = self.model(tensor)
                    prob = torch.nn.functional.softmax(out, dim=1)
                    conf, pred = torch.max(prob, 1)
                    conf = max(0.0, min(0.9999, conf.item()))   # 符合 CNN
                    label = ANIMAL_LABELS[pred.item()]
                if conf > best_conf:
                    best_conf, best_label, best_frame = conf, label, pil_img
                self.progress.emit(int(100 * (idx + 1) / total))
            cap.release()
            self.progress.emit(100)
            self.finished.emit({'label': best_label,
                                'conf': best_conf,
                                'frame': best_frame})
        except Exception as e:
            self.error.emit(str(e))


# --------------------  主界面  --------------------
class AnimalClassifierApp(QWidget):
    request_classify = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("动物识别系统")
        self.setWindowIcon(QIcon(ICON_PATH))
        self._pixmap = QPixmap()
        self.initUI()
        self.load_model()
        self.video_thread = None

    # ---------- UI ----------
    def initUI(self):
        self.showMaximized()
        self.setObjectName("mainWidget")
        self.setStyleSheet("""
            QWidget#mainWidget{
                background-color: #f3f4f6;
            }
            QLabel#title{
                color:#1f2937;
                font: 32pt "Microsoft YaHei";
                font-weight:bold;
            }
            QLabel#sub{
                color:#6b7280;
                font: 14pt "Microsoft YaHei";
            }
            QPushButton{
                border:none;
                padding:12px 28px;
                border-radius:10px;
                font:14px "Microsoft YaHei";
                font-weight:bold;
                color:white;
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 #38bdf8, stop:1 #0ea5e9);
            }
            QPushButton:hover{
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 #0ea5e9, stop:1 #0284c7);
            }
            QPushButton:pressed{
                padding:13px 27px 11px 29px;
            }
            QTextEdit{
                border:2px solid #e5e7eb;
                border-radius:12px;
                padding:12px;
                font:15px "Microsoft YaHei";
                color:#111827;
                background:#ffffff;
            }
            QFrame#card{
                background-color:rgba(255,255,255,230);
                border-radius:20px;
            }
            QProgressBar{
                border: 1px solid #bbb;
                border-radius:6px;
                text-align:center;
                height:10px;
                background:#e0e0e0;
            }
            QProgressBar::chunk{
                background-color:#10b981;
                border-radius:6px;
            }
        """)

        main = QVBoxLayout(self)
        main.setSpacing(18)
        main.setContentsMargins(30, 30, 30, 30)

        title = QLabel("动物识别系统")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        main.addWidget(title)
        subtitle = QLabel("基于深度学习的智能识别技术（支持图片 & 视频）")
        subtitle.setObjectName("sub")
        subtitle.setAlignment(Qt.AlignCenter)
        main.addWidget(subtitle)

        content = QHBoxLayout()
        content.setSpacing(25)

        # 左侧图片/视频卡片
        left_frame = QFrame()
        left_frame.setObjectName("card")
        left_layout = QVBoxLayout(left_frame)
        self.image_label = QLabel("点击上传图片或视频开始识别")
        self.image_label.setMinimumSize(360, 360)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel{
                border:3px dashed #d1d5db;
                border-radius:18px;
                color:#9ca3af;
                font-size:20px;
                font-weight:bold;
            }
            QLabel:hover{
                border-color:#38bdf8;
                color:#38bdf8;
            }
        """)
        shadow = QGraphicsDropShadowEffect(blurRadius=25, offset=QPoint(0, 8), color=QColor(0, 0, 0, 60))
        left_frame.setGraphicsEffect(shadow)
        left_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        content.addWidget(left_frame, stretch=3)

        # 右侧结果卡片
        right_frame = QFrame()
        right_frame.setObjectName("card")
        right_frame.setMaximumWidth(380)
        right_layout = QVBoxLayout(right_frame)
        res_title = QLabel("识别结果")
        res_title.setAlignment(Qt.AlignCenter)
        res_title.setFont(QFont("Microsoft YaHei", 20, QFont.Bold))
        right_layout.addWidget(res_title)

        self.result_label = QTextEdit("等待识别...")
        self.result_label.setReadOnly(True)
        right_layout.addWidget(self.result_label)

        self.conf_bar = ConfidenceBar()
        right_layout.addWidget(self.conf_bar)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)
        right_layout.addStretch()

        shadow2 = QGraphicsDropShadowEffect(blurRadius=25, offset=QPoint(0, 8), color=QColor(0, 0, 0, 60))
        right_frame.setGraphicsEffect(shadow2)
        content.addWidget(right_frame, stretch=2)
        main.addLayout(content)

        # 按钮区
        btn_frame = QFrame()
        btn_frame.setObjectName("card")
        btn_frame.setMaximumHeight(110)
        btn_layout = QHBoxLayout(btn_frame)
        btn_layout.setAlignment(Qt.AlignCenter)

        upload_img_btn = QPushButton("📷 上传图片")
        upload_img_btn.clicked.connect(self.load_image)
        upload_vid_btn = QPushButton("📹 上传视频")
        upload_vid_btn.clicked.connect(self.load_video)
        recognize_btn = QPushButton("🔍 开始识别")
        recognize_btn.clicked.connect(self.classify_image)
        exit_btn = QPushButton("❌ 退出系统")
        exit_btn.clicked.connect(self.close)

        for b in (upload_img_btn, upload_vid_btn, recognize_btn, exit_btn):
            b.setMinimumHeight(48)
            btn_layout.addWidget(b)
            btn_layout.addSpacing(20)
        main.addWidget(btn_frame)

        self.image_path = ""
        self.is_video = False
        self.best_frame = None

    # ---------- 模型 ----------
    def load_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AnimalCNN(NUM_CLASSES)
        ckpt_path = MODEL_PATH  # 默认 best 权重
        if not os.path.isfile(ckpt_path):
            self.result_label.setText("⚠️ 模型文件未找到，请检查路径！")
            return

        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        # ① 断点文件：提取模型权重  ② 纯权重：直接加载
        model_weights = state['model'] if 'model' in state else state
        self.model.load_state_dict(model_weights, strict=True)
        self.model.to(self.device).eval()

    # ---------- 上传 ----------
    def load_image(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "选择图片文件", "", "图片文件 (*.jpg *.jpeg *.png *.bmp *.gif)"
        )
        if file:
            self.image_path = file
            self.is_video = False
            self._pixmap = QPixmap(file)
            self._scale_image()
            self.result_label.setText("🔄 图片已上传，点击识别开始分析...")
            self.conf_bar.setValue(0)
            self.progress_bar.setVisible(False)

    def load_video(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        if file:
            self.image_path = file
            self.is_video = True
            cap = cv2.VideoCapture(file)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.best_frame = Image.fromarray(rgb)
                self._pixmap = QPixmap.fromImage(
                    QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)
                )
                self._scale_image()
            cap.release()
            self.result_label.setText("🔄 视频已上传，点击识别开始分析...")
            self.conf_bar.setValue(0)
            self.progress_bar.setVisible(False)

    def _scale_image(self):
        if self._pixmap.isNull():
            return
        scaled = scale_pixmap(self._pixmap,
                              self.image_label.width(),
                              self.image_label.height())
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_image()

    # ---------- 识别 ----------
    def classify_image(self):
        if not self.image_path:
            self.result_label.setText("⚠️ 请先上传图片或视频！")
            return
        if self.is_video:
            self.run_video_worker()
        else:
            self.run_image_inference()

    # 图片推理
    def run_image_inference(self):
        try:
            start = time.time()
            image = Image.open(self.image_path).convert('RGB')
            tensor = TRANSFORMS(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(tensor)
                prob = torch.nn.functional.softmax(out, dim=1)
                conf, pred = torch.max(prob, 1)
                conf = max(0.0, min(0.9999, conf.item()))  # ← 关键
                label = ANIMAL_LABELS[pred.item()]
            elapsed = time.time() - start
            self.result_label.setText(
                f"🎯 识别结果：{label}\n"
                f"📊 置信度：{conf:.4f} ({conf:.2%})\n"
                f"⏱️ 处理时间：{elapsed:.3f}s"
            )
            self.conf_bar.setValue(conf)
        except Exception as e:
            self.result_label.setText(f"❌ 识别失败：{e}")
            self.conf_bar.setValue(0)

    # 视频推理
    def run_video_worker(self):
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.result_label.setText("🎥 正在分析视频，请稍候...")
        if self.video_thread is not None and self.video_thread.isRunning():
            self.video_thread.terminate()
            self.video_thread.wait()
        self.video_thread = VideoWorker(self.image_path, self.model, self.device)
        self.video_thread.progress.connect(self.progress_bar.setValue)
        self.video_thread.finished.connect(self.on_video_finished)
        self.video_thread.error.connect(lambda e: self.result_label.setText(f"❌ 视频处理失败：{e}"))
        self.video_thread.start()

    def on_video_finished(self, data):
        label = data['label']
        conf = data['conf']
        frame = data['frame']
        if frame is not None:
            rgb = np.array(frame)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)
            self._pixmap = QPixmap.fromImage(qimg)
            self._scale_image()
        self.result_label.setText(
            f"🎯 视频识别结果：{label}\n"
            f"📊 最高置信度：{conf:.4f} ({conf:.2%})\n"
            f"⏱️ 处理完成"
        )
        self.conf_bar.setValue(conf)
        self.progress_bar.setVisible(False)