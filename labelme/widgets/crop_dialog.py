import sys
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QApplication,
    QWidget,
    QStyleOption,
    QStyle,
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QRect, QPoint
from PIL import Image
import io


class CropLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing = False
        self.start_pos = QPoint()
        self.end_pos = QPoint()
        self.rect = QRect()  # 存储最终的裁剪区域

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_pos = event.pos()
            self.end_pos = self.start_pos
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.end_pos = event.pos()
            # 计算最终的矩形区域
            self.rect = QRect(self.start_pos, self.end_pos).normalized()
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.drawing or not self.rect.isNull():
            # 绘制虚线矩形框
            pen = QPen(QColor(255, 0, 0, 128), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)

            if self.drawing:
                rect = QRect(self.start_pos, self.end_pos).normalized()
                painter.drawRect(rect)
            else:
                painter.drawRect(self.rect)


class CropDialog(QDialog):
    def __init__(self, img: Image.Image, parent=None):
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Crop")
        self.original_image = img
        self.cropped_image = None
        self.crop_rect = QRect()

        # 将PIL.Image转换为QPixmap
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        qimage = QImage()
        qimage.loadFromData(buffer.getvalue(), "PNG")
        self.pixmap = QPixmap.fromImage(qimage)

        # 自定义的CropLabel用于显示图片和绘制矩形
        self.imageLabel = CropLabel()
        self.imageLabel.setPixmap(self.pixmap)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setScaledContents(True)  # 图片适应Label大小

        layout = QVBoxLayout()
        layout.addWidget(self.imageLabel)

        # 添加按钮
        btn_ok = QPushButton("另存为副本", self)
        btn_ok.clicked.connect(self.accept_crop)
        layout.addWidget(btn_ok)

        btn_cancel = QPushButton("取消", self)
        btn_cancel.clicked.connect(self.reject)
        layout.addWidget(btn_cancel)

        self.setLayout(layout)
        self.setFixedSize(800, 600)  # 设置窗口大小

    def accept_crop(self):
        label = self.imageLabel
        if label.rect.isNull():
            print("未选择裁剪区域")
            return

        # 计算原始图片的裁剪区域坐标
        label_size = label.size()
        original_size = self.original_image.size
        scale_w = original_size[0] / label_size.width()
        scale_h = original_size[1] / label_size.height()

        # 转换坐标到原始图片的坐标系
        x1 = int(label.rect.x() * scale_w)
        y1 = int(label.rect.y() * scale_h)
        x2 = int(label.rect.right() * scale_w)
        y2 = int(label.rect.bottom() * scale_h)

        # 确保坐标在有效范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(original_size[0], x2)
        y2 = min(original_size[1], y2)

        # 裁剪图片
        self.crop_rect = QRect(x1, y1, x2 - x1, y2 - y1)
        self.cropped_image = self.original_image.crop((x1, y1, x2, y2))

        self.accept()  # 关闭对话框
