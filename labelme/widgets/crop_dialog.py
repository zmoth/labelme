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
        self.rect = QRect()  # 存储最终的裁剪区域（在原始pixmap坐标系中）
        self.scale_factor = 1.0  # 当前缩放比例
        self.pixmap = None  # 原始pixmap
        self.scaled_pixmap = None  # 缩放后的pixmap
        self.offset = QPoint(0, 0)  # 图片偏移量（用于平移）
        self.last_pan_pos = QPoint()  # 上次平移位置

    def setPixmap(self, pixmap):
        self.pixmap = pixmap
        self.update_scaled_pixmap()

    def update_scaled_pixmap(self):
        if self.pixmap:
            self.scaled_pixmap = self.pixmap.scaled(
                self.pixmap.size() * self.scale_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.update()

    def wheelEvent(self, event):
        # 计算鼠标位置在图片上的坐标（考虑当前偏移和缩放）
        mouse_pos = event.pos() - self.offset
        # 计算缩放前的图片坐标比例
        if self.scaled_pixmap:
            img_x = mouse_pos.x() / self.scaled_pixmap.width()
            img_y = mouse_pos.y() / self.scaled_pixmap.height()

            # 调整缩放比例
            factor = 1.1
            if event.angleDelta().y() > 0:
                # 放大
                self.scale_factor *= factor
            else:
                # 缩小
                self.scale_factor /= factor

            # 限制缩放范围
            self.scale_factor = max(0.1, min(self.scale_factor, 10.0))

            # 更新缩放后的pixmap
            self.update_scaled_pixmap()

            # 调整偏移量，使鼠标下的点保持在相同位置
            if self.scaled_pixmap:
                new_img_pos = QPoint(
                    int(img_x * self.scaled_pixmap.width()),
                    int(img_y * self.scaled_pixmap.height())
                )
                self.offset = event.pos() - new_img_pos
                self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_pos = self.map_to_pixmap(event.pos())
            self.end_pos = self.start_pos
            self.update()
        elif event.button() == Qt.MiddleButton:
            # 中键按下开始平移
            self.last_pan_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_pos = self.map_to_pixmap(event.pos())
            self.update()
        elif event.buttons() & Qt.MiddleButton:
            # 计算平移距离
            delta = event.pos() - self.last_pan_pos
            self.offset += delta
            self.last_pan_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.end_pos = self.map_to_pixmap(event.pos())
            # 计算最终的矩形区域（在原始pixmap坐标系中）
            self.rect = QRect(self.start_pos, self.end_pos).normalized()
            self.update()

    def map_to_pixmap(self, pos):
        """将视图坐标转换为原始pixmap坐标"""
        if not self.scaled_pixmap:
            return QPoint()

        # 1. 减去偏移量
        adjusted_pos = pos - self.offset

        # 2. 计算在scaled_pixmap中的比例位置
        pixmap_x = adjusted_pos.x() / self.scaled_pixmap.width()
        pixmap_y = adjusted_pos.y() / self.scaled_pixmap.height()

        # 3. 转换为原始pixmap坐标
        original_x = pixmap_x * self.pixmap.width()
        original_y = pixmap_y * self.pixmap.height()

        return QPoint(int(original_x), int(original_y))

    def map_from_pixmap(self, point):
        """将原始pixmap坐标转换为视图坐标（用于绘制）"""
        if not self.scaled_pixmap:
            return QPoint()

        # 1. 计算在scaled_pixmap中的位置
        pixmap_x = point.x() / self.pixmap.width() * self.scaled_pixmap.width()
        pixmap_y = point.y() / self.pixmap.height() * self.scaled_pixmap.height()

        # 2. 加上偏移量
        view_x = pixmap_x + self.offset.x()
        view_y = pixmap_y + self.offset.y()

        return QPoint(int(view_x), int(view_y))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # 绘制背景（棋盘格）
        self.draw_background(painter)

        # 绘制图片（考虑偏移和缩放）
        if self.scaled_pixmap:
            painter.translate(self.offset)
            painter.drawPixmap(0, 0, self.scaled_pixmap)
            painter.translate(-self.offset)

        # 绘制裁剪框
        if self.drawing or not self.rect.isNull():
            pen = QPen(QColor(255, 0, 0, 128), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)

            if self.drawing and self.start_pos and self.end_pos:
                # 绘制实时矩形
                top_left = self.map_from_pixmap(self.start_pos)
                bottom_right = self.map_from_pixmap(self.end_pos)
                painter.drawRect(QRect(top_left, bottom_right).normalized())
            elif not self.rect.isNull():
                # 绘制最终矩形
                top_left = self.map_from_pixmap(self.rect.topLeft())
                bottom_right = self.map_from_pixmap(self.rect.bottomRight())
                painter.drawRect(QRect(top_left, bottom_right).normalized())

    def draw_background(self, painter):
        """绘制棋盘格背景"""
        tile_size = 20
        color1 = QColor(240, 240, 240)
        color2 = QColor(200, 200, 200)

        for x in range(0, self.width(), tile_size):
            for y in range(0, self.height(), tile_size):
                painter.fillRect(
                    x, y, tile_size, tile_size,
                    color1 if (x + y) // tile_size % 2 == 0 else color2
                )


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
        self.resize(800, 600)  # 设置窗口初始大小

    def accept_crop(self):
        label = self.imageLabel
        if label.rect.isNull():
            print("未选择裁剪区域")
            return

        # 直接使用label.rect中的坐标（已经在原始pixmap坐标系中）
        rect = label.rect

        # 确保坐标在有效范围内
        x1 = max(0, rect.left())
        y1 = max(0, rect.top())
        x2 = min(self.original_image.width, rect.right())
        y2 = min(self.original_image.height, rect.bottom())

        # 裁剪图片
        self.crop_rect = QRect(x1, y1, x2 - x1, y2 - y1)
        self.cropped_image = self.original_image.crop((x1, y1, x2, y2))

        self.accept()
