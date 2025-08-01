import os
import os.path as osp
import cv2
import numpy as np
from PIL import Image
from PyQt5 import QtCore
from PyQt5.QtGui import QImage
from pylibdmtx.pylibdmtx import decode

from labelme import utils


def _qimage_to_pil_image(qimage: QImage):
    # 获取图像的宽度和高度
    width = qimage.width()
    height = qimage.height()

    # 获取图像的格式
    format = qimage.format()

    if format == QImage.Format_RGB32:
        # 对于 RGB32 格式，转换为 RGBA
        qimage = qimage.convertToFormat(QImage.Format_RGBA8888)
        buffer = qimage.bits().asstring(width * height * 4)
        return Image.frombuffer("RGBA", (width, height), buffer, "raw", "RGBA", 0, 1)
    elif format == QImage.Format_ARGB32:
        # 对于 ARGB32 格式，直接转换为 RGBA
        buffer = qimage.bits().asstring(width * height * 4)
        return Image.frombuffer("RGBA", (width, height), buffer, "raw", "RGBA", 0, 1)
    elif format == QImage.Format_RGB888:
        # 对于 RGB888 格式，转换为 RGB
        buffer = qimage.bits().asstring(width * height * 3)
        return Image.frombuffer("RGB", (width, height), buffer, "raw", "RGB", 0, 1)
    else:
        # 对于其他格式，先转换为 RGB32 再处理
        qimage = qimage.convertToFormat(QImage.Format_RGB32)
        buffer = qimage.bits().asstring(width * height * 4)
        pil_image = Image.frombuffer(
            "RGBA", (width, height), buffer, "raw", "RGBA", 0, 1
        )
        return pil_image.convert("RGB")


def decode_barcode(image: QImage):
    results = decode(
        image=_qimage_to_pil_image(image),
    )

    points_list: list[list[QtCore.QPoint]] = []

    for result in results:
        points = result.rect
        bottom_left = QtCore.QPoint(
            points["00"]["x"], image.height() - points["00"]["y"]
        )
        top_left = QtCore.QPoint(points["01"]["x"], image.height() - points["01"]["y"])
        top_right = QtCore.QPoint(points["11"]["x"], image.height() - points["11"]["y"])
        bottom_right = QtCore.QPoint(
            points["10"]["x"], image.height() - points["10"]["y"]
        )
        points_list.append([bottom_left, bottom_right, top_right, top_left])
    return points_list


# 获取当前脚本所在的目录
script_dir = osp.dirname(osp.abspath(__file__))
# 构建模型文件的绝对路径
model_path = osp.join(os.path.dirname(script_dir), "model", "yolov8n-barcode-seg.onnx")
model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(model_path)

CLASSES = ["datamatrix", "qrcode", "1d", "pdf417", "aztec"]


def convertQImageToMat(image: QImage):
    """Converts a QImage into an opencv MAT format"""

    rgb_image = image.convertToFormat(QImage.Format_RGB888)

    width = rgb_image.width()
    height = rgb_image.height()

    ptr = rgb_image.bits()
    ptr.setsize(rgb_image.bytesPerLine() * height)
    arr = np.array(ptr).reshape(height, width, 3)
    return arr


def yolo_barcode(image: QImage):
    # rgb_image = image.convertToFormat(QImage.Format_BGR888)
    original_image = utils.img_qt_to_arr(image)
    [height, width, depth] = original_image.shape

    # Prepare a square image for inference
    length = max((height, width))
    if depth == 4:
        original_image = original_image[:, :, :3]  # 去掉 Alpha 通道
        depth = 3  # 更新通道数
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image

    # Calculate scale factor
    scale = length / 640

    # Preprocess the image and prepare blob for model
    blob = cv2.dnn.blobFromImage(
        image, scalefactor=1 / 255, size=(640, 640), swapRB=True
    )
    model.setInput(blob)

    # Perform inference
    output0, output1 = model.forward(["output0", "output1"])

    # 获取检测框数量
    rows = output0.shape[2]

    boxes = []
    scores = []
    class_ids = []
    masks = []

    # 遍历所有检测框
    for i in range(rows):
        # 提取检测框信息：x, y, w, h
        x = output0[0, 0, i]
        y = output0[0, 1, i]
        w = output0[0, 2, i]
        h = output0[0, 3, i]

        # 提取对象置信度和类别得分
        class_scores = output0[0, 4:9, i]  # 假设模型为80类

        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(
            class_scores
        )

        if maxScore >= 0.25:
            # 转换为左上角坐标 + 宽高
            box = [x - w / 2, y - h / 2, w, h]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

            # 生成分割掩码（简化版，具体逻辑取决于模型结构）
            seg_params = output0[0, 9:41, i]
            feature_map = output1[0]  # shape: [32, 160, 160]
            seg_mask = np.dot(seg_params, feature_map.reshape(32, -1))
            seg_mask = np.reshape(seg_mask, (160, 160))
            masks.append(seg_mask)

    # Apply NMS (Non-maximum suppression)
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.5)

    points_list = []

    # Iterate through NMS results to draw bounding boxes and labels
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        mask = masks[index]

        # 1. 归一化与二值化
        mask_normalized = cv2.normalize(
            mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1
        )
        _, mask_binary = cv2.threshold(mask_normalized, 127, 255, cv2.THRESH_BINARY)

        # 2. 尺寸调整
        mask_resized = cv2.resize(mask_binary, (width, height))

        # 3. 提取轮廓
        contours, _ = cv2.findContours(
            mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 4. 筛选并提取轮廓点
        max_contour = None
        max_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        if max_contour is None:
            continue

        # 转换为二维点列表
        points = max_contour.reshape(-1, 2).tolist()

        # # 转换为 QPoint 列表
        points_list.append([QtCore.QPoint(int(x), int(y)) for x, y in points])

    return points_list
