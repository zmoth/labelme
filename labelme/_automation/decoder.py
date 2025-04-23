import os
import os.path as osp
import cv2
import numpy as np
from PIL import Image
from PyQt5 import QtCore
from PyQt5.QtGui import QImage
from pylibdmtx.pylibdmtx import decode


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
model_path = osp.join(
    os.path.dirname(script_dir), "model", "yolov8n-barcode-keypoint.onnx"
)
model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(model_path)

CLASSES = ["datamatrix", "qrcode", "1d", "pdf417", "aztec"]


def convertQImageToMat(incomingImage):
    """Converts a QImage into an opencv MAT format"""

    incomingImage = incomingImage.convertToFormat(QImage.Format_RGB888)

    width = incomingImage.width()
    height = incomingImage.height()

    ptr = incomingImage.bits()
    ptr.setsize(incomingImage.byteCount())
    arr = np.array(ptr).reshape(height, width, 3)  #  Copies the data
    return arr


def yolo_barcode(image: QImage):
    original_image = convertQImageToMat(image)
    [height, width, _] = original_image.shape

    # Prepare a square image for inference
    length = max((height, width))
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
    outputs = model.forward()

    # Prepare output array
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []
    keypoints = []

    # Iterate through output to collect bounding boxes, confidence scores, and class IDs
    for i in range(rows):
        classes_scores = outputs[0][i][4:9]
        keypoint = outputs[0][i][9:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(
            classes_scores
        )
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)
            keypoints.append(keypoint)

    # Apply NMS (Non-maximum suppression)
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    points_list = []

    # Iterate through NMS results to draw bounding boxes and labels
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]

        if (
            keypoints[index][2] < 0.5
            or keypoints[index][5] < 0.5
            or keypoints[index][8] < 0.5
            or keypoints[index][11] < 0.5
        ):
            continue

        bottom_left = QtCore.QPoint(
            int(keypoints[index][0] * scale), int(keypoints[index][1] * scale)
        )
        top_left = QtCore.QPoint(
            int(keypoints[index][3] * scale), int(keypoints[index][4] * scale)
        )
        top_right = QtCore.QPoint(
            int(keypoints[index][6] * scale), int(keypoints[index][7] * scale)
        )
        bottom_right = QtCore.QPoint(
            int(keypoints[index][9] * scale), int(keypoints[index][10] * scale)
        )

        points_list.append([bottom_left, bottom_right, top_right, top_left])

    return points_list
