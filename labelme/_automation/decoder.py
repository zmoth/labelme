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
