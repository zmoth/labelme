from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt

import os

from labelme.config import get_config
from labelme import utils


class FileListWidget(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # 启用自定义上下文菜单
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

        config = get_config()
        shortcuts = config["shortcuts"]

        # 创建右键菜单
        self.menu = QtWidgets.QMenu(self)
        self.copy_action = utils.newAction(
            self,
            self.tr("Copy"),
            self.copy_selected_items,
            None,
            "copy",
        )
        self.delete_action = utils.newAction(
            self,
            self.tr("Delete"),
            self.delete_selected_items,
            shortcuts["delete_polygon"],
            "cancel",
        )

        self.menu.addAction(self.copy_action)
        self.menu.addAction(self.delete_action)

    def show_context_menu(self, pos):
        # 显示菜单在鼠标位置
        self.menu.exec_(QtGui.QCursor.pos())

    def copy_selected_items(self):
        # 获取所有选中的项
        selected_items = self.selectedItems()
        if not selected_items:
            return

        # 拼接选中项的文本（多行用换行符分隔）
        text = "\n".join([item.text() for item in selected_items])

        # 复制到剪贴板
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText(text)

    def delete_selected_items(self):
        selected_items = self.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "错误", "未选中任何项！")
            return

        # 弹出确认对话框
        reply = QtWidgets.QMessageBox.question(
            self,
            "确认删除",
            "确定要删除选中的项及其对应的文件吗？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.No:
            return

        # 遍历删除选中的项及对应的文件
        for item in selected_items:
            image_path = item.text()
            if not image_path:
                continue  # 如果路径无效，跳过

            # 构造对应的JSON路径（假设JSON文件名与图片名相同，扩展名不同）
            base_path = os.path.splitext(image_path)[0]
            json_path = f"{base_path}.json"

            # 删除图片文件
            try:
                os.remove(image_path)
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "错误", f"删除图片文件失败：{str(e)}\n路径：{image_path}"
                )
                continue  # 继续处理其他项

            # 删除JSON文件
            if os.path.exists(json_path):
                try:
                    os.remove(json_path)
                except Exception as e:
                    QtWidgets.QMessageBox.warning(
                        self, "错误", f"删除JSON文件失败：{str(e)}\n路径：{json_path}"
                    )

            # 从列表中删除项
            row = self.row(item)
            self.takeItem(row)  # 安全删除项（避免索引混乱）

        QtWidgets.QMessageBox.information(
            self, "删除成功", "已删除选中的项及其对应的文件！"
        )
